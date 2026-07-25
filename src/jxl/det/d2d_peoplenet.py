"""PeopleNet (DetectNet_v2 + ResNet34) 2D 目标检测器 — jxl 适配。

用 onnxruntime 加载 NGC 上的 ``resnet34_peoplenet_int8.onnx``
(decrypted deployable, 标准算子, onnxruntime 可直接推理),
手写 DetectNet_v2 GridBox 后处理 (grid decode + DBSCAN clustering)。

后处理参考 NVIDIA 官方实现
``tao-toolkit-triton-apps/tao_triton/python/postprocessing/detectnet_processor.py``。

模型规格 (来自 NGC ``nvinfer_config.txt``):
- 输入: RGB 960x544, NCHW, scale 1/255, 无 mean 减
- 输出 cov: ``[B, 3, 34, 60]`` 已 Sigmoid 的逐 grid 置信度
- 输出 bbox: ``[B, 12, 34, 60]`` = 4 坐标 x 3 类, GridBox 回归
- 类别: person(0) / bag(1) / face(2)
- grid: 34(H) x 60(W), stride 16

注: onnxruntime 仅做推理 + 后处理; 后处理 (decode + DBSCAN) 不在 ONNX 图内,
DetectNet_v2 的 GridBox clustering 必须由调用方实现 (本模块即此实现)。
"""

from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

from jxl.det.d2d import D2dObject, D2dOpt, D2dResult, Detector2D


class PeopleNetClass(IntEnum):
    """PeopleNet 检测类别 (DetectNet_v2, 单一数据源)。"""

    PERSON = 0
    BAG = 1
    FACE = 2


# ---- 模型输入规格 (NGC PeopleNet nvinfer_config.txt) ----
INPUT_W: int = 960
INPUT_H: int = 544
NUM_CLASSES: int = len(PeopleNetClass)
NET_SCALE: float = 1.0 / 255.0

# ---- DetectNet_v2 GridBox decode 常量 ----
STRIDE: int = 16
OFFSET: float = 0.5
BBOX_NORM: float = 35.0
GRID_W: int = INPUT_W // STRIDE  # 60
GRID_H: int = INPUT_H // STRIDE  # 34

# 预计算 grid cell 中心 (像素, model 空间); cov/bbox 为 [C, H=34, W=60]。
# setflags(write=False): 常量数组不可变, 误写抛 RuntimeError (不可变优先)。
_GC_X = np.arange(GRID_W, dtype=np.float32) * STRIDE + OFFSET  # [60]
_GC_Y = np.arange(GRID_H, dtype=np.float32) * STRIDE + OFFSET  # [34]
_GC_X.setflags(write=False)
_GC_Y.setflags(write=False)

# ---- DBSCAN clustering (DetectNet_v2 / PeopleNet spec 固定值, 不可调) ----
CLUSTER_CONF_THR: float = 1.0  # 聚类后 aggregated coverage 下限 (spec 固定)

# ONNX 输入输出张量名 (验证过的标准 DetectNet_v2 graph)
_IN_NAME = "input_1:0"
_OUT_COV = "output_cov/Sigmoid:0"
_OUT_BBOX = "output_bbox/BiasAdd:0"

ByteImg = NDArray[np.uint8]
F32 = NDArray[np.float32]


def _preprocess(bgr_u8_hwc: ByteImg) -> F32:
    """BGR HWC uint8 → RGB 960x544, ×1/255, NCHW float32。

    maintain_aspect_ratio=0 (直接 resize, 不保比/不 letterbox)。
    """
    rgb = cv2.cvtColor(bgr_u8_hwc, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    out = resized.astype(np.float32) * NET_SCALE
    return np.ascontiguousarray(out.transpose(2, 0, 1))[None]  # 1,3,H,W


def _iou_matrix(rects_ltrb: F32) -> F32:
    """N 个 LTRB 框的两两 IoU 矩阵 [N,N]; 输入 [N,4] model 空间像素。"""
    left, t, r, b = rects_ltrb.T
    il = np.maximum(left[:, None], left[None, :])
    it = np.maximum(t[:, None], t[None, :])
    ir = np.minimum(r[:, None], r[None, :])
    ib = np.minimum(b[:, None], b[None, :])
    iw = np.clip(ir - il, 0, None)
    ih = np.clip(ib - it, 0, None)
    inter = iw * ih
    area = (r - left) * (b - t)
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _decode_class_bbox(bbox_c: F32) -> tuple[F32, F32, F32, F32]:
    """单类 bbox [4, GRID_H, GRID_W] → model 空间 LTRB (各 [GRID_H, GRID_W])。

    DetectNet_v2 GridBox decode:
        x1 = (col*stride + offset) - a*bbox_norm
        y1 = (row*stride + offset) - b*bbox_norm
        x2 = (col*stride + offset) + c*bbox_norm
        y2 = (row*stride + offset) + d*bbox_norm
    """
    x1 = np.clip(_GC_X[None, :] - bbox_c[0] * BBOX_NORM, 0, INPUT_W)
    y1 = np.clip(_GC_Y[:, None] - bbox_c[1] * BBOX_NORM, 0, INPUT_H)
    x2 = np.clip(_GC_X[None, :] + bbox_c[2] * BBOX_NORM, 0, INPUT_W)
    y2 = np.clip(_GC_Y[:, None] + bbox_c[3] * BBOX_NORM, 0, INPUT_H)
    return x1, y1, x2, y2


def _cluster_one_class(
    cov_c: F32,
    x1: F32,
    y1: F32,
    x2: F32,
    y2: F32,
    conf_thr: float,
    eps: float,
) -> list[tuple[float, float, float, float, float]]:
    """单类: coverage 阈值 + DBSCAN(1-IoU, sample_weight=coverage) 聚类。

    返回 [(score, x1, y1, x2, y2), ...], 坐标为 model 空间像素。
    """
    mask = cov_c > conf_thr
    if not np.any(mask):
        return []
    covs = cov_c[mask]
    rects = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1).astype(
        np.float32
    )
    dist = 1.0 - _iou_matrix(rects)
    labels = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit_predict(
        dist, sample_weight=covs
    )
    result: list[tuple[float, float, float, float, float]] = []
    for lab in set(labels.tolist()):
        if lab < 0:
            continue
        members = labels == lab
        w = covs[members]
        agg = float(w.sum())
        if agg < CLUSTER_CONF_THR:
            continue
        wn = w / agg
        mean = (rects[members] * wn[:, None]).sum(axis=0)
        result.append(
            (agg, float(mean[0]), float(mean[1]), float(mean[2]), float(mean[3]))
        )
    return result


def _postprocess(
    cov: F32,
    bbox: F32,
    orig_w: int,
    orig_h: int,
    conf_thr: float,
    iou_thr: float,
) -> list[D2dObject]:
    """cov [3,34,60] + bbox [12,34,60] → list[D2dObject] (归一化 rect)。

    纯函数: decode → 逐类阈值+DBSCAN → 缩放到原图 → 归一化 LTRB。
    """
    sx = orig_w / INPUT_W
    sy = orig_h / INPUT_H
    eps = max(0.0, 1.0 - iou_thr)
    objects: list[D2dObject] = []
    oid = 0
    for c in range(NUM_CLASSES):
        x1, y1, x2, y2 = _decode_class_bbox(bbox[c * 4 : (c + 1) * 4])
        for score, bx1, by1, bx2, by2 in _cluster_one_class(
            cov[c], x1, y1, x2, y2, conf_thr, eps
        ):
            # model 空间 → 原图像素 → 归一化 LTRB
            left = bx1 * sx / orig_w
            t = by1 * sy / orig_h
            r = bx2 * sx / orig_w
            b = by2 * sy / orig_h
            objects.append(
                D2dObject(
                    id=oid,
                    cls=c,
                    conf=min(score, 1.0),
                    rect=Rect.from_ltrb_list([left, t, r, b]),
                )
            )
            oid += 1
    return objects


# ORT provider / device 标识 (单一数据源, 避免字面量漂移)
_CUDA_EP = "CUDAExecutionProvider"
_CPU_DEVICE = "cpu"


def _providers_for(device_name: str) -> list[str]:
    """device_name → onnxruntime providers 列表。

    device_name == _CPU_DEVICE 显式走 CPU（保留 ml-peoplenet 数值基准的对照能力）;
    其余（"cuda:0" / "" 等）走 GPU。
    """
    if device_name == _CPU_DEVICE:
        return ["CPUExecutionProvider"]
    return [_CUDA_EP]


class D2dPeopleNet(Detector2D):
    """PeopleNet 2D 检测器 (DetectNet_v2 + ResNet34)。

    onnxruntime 推理 + 手写 GridBox 后处理。默认 GPU (CUDAExecutionProvider);
    device_name="cpu" 显式走 CPU。GPU 不可用时立即报错, 不静默回退 CPU
    （No Silent Degradation, 对齐 jxl/bin/person_embed.py 的 device 校验惯例）。
    """

    model_class = "D2dPeopleNet"

    def __init__(
        self,
        model_path: Path,
        opt: D2dOpt,
        device_name: str = "",
        verbose: bool = False,
    ) -> None:
        super().__init__(model_path, opt, device_name, verbose)
        providers = _providers_for(device_name)
        self._sess = ort.InferenceSession(str(model_path), providers=providers)
        # No Silent Degradation: 要求 GPU 时校验 CUDAExecutionProvider 实际生效。
        if device_name != _CPU_DEVICE and _CUDA_EP not in self._sess.get_providers():
            raise RuntimeError(
                f"要求 GPU 推理但 {_CUDA_EP} 未生效 "
                f"(actual={self._sess.get_providers()}); "
                f"确认已安装 onnxruntime-gpu 且 CUDA/cuDNN 可用"
            )

    def detect(self, image: ImageNda) -> D2dResult:
        """对单张图像执行 PeopleNet 检测。

        流程: BGR→RGB → resize 960x544 → ×1/255 → NCHW → 推理 →
        GridBox decode → 逐类 DBSCAN → 归一化 D2dObject。
        """
        src = np.asarray(image.data(), dtype=np.uint8)
        orig_h, orig_w = int(src.shape[0]), int(src.shape[1])
        inp = _preprocess(src)
        cov_raw, bbox_raw = self._sess.run([_OUT_COV, _OUT_BBOX], {_IN_NAME: inp})
        # onnxruntime 无 stub, 输出推断为 Any; np.asarray 收口类型 + 断言 float32 dtype。
        cov = np.asarray(cov_raw, dtype=np.float32)
        bbox = np.asarray(bbox_raw, dtype=np.float32)
        objects = _postprocess(
            cov[0], bbox[0], orig_w, orig_h, self._opt.conf_thr, self._opt.iou_thr
        )
        return D2dResult(objects=objects)
