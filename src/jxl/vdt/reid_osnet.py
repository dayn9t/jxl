"""OSNet（BN 家族）ReID 特征——``Embedder`` 协议第二实现（自 iapx 移交，P2-T7）。

``Embedder`` 协议单一数据源在 :mod:`jxl.vdt.reid_assoc`；第一实现 =
:mod:`jxl.vdt.reid` 的 ``ReidEmbedder``（DINOv2 ViT-S/14 ONNX）。本模块自
iapx ``pipeline/osnet.py`` 迁移（iapx 解散移交，2026-09-19），加载与推理
逻辑原样，仅做协议适配：

- **协议适配（唯一语义改动）**：iapx 批量接口 ``embed(crops: list) ->
  list[list[float]]``（``.tolist()`` 供其 cache 序列化）改为协议单 crop 接口
  ``embed(crop) -> np.ndarray``（512-d L2 归一化 ``float32``）；退化 crop 按
  协议契约返回**全零哨兵**（``associate`` 见零范数即 ``id=0`` 不匹配不新建；
  iapx 原批量入口此处 raise ValueError）。批量语义保留为 ``embed_batch``
  （离线重嵌入批量 GPU 工作负载入口，fail-loud 语义原样，输出 ndarray 化）。

契约与 Rust ``OsnetReid`` 对齐：输入 256×128、ImageNet 归一化、512-d 输出
运行时 L2 归一化、无 TTA（不翻转）。

模型定义 = ``jxl.vdt.vendored.osnet``（KaiyangZhou/deep-person-reid MIT
vendored，勿改）；x1_0 与 x0_75 同 BN 家族、同定义文件，tag → (vendor 工厂,
权重路径) 映射见 ``OSNET_WEIGHTS``（模型指纹单一数据源）。权重 = MSMT17
combineall 训练（分类头 4101 类，加载后剥除 → 512-d 特征）。spec/plan 写的
AIN 变体系笔误，实证为 BN：钦定权重文件名无 ``ain`` 前缀（torchreid
MODEL_ZOO 命名法）且 state_dict 零 InstanceNorm 键；AIN 定义对 x0.75 权重
strict load 440 missing / 455 unexpected，BN 定义 0/0 直载；x1_0 权重同法
验证 0/0（2026-09-12，源 = 作者 HF 镜像 kaiyangzhou/osnet）。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch

from jxl.vdt.vendored.osnet import osnet_x0_75, osnet_x1_0

OSNET_INPUT_HW = (256, 128)
"""网络输入 (H, W)——MSMT17 训练分辨率（taller-than-wide 人像）。"""

OSNET_DIM = 512
"""特征维度（fc 层输出；两变体 backbone 384/512 → fc 统一升维 512）。"""

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

WEIGHTS_DIR = Path("/mnt/data/jiang/ws/sgcc/person/osnet_weights")
"""权重目录（n001 数据盘；jxl/sgcc 项目资产）。"""


class WeightSpec(NamedTuple):
    """单一 osnet 变体的接入规格：vendor 工厂 + 权重文件。"""

    build: Callable[..., torch.nn.Module]
    path: Path


OSNET_WEIGHTS: dict[str, WeightSpec] = {
    "osnet-x0_75": WeightSpec(
        osnet_x0_75,
        WEIGHTS_DIR
        / "osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    ),
    "osnet-x1_0": WeightSpec(
        osnet_x1_0,
        WEIGHTS_DIR
        / "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    ),
    "osnet-x0_75-ft-v2": WeightSpec(
        osnet_x0_75,
        WEIGHTS_DIR / "osnet_x0_75_ft_v2.pth",
    ),
    "osnet-x0_75-ft-v21": WeightSpec(
        osnet_x0_75,
        WEIGHTS_DIR / "osnet_x0_75_ft_v21.pth",
    ),
}
"""reid tag → 变体规格映射（模型指纹单一数据源；cfg ``REID_KINDS`` osnet
侧与其对齐——tag 值即 ``cache_cfg_tag`` 里的模型指纹，换变体即 cache miss）。

``osnet-x0_75-ft-v2`` = jxl 域微调交付（2026-09-16，需求 C §7.5 命名
``osnet-x0_75-ft-v{n}``）：x0_75 基座 triplet batch-hard 微调（sgcc0），
train 正对 848 + 远距负对 861 + 硬负对 7、eval held-out 328 不进训练；
md5 ``a33775ef``。基座 MSMT17 通用权重定标门七格全败（裁决 B）→ 域微调
治本线；§7.4 两段验收门（定标 same_p5 > diff_p95 + GT 15 窗 F1）复测通过
后方可切生产 reid 轴。

``osnet-x0_75-ft-v21`` = v2.1 追加交付（2026-09-16）：姿态漂移正对 jxl 侧
自挖重训（伪轨迹关联重连对 + ft_v2 难例带 cos<0.62 + 10-41 段定向收录），
详见 iapx ``docs/jxl-notice-2026-09-16-osnet-v21-pair-self-mining.md``。"""

OSNET_DEFAULT_TAG = "osnet-x1_0"
"""默认变体（Phase 4 质量优先裁决：最强特征 x1_0，全宽 channels 64/256/384/512）。"""

_BATCH = 64
"""``embed_batch`` 单次推理批量（spec §2.2；全 corpus 重嵌入约 43k crops 的
吞吐/显存折中）。"""


class OsnetEmbedder:
    """OSNet 人像 crop → 512-d L2 归一化特征（GPU 推理，tag 选变体）。

    ``Embedder`` 协议（:mod:`jxl.vdt.reid_assoc` 单一数据源）的第二实现。
    ``__init__(tag)`` 经 ``OSNET_WEIGHTS`` 解析变体并加载权重剥分类头；
    ``embed`` 协议单 crop 推理、``embed_batch`` 批量推理（iapx 原批量语义）。
    GPU 不可用即报错（No Silent Degradation——重嵌入路径是批量 GPU 工作负载，
    无 CPU 降级）；未知 tag 报错（先于权重加载）。
    """

    def __init__(
        self, tag: str = OSNET_DEFAULT_TAG, device: str = "cuda"
    ) -> None:
        if device.startswith("cuda"):
            assert torch.cuda.is_available(), (
                "CUDA unavailable — OSNet reembed requires GPU (no silent CPU fallback)"
            )
        spec = OSNET_WEIGHTS.get(tag)
        if spec is None:
            msg = f"unknown osnet tag {tag!r} (known: {sorted(OSNET_WEIGHTS)})"
            raise ValueError(msg)
        self.tag = tag
        self.device = torch.device(device)
        model = spec.build(num_classes=4101, pretrained=False)
        ckpt = torch.load(spec.path, map_location="cpu", weights_only=True)
        # 三态 unwrap：zoo 权重 = 裸 state_dict；训练 checkpoint = {"state_dict"}
        # （torchreid save_ckpt 惯例）或 {"model"}（jxl 微调交付格式，ft-v2 起）。
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            # 分类头 shape 不符（换权重文件类别数不同）→ 从 checkpoint 读
            # 真实类别数重建再 strict load；无 classifier 键则原错误上抛。
            w = state.get("classifier.weight")
            if w is None:
                raise
            model = spec.build(num_classes=int(w.shape[0]), pretrained=False)
            model.load_state_dict(state, strict=True)
        model.classifier = torch.nn.Identity()  # 剥分类头 → 512-d 特征
        self.model = model.to(self.device).eval()

    def _preprocess(self, crops: list[np.ndarray]) -> torch.Tensor:
        """BGR HxWx3 crops → N×3×256×128 归一化张量（ImageNet mean/std，RGB 序）。"""
        h, w = OSNET_INPUT_HW
        arr = np.stack(
            [
                (cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)[..., ::-1]
                 .astype(np.float32) / 255.0 - _MEAN)
                / _STD
                for c in crops
            ]
        )  # BGR→RGB（[::-1] 在 HWC 上作用于通道维）
        return torch.from_numpy(arr.transpose(0, 3, 1, 2))

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """``Embedder`` 协议实现：单 BGR crop → 512-d L2 归一化 ``float32`` 向量。

        退化 crop（零面积 / 非 HxWx3）→ **全零 512-d 哨兵**（协议契约：
        ``associate`` 见零范数即 ``id=0`` 不匹配不新建——iapx 原批量接口此处
        raise ValueError，协议钉死哨兵语义）。零范数输出原样返回（同哨兵路径，
        ``reid_assoc.embedding_valid`` 视为提取失败）。
        """
        if crop.ndim != 3 or crop.shape[2] != 3 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros(OSNET_DIM, dtype=np.float32)
        return self.embed_batch([crop])[0]

    def embed_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """批量 crop → 512-d L2 归一化向量列表（与输入顺序一致；iapx 原语义）。

        零尺寸/非 HxWx3 输入 ValueError fail-loud（原实现语义保留——批量入口
        供离线重嵌入调用方，crop 裁剪异常是调用方 bug，不静默退化到零向量）。
        输出 ``np.ndarray float32``（jxl 协议原生类型；iapx 原返回
        ``list[list[float]]`` 为其 cache 序列化形态，非本仓消费面）。零范数
        输出原样返回（同上，视为提取失败哨兵）。
        """
        if not crops:
            return []
        for c in crops:
            if c.ndim != 3 or c.shape[2] != 3 or c.shape[0] == 0 or c.shape[1] == 0:
                msg = f"zero-area or non-HWC crop {getattr(c, 'shape', None)}"
                raise ValueError(msg)
        feats: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(crops), _BATCH):
                batch = self._preprocess(crops[i : i + _BATCH]).to(self.device)
                out = self.model(batch).cpu().numpy()
                for v in out:
                    n = float(np.linalg.norm(v))
                    feats.append(v / n if n > 0 else v)
        return feats
