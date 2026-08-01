"""RTMPose Functional Core —— 纯函数预处理与 SimCC 解码（spec §6）。

1:1 移植 dayn9t usls（Rust）RTMPose 的 ``hbb2cs`` / ``top_down_affine`` /
``get_warp_matrix`` / SimCC ``postprocess``。本模块**零模型、零 onnxruntime**，
仅依赖 numpy / opencv / pydantic Point，可充分单测；命令式外壳（ort session
持有、批 forward）在 ``pose.py``，本模块是其 Functional Core。

关键决策（与 usls 钉死一致）：

- **normalize=False ⇒ 不除 255**：``normalized = (img_float - mean) / std``，
  img 取值 [0,255]。常见错误是先 ``img/255`` 再减 mean——那会引入 1/256 的尺度偏差，
  导致 SimCC 输出整体偏移。此处**直接** [0,255]→减 mean→除 std。
- **通道序 BGR→RGB**：``preprocess_crop`` 入参为 opencv 原生 **BGR** uint8 crop，
  内部 ``crop[:,:,::-1]`` 转 **RGB**——RTMPose(mmpose) 训练于 RGB，且 ImageNet
  mean/std ``(123.675,116.28,103.53)=[R,G,B]`` 序，配 RGB 图才通道对齐（usls 直接喂
  BGR 系其小瑕疵：R/B 通道用错均值；实测 RGB 关键点可见数 +1、均值 conf +0.04）。
- **坐标参照系**：本模块所有输出（``Keypoints.pts``）在 **crop 自身像素坐标系**，
  origin=crop 左上角。调用方（``pose.py``）负责 ``+= crop.xymin`` 平移到全图、
  再按全图尺寸归一化。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from jvi.geo.point2d import Point
from jxl.vdt.types import Keypoints

# ---------------------------------------------------------------------------
# 权威常量（usls config.rs，钉死）
# ---------------------------------------------------------------------------

RTMPOSE_HW: tuple[int, int] = (256, 192)
"""RTMPose 输入 (H, W)。H=256, W=192（aspect=0.75）。"""

RTMPOSE_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
"""RGB 通道均值（ImageNet [R,G,B]×255），0-255 空间（normalize=False 不除 255）。"""

RTMPOSE_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)
"""RGB 通道标准差（ImageNet [R,G,B]×255）。"""

SIMCC_SPLIT_RATIO: float = 2.0
"""SimCC 分辨率放大比：x 轴 W*2=384，y 轴 H*2=512。"""

HBB2CS_PADDING: float = 1.25
"""hbb→center/scale 的外扩系数（usls 默认）。"""

DEFAULT_KCONF: float = 0.35
"""默认关键点置信阈值（低于此 conf 视为不可见）。"""


# ---------------------------------------------------------------------------
# 仿射几何内部辅助（移植 usls）
# ---------------------------------------------------------------------------


def _rotate(p: tuple[float, float], rad: float) -> tuple[float, float]:
    """二维旋转（移植 usls rotate）。"""
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    return (p[0] * cos_r - p[1] * sin_r, p[0] * sin_r + p[1] * cos_r)


def _get_3rd_point(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float]:
    """由两点求仿射第三点（移植 usls get_3rd_point）。

    取 a→b 的垂直方向，从 b 沿该方向延伸，保证三点不共线。
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (b[0] - dy, b[1] + dx)


# ---------------------------------------------------------------------------
# 公开 API（pose.py 依赖这些精确签名）
# ---------------------------------------------------------------------------


def hbb2cs(
    crop_w: float, crop_h: float, padding: float = HBB2CS_PADDING
) -> tuple[Point, Point]:
    """crop 尺寸(像素) → ``(center, scale)``，均在 crop 自身像素坐标系。

    移植 usls ``hbb2cs``：把紧 bounding box（x1=0, y1=0, x2=crop_w, y2=crop_h）
    转成 center（几何中心）与 scale（外扩后的 box 宽高）。

    :param crop_w: crop 宽（像素）。
    :param crop_h: crop 高（像素）。
    :param padding: 外扩系数，默认 1.25。
    :returns: ``(Point(cx, cy), Point(sx, sy))``，sx=crop_w*padding，sy=crop_h*padding。
    """
    cx = crop_w * 0.5
    cy = crop_h * 0.5
    sx = crop_w * padding
    sy = crop_h * padding
    return Point(x=cx, y=cy), Point(x=sx, y=sy)


def get_warp_matrix(
    center: Point,
    scale: Point,
    rot: float,
    out_wh: tuple[int, int],
    shift: tuple[float, float],
    inv: bool,
) -> np.ndarray:
    """三点仿射矩阵 ``(2,3) float32``（1:1 移植 usls get_warp_matrix）。

    :param center: 源 crop 中心。
    :param scale: 源 scale（外扩 box 宽高）；``scale.x`` 作为源参考宽度。
    :param rot: 旋转角度（度）。
    :param out_wh: 目标 ``(W, H)``。
    :param shift: 中心偏移（scale 相对单位）。
    :param inv: True → 求逆变换（src/dst 互换）。
    :returns: ``np.ndarray`` shape ``(2, 3)`` dtype ``float32``；退化（|det|<1e-6）
        返回单位仿射。
    """
    src_w = scale.x
    dst_w = out_wh[0]
    dst_h = out_wh[1]
    rot_rad = rot * math.pi / 180.0

    src_dir = _rotate((0.0, src_w * -0.5), rot_rad)
    dst_dir = (0.0, dst_w * -0.5)

    src_0 = (center.x + scale.x * shift[0], center.y + scale.y * shift[1])
    src_1 = (
        center.x + src_dir[0] + scale.x * shift[0],
        center.y + src_dir[1] + scale.y * shift[1],
    )
    src_2 = _get_3rd_point(src_0, src_1)

    dst_0 = (dst_w * 0.5, dst_h * 0.5)
    dst_1 = (dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1])
    dst_2 = _get_3rd_point(dst_0, dst_1)

    if inv:
        src_0, src_1, src_2, dst_0, dst_1, dst_2 = (
            dst_0,
            dst_1,
            dst_2,
            src_0,
            src_1,
            src_2,
        )

    (x1, y1), (x2, y2), (x3, y3) = src_0, src_1, src_2
    (u1, v1), (u2, v2), (u3, v3) = dst_0, dst_1, dst_2

    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if abs(det) < 1e-6:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    m00 = (u1 * (y2 - y3) + u2 * (y3 - y1) + u3 * (y1 - y2)) / det
    m01 = (x1 * (u3 - u2) + x2 * (u1 - u3) + x3 * (u2 - u1)) / det
    m02 = (
        x1 * (y2 * u3 - y3 * u2) + x2 * (y3 * u1 - y1 * u3) + x3 * (y1 * u2 - y2 * u1)
    ) / det
    m10 = (v1 * (y2 - y3) + v2 * (y3 - y1) + v3 * (y1 - y2)) / det
    m11 = (x1 * (v2 - v3) + x2 * (v3 - v1) + x3 * (v1 - v2)) / det
    m12 = (
        x1 * (y2 * v3 - y3 * v2) + x2 * (y3 * v1 - y1 * v3) + x3 * (y1 * v2 - y2 * v1)
    ) / det

    return np.array([[m00, m01, m02], [m10, m11, m12]], dtype=np.float32)


def top_down_affine(
    crop: np.ndarray,
    center: Point,
    scale: Point,
    out_hw: tuple[int, int] = RTMPOSE_HW,
) -> tuple[np.ndarray, Point]:
    """把 crop 仿射到 ``out_hw``，返回 ``(warped, adjusted_scale)``。

    移植 usls ``top_down_affine``：先按输出 aspect 调整 scale（保持宽高比），
    再用 ``get_warp_matrix`` 求矩阵，``cv2.warpAffine`` 重采样到 ``(W, H)``。

    :param crop: 源 crop，BGR ``uint8``，shape ``(Hc, Wc, 3)``。
    :param center: crop 中心（``hbb2cs`` 输出）。
    :param scale: crop scale（``hbb2cs`` 输出，将被 aspect 调整）。
    :param out_hw: 目标 ``(H, W)``，默认 ``(256, 192)``。
    :returns: ``(warped [H,W,3] BGR uint8, adjusted_scale Point)``。
    """
    h, w = out_hw[0], out_hw[1]
    aspect = w / h
    b_w = scale.x
    b_h = scale.y
    if b_w > b_h * aspect:
        adjusted = Point(x=b_w, y=b_w / aspect)
    else:
        adjusted = Point(x=b_h * aspect, y=b_h)

    warp_mat = get_warp_matrix(
        center=center,
        scale=adjusted,
        rot=0.0,
        out_wh=(w, h),
        shift=(0.0, 0.0),
        inv=False,
    )
    warped = cv2.warpAffine(
        crop, warp_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )
    return warped, adjusted


def preprocess_crop(
    crop: np.ndarray,
) -> tuple[np.ndarray, Point, Point]:
    """crop ``[Hc,Wc,3]`` BGR uint8 → ``(tensor, center, scale)``。

    全流程：``hbb2cs`` → ``top_down_affine`` → normalize（**不除 255**）→ HWC→CHW
    →加 batch 维。BGR 通道序保持不变。

    :returns: ``(tensor [1,3,256,192] float32 CHW normalized, center, scale)``。
        ``scale`` 为 ``top_down_affine`` 的 aspect-adjusted scale（非原始 hbb2cs scale）。
    """
    crop_h, crop_w = crop.shape[0], crop.shape[1]
    center, scale = hbb2cs(float(crop_w), float(crop_h))
    # BGR(opencv 原生) → RGB：RTMPose 训练于 RGB，mean/std 为 [R,G,B] 序，须通道对齐。
    warped, adjusted_scale = top_down_affine(crop[:, :, ::-1], center, scale)

    mean = np.array(RTMPOSE_MEAN, dtype=np.float32)
    std = np.array(RTMPOSE_STD, dtype=np.float32)
    normalized = (warped.astype(np.float32) - mean) / std
    tensor = normalized.transpose(2, 0, 1)[None]
    return tensor, center, adjusted_scale


def simcc_decode(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    center: Point,
    scale: Point,
    out_hw: tuple[int, int] = RTMPOSE_HW,
    split_ratio: float = SIMCC_SPLIT_RATIO,
    kconf: float = DEFAULT_KCONF,
) -> Keypoints | None:
    """SimCC 解码 → ``Keypoints``（17 点，crop 像素坐标系）。

    移植 usls ``postprocess``：对每个关键点，分别取 x/y SimCC 的 argmax 位置与
    max 激活，conf = ``0.5*(mx+my)``；conf > kconf → 由 argmax 位置反算 crop 像素
    坐标，否则置 ``(0, 0)`` 且 conf 记 0（不可见）。

    :param simcc_x: shape ``[K, W*split]``（K=关键点数，从数组读）。
    :param simcc_y: shape ``[K, H*split]``。
    :param center: crop 中心（``preprocess_crop`` 输出）。
    :param scale: adjusted scale（``preprocess_crop`` 输出）。
    :param out_hw: 参照 ``(H, W)``，默认 ``(256, 192)``。
    :param split_ratio: SimCC 放大比，默认 2.0。
    :param kconf: 关键点置信阈值，默认 0.35。
    :returns: ``Keypoints(pts, conf)``；**若全部 K 点 conf 均 ≤ kconf（crop 退化/
        无人）→ 返回 None**（显式 null，spec §9）。
    """
    h, w = out_hw[0], out_hw[1]
    x_factor = scale.x / (split_ratio * w)
    y_factor = scale.y / (split_ratio * h)
    x_offset = center.x - scale.x * 0.5
    y_offset = center.y - scale.y * 0.5

    k = simcc_x.shape[0]
    pts: list[Point] = []
    confs: list[float] = []
    any_visible = False

    for i in range(k):
        x_loc = int(np.argmax(simcc_x[i]))
        y_loc = int(np.argmax(simcc_y[i]))
        mx = float(np.max(simcc_x[i]))
        my = float(np.max(simcc_y[i]))
        conf = 0.5 * (mx + my)

        if conf > kconf:
            any_visible = True
            px = x_loc * x_factor + x_offset
            py = y_loc * y_factor + y_offset
            pts.append(Point(x=px, y=py))
            confs.append(conf)
        else:
            pts.append(Point(x=0.0, y=0.0))
            confs.append(0.0)

    if not any_visible:
        return None
    return Keypoints(pts=pts, conf=confs)


# ---------------------------------------------------------------------------
# 单测（自包含，合成数据，零模型依赖；pytest 发现 test_* 函数）
# ---------------------------------------------------------------------------


def test_hbb2cs_center_and_scale() -> None:
    """crop 100×200 → center=(50,100), scale=(125,250)（padding 1.25）。"""
    center, scale = hbb2cs(100.0, 200.0)
    assert center.x == 50.0
    assert center.y == 100.0
    assert scale.x == 125.0
    assert scale.y == 250.0


def test_hbb2cs_custom_padding() -> None:
    """自定义 padding 改变 scale 不改变 center。"""
    center, scale = hbb2cs(100.0, 200.0, padding=1.0)
    assert center.x == 50.0
    assert center.y == 100.0
    assert scale.x == 100.0
    assert scale.y == 200.0


def test_get_warp_matrix_square_centered() -> None:
    """正方形 scale、center 居中、rot=0 → 对角元 ≈ dst_w/src_w，非对角 ≈ 0。"""
    center = Point(x=50.0, y=50.0)
    scale = Point(x=100.0, y=100.0)
    m = get_warp_matrix(
        center=center, scale=scale, rot=0.0, out_wh=(192, 256), shift=(0.0, 0.0), inv=False
    )
    assert m.shape == (2, 3)
    assert m.dtype == np.float32
    # src_w=100, dst_w=192 → 缩放比 1.92（src_dir 用 scale.x 作 y 参考，两轴同比）
    ratio = 192.0 / 100.0
    assert abs(m[0, 0] - ratio) < 1e-4
    assert abs(m[1, 1] - ratio) < 1e-4
    assert abs(m[0, 1]) < 1e-4
    assert abs(m[1, 0]) < 1e-4


def test_get_warp_matrix_degenerate_returns_identity() -> None:
    """零 scale → src 三点重合（det=0）→ 单位仿射。"""
    m = get_warp_matrix(
        center=Point(x=0.0, y=0.0),
        scale=Point(x=0.0, y=0.0),
        rot=0.0,
        out_wh=(192, 256),
        shift=(0.0, 0.0),
        inv=False,
    )
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    assert np.allclose(m, expected)


def test_top_down_affine_shape_and_aspect() -> None:
    """合成 crop → 输出 (256,192,3) uint8；adjusted_scale 保持 aspect 0.75。"""
    crop = np.zeros((300, 200, 3), dtype=np.uint8)
    center, scale = hbb2cs(200.0, 300.0)
    warped, adjusted = top_down_affine(crop, center, scale)
    assert warped.shape == (256, 192, 3)
    assert warped.dtype == np.uint8
    # adjusted 宽/高比 = W/H = 192/256 = 0.75
    assert abs(adjusted.x / adjusted.y - 192.0 / 256.0) < 1e-5


def test_top_down_affine_aspect_wide_crop() -> None:
    """宽 crop（b_w > b_h*aspect）走 (b_w, b_w/aspect) 分支，仍保持 aspect。"""
    crop = np.zeros((100, 400, 3), dtype=np.uint8)
    center, scale = hbb2cs(400.0, 100.0)
    _, adjusted = top_down_affine(crop, center, scale)
    assert abs(adjusted.x / adjusted.y - 192.0 / 256.0) < 1e-5
    # 宽 crop：adjusted.x = scale.x = 500
    assert abs(adjusted.x - 500.0) < 1e-4


def test_preprocess_crop_shape_dtype() -> None:
    """输出 tensor shape == (1,3,256,192) float32。"""
    crop = np.zeros((300, 200, 3), dtype=np.uint8)
    tensor, center, scale = preprocess_crop(crop)
    assert tensor.shape == (1, 3, 256, 192)
    assert tensor.dtype == np.float32


def test_preprocess_crop_normalize_not_divide_255() -> None:
    """crop 各通道填"该通道 RGB 均值" → 内部像素 normalized ≈ 0（验证不除 255）。

    preprocess_crop 内部 BGR→RGB：故 BGR crop[c] 应填 ``RGB_MEAN[2-c]``，使转换后
    RGB[c]=MEAN[c] → ``(MEAN[c]-MEAN[c])/STD[c]=0``。不除 255 → ≈0；若误除 255 →
    ``(MEAN/255 - MEAN)/STD ≈ -2.1``。padding=1.25 使 affine 源 box 大于 crop，
    box 外边界由 warpAffine borderValue=0 填充 → ``(0-MEAN[c])/STD[c]``。
    """
    crop = np.zeros((256, 192, 3), dtype=np.uint8)
    # BGR[c] = RGB_MEAN[2-c]，使 BGR→RGB 后通道对齐各自均值。
    crop[:, :, 0] = int(round(RTMPOSE_MEAN[2]))
    crop[:, :, 1] = int(round(RTMPOSE_MEAN[1]))
    crop[:, :, 2] = int(round(RTMPOSE_MEAN[0]))
    tensor, _, _ = preprocess_crop(crop)
    # 中心像素映射自 crop 内部 → 不除 255 时三通道均 ≈ 0
    for c in range(3):
        center_val = float(tensor[0, c, 128, 96])
        assert abs(center_val) < 0.01, f"ch{c} center={center_val}（应≈0，证不除255）"
    # 边界（box 外）被 0 填充 → (0-MEAN[c])/STD[c]（佐证 borderValue 生效）
    for c in range(3):
        corner_val = float(tensor[0, c, 0, 0])
        expected_border = (0.0 - RTMPOSE_MEAN[c]) / RTMPOSE_STD[c]
        assert abs(corner_val - expected_border) < 1e-3


def test_simcc_decode_spike_at_center() -> None:
    """在 kpt0 的 x/y 中央放尖峰 → 解码点 ≈ crop center。"""
    center = Point(x=100.0, y=150.0)
    scale = Point(x=250.0, y=375.0)  # adjusted scale（宽高比已修正）
    simcc_x = np.zeros((17, 384), dtype=np.float32)
    simcc_y = np.zeros((17, 512), dtype=np.float32)
    # 中央位置：x_loc=192 (W*split/2), y_loc=256 (H*split/2)
    simcc_x[0, 192] = 1.0
    simcc_y[0, 256] = 1.0

    kpts = simcc_decode(simcc_x, simcc_y, center, scale)
    assert kpts is not None
    assert len(kpts.pts) == 17
    assert len(kpts.conf) == 17
    # kpt0：conf=1.0，坐标反算到 center
    assert abs(kpts.conf[0] - 1.0) < 1e-5
    assert abs(kpts.pts[0].x - 100.0) < 1e-3
    assert abs(kpts.pts[0].y - 150.0) < 1e-3
    # 其余 kpt 全零 simcc → conf=0
    assert kpts.conf[1] == 0.0
    assert kpts.pts[1].x == 0.0


def test_simcc_decode_explicit_offset_formula() -> None:
    """独立验证 x_factor/x_offset 公式：非中央尖峰。"""
    center = Point(x=100.0, y=150.0)
    scale = Point(x=250.0, y=375.0)
    # x_factor=250/(2*192)=0.65104, x_offset=100-125=-25
    # y_factor=375/(2*256)=0.73242, y_offset=150-187.5=-37.5
    simcc_x = np.zeros((17, 384), dtype=np.float32)
    simcc_y = np.zeros((17, 512), dtype=np.float32)
    simcc_x[5, 96] = 0.8
    simcc_y[5, 128] = 0.6
    kpts = simcc_decode(simcc_x, simcc_y, center, scale)
    assert kpts is not None
    expected_x = 96 * (250.0 / (2.0 * 192.0)) + (100.0 - 125.0)
    expected_y = 128 * (375.0 / (2.0 * 256.0)) + (150.0 - 187.5)
    assert abs(kpts.pts[5].x - expected_x) < 1e-3
    assert abs(kpts.pts[5].y - expected_y) < 1e-3
    assert abs(kpts.conf[5] - 0.5 * (0.8 + 0.6)) < 1e-5


def test_simcc_decode_all_low_conf_returns_none() -> None:
    """全 17 点 conf ≤ kconf → None（crop 退化/无人，显式 null）。"""
    simcc_x = np.zeros((17, 384), dtype=np.float32)
    simcc_y = np.zeros((17, 512), dtype=np.float32)
    center = Point(x=100.0, y=150.0)
    scale = Point(x=250.0, y=375.0)
    kpts = simcc_decode(simcc_x, simcc_y, center, scale)
    assert kpts is None


def test_simcc_decode_kconf_threshold_boundary() -> None:
    """conf == kconf（≤ 不 >）→ 视为不可见；略高于 → 可见。"""
    center = Point(x=0.0, y=0.0)
    scale = Point(x=192.0, y=256.0)
    # conf = 0.5*(mx+my)；取 mx+my = 2*kconf → conf == kconf（边界，不可见）
    kconf = 0.35
    val = 2.0 * kconf  # 均分到 mx/my 各一半 → 各 0.35
    simcc_x = np.zeros((17, 384), dtype=np.float32)
    simcc_y = np.zeros((17, 512), dtype=np.float32)
    simcc_x[0, 0] = val / 2.0
    simcc_y[0, 0] = val / 2.0
    kpts = simcc_decode(simcc_x, simcc_y, center, scale, kconf=kconf)
    # conf 恰 == kconf，不 > kconf → 全不可见 → None
    assert kpts is None

    # 略高于阈值 → 可见
    simcc_x[0, 0] = (val / 2.0) + 1e-3
    kpts2 = simcc_decode(simcc_x, simcc_y, center, scale, kconf=kconf)
    assert kpts2 is not None
    assert kpts2.conf[0] > kconf
