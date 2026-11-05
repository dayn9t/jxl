"""像素域 bbox 裁片单一数据源（spec v2 A1）——rs ``ulabel-utils/src/crop.rs`` 的对齐镜像.

ulabel docs/CONTRACT.md「crop」行登记的跨语言契约：同名函数、同语义
（BorderExtend 逐边外延、负值内缩、帧界 clamp、半远离零取整——
``math.floor(v + 0.5)``，py 内建 ``round`` 是银行家舍入，0.5 处与 rs 差 1px，
禁止使用）。外延规范见 skill ``bbox-crop-expansion``。
"""

import math
from pathlib import Path

from PIL import Image

from jxl.label.meta import BorderExtend

XYXY = tuple[float, float, float, float]


def be_uniform(ratio: float) -> BorderExtend:
    """四值同值的 BorderExtend 便捷构造（等价旧单值 ratio 风格）。"""
    return BorderExtend(left=ratio, right=ratio, top=ratio, bottom=ratio)


def xyxy_iou(a: XYXY, b: XYXY) -> float:
    """两 xyxy 框 IoU（无交集/零并集返回 0）。"""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def expand_xyxy(bbox: XYXY, be: BorderExtend, width: int, height: int) -> XYXY:
    """bbox 按 BorderExtend 逐边外延（left/right 相对宽、top/bottom 相对高，负值内缩），clamp 到帧界。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (
        min(max(x1 - be.left * w, 0.0), float(width)),
        min(max(y1 - be.top * h, 0.0), float(height)),
        min(max(x2 + be.right * w, 0.0), float(width)),
        min(max(y2 + be.bottom * h, 0.0), float(height)),
    )


def crop_rect_xyxy(
    bbox: XYXY, be: BorderExtend, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """外延 + 取整后的像素裁切矩形 (x, y, w, h)；空框（w/h < 1）返回 None。"""
    x1, y1, x2, y2 = expand_xyxy(bbox, be, width, height)
    x1 = math.floor(x1 + 0.5)
    y1 = math.floor(y1 + 0.5)
    x2 = math.floor(x2 + 0.5)
    y2 = math.floor(y2 + 0.5)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def crop_expanded(im: Image.Image, bbox: XYXY, be: BorderExtend) -> Image.Image:
    """按外延取整裁切 PIL 图像（帧界 clamp 内置）；空框 raise。"""
    rect = crop_rect_xyxy(bbox, be, im.width, im.height)
    if rect is None:
        raise ValueError(f"外延后空框: bbox={bbox} be={be} frame={im.size}")
    x, y, w, h = rect
    return im.crop((x, y, x + w, y + h))


def save_expanded(im: Image.Image, bbox: XYXY, dst: Path, be: BorderExtend, quality: int = 90) -> None:
    """crop_expanded + 落盘（建父目录；JPEG 按 quality，其余格式直存——与 rs 对齐）."""
    crop = crop_expanded(im, bbox, be)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        crop.save(dst, quality=quality)
    else:
        crop.save(dst)
