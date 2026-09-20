"""检测框工具(IoU/外延裁切等), 供 hardmine / rmb_eval_grounding 等共用, 消除重复实现.

外延裁切规范见 skill `bbox-crop-expansion`: bbox 裁片给下游(分类器/VLM/训练集)
时各边外延约 10%, 禁止固定尺寸窗口(2026-09-20 rolepool target_crop 实证).
"""

from pathlib import Path

from PIL import Image

XYXY = tuple[float, float, float, float]


def xyxy_iou(
    a: XYXY,
    b: XYXY,
) -> float:
    """两 xyxy 框 IoU.

    几何逻辑: 交集 = max(0, min(x2) - max(x1)) * max(0, min(y2) - max(y1));
    并集 = a面积 + b面积 - 交集; 无交集/零并集返回 0.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def expand_xyxy(
    bbox: XYXY,
    ratio: float = 0.10,
    width: int | None = None,
    height: int | None = None,
) -> XYXY:
    """bbox 各边按 ratio 比例外延(相对边长), 可选 clamp 到帧界.

    规范值 ratio=0.10(用户裁决 2026-09-20); 禁止用固定像素或固定窗口替代——
    固定像素对小框过度外延/大框不足, 固定窗口则切边与稀释并存.
    返回 float xyxy(未取整); 取整与像素 clamp 交给 crop_expanded 或调用方.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    out = (x1 - w * ratio, y1 - h * ratio, x2 + w * ratio, y2 + h * ratio)
    if width is not None:
        out = (max(0.0, out[0]), out[1], min(float(width), out[2]), out[3])
    if height is not None:
        out = (out[0], max(0.0, out[1]), out[2], min(float(height), out[3]))
    return out


def crop_expanded(
    im: Image.Image,
    bbox: XYXY,
    ratio: float = 0.10,
) -> Image.Image:
    """按 expand_xyxy 外延并取整裁切 PIL 图像(帧界 clamp 内置).

    expand_xyxy 的消费薄壳; 训练/抽检出图统一走这里, 避免各脚本手写外延.
    """
    x1, y1, x2, y2 = expand_xyxy(bbox, ratio, im.width, im.height)
    box = (round(x1), round(y1), round(x2), round(y2))
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        raise ValueError(f"外延后空框: bbox={bbox} ratio={ratio} frame={im.size}")
    return im.crop(box)


def save_expanded(
    im: Image.Image,
    bbox: XYXY,
    dst: Path,
    ratio: float = 0.10,
    quality: int = 90,
) -> None:
    """crop_expanded + 落盘(建父目录); 产出物即分类器/VLM 下游输入."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop_expanded(im, bbox, ratio).save(dst, quality=quality)
