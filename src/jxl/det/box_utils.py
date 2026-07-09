"""检测框工具(IoU 等), 供 hardmine / rmb_eval_grounding 共用, 消除重复实现."""


def xyxy_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
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
