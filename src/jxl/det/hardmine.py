"""Cross-detector 难例挖掘核心算法（Functional Core）。

纯函数: 双检测器框级比对 + 难例分类 + YOLO 标注生成。
供 bin/person_mine.py（Imperative Shell）调用。无 IO/模型依赖，充分单测。
"""
from __future__ import annotations

from enum import StrEnum

# 归一化 xyxy + 置信度: (x1, y1, x2, y2, conf), 坐标 ∈ [0,1]
Box = tuple[float, float, float, float, float]


class SampleClass(StrEnum):
    """双检测器比对后的样本分类。"""

    DROP_EMPTY = "drop_empty"  # 两检测器均无框（空帧）→ 丢弃
    DROP_AGREE = "drop_agree"  # 两检测器框完全配对（一致）→ 丢弃
    POSITIVE = "positive"  # YOLOE 有框且与 person.pt 分歧 → 正样本（YOLOE 框）
    NEGATIVE = "negative"  # YOLOE 无框、person.pt 有框（误检）→ 负样本（空 txt）


def xyxy_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """两归一化 xyxy 框 (x1,y1,x2,y2) 的 IoU。无交集/零面积返回 0。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(
    boxes_a: list[Box],
    boxes_b: list[Box],
    iou_thr: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """贪心 IoU 匹配（按 IoU 降序配对，IoU<iou_thr 不配）。

    Returns:
        (matched[(idx_a, idx_b, iou)], unmatched_a[idx], unmatched_b[idx])
    """
    pairs = sorted(
        (
            (xyxy_iou(a[:4], b[:4]), ia, ib)
            for ia, a in enumerate(boxes_a)
            for ib, b in enumerate(boxes_b)
        ),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iov, ia, ib in pairs:
        if iov < iou_thr:
            break
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        matched.append((ia, ib, iov))
    unmatched_a = [i for i in range(len(boxes_a)) if i not in used_a]
    unmatched_b = [i for i in range(len(boxes_b)) if i not in used_b]
    return matched, unmatched_a, unmatched_b


def to_yolo_label(boxes: list[Box], cls_id: int = 0) -> str:
    """归一化 xyxy Box 列表 → YOLO 标注行（cls cx cy w h），每行一框。

    Box 坐标已归一化（caller 从 ultralytics boxes.xyxyn 取），无需图像尺寸。
    """
    lines: list[str] = []
    for box in boxes:
        x1, y1, x2, y2, _conf = box
        ax1, ax2 = min(x1, x2), max(x1, x2)
        ay1, ay2 = min(y1, y2), max(y1, y2)
        cx, cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        w, h = ax2 - ax1, ay2 - ay1
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def classify_sample(
    person_boxes: list[Box],
    yoloe_boxes: list[Box],
    iou_thr: float,
) -> SampleClass:
    """双检测器框级比对 → 难例分类（决策见设计文档 §5）。

    判据: ① YOLOE 有无框（决定正/负/丢）; ② 有无未配对框（决定分歧）。
    只要 YOLOE 有框，正标注一律用 YOLOE 框（更可信）。
    """
    if not yoloe_boxes and not person_boxes:
        return SampleClass.DROP_EMPTY
    if not yoloe_boxes:
        return SampleClass.NEGATIVE
    if not person_boxes:
        return SampleClass.POSITIVE
    _matched, unmatched_p, unmatched_y = greedy_match(person_boxes, yoloe_boxes, iou_thr)
    if not unmatched_p and not unmatched_y:
        return SampleClass.DROP_AGREE
    return SampleClass.POSITIVE
