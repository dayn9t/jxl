"""Cross-detector 难例挖掘核心算法（Functional Core）。

纯函数: 双检测器框级比对 + 难例分类 + YOLO 标注生成。
供 bin/det_mine.py（Imperative Shell）调用。无 IO/模型依赖，充分单测。

注: xyxy_iou 几何逻辑与 bin/rmb_eval_grounding.py:21 同源（该处签名 list[float]，
此处收紧为 4-tuple 强类型化）。若修 IoU bug 需两处同步；未来统一可抽 det/box_utils.py。
"""
from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple

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


def find_consensus_positions(
    validators: dict[str, list[Box]],
    iou_thr: float,
    k: int,
) -> list[tuple[Box, dict[str, Box]]]:
    """跨校验器贪心聚类: 找 ≥k 个校验器 IoU 重叠的共识位置。

    按 conf 降序以每个框为种子，向其他校验器找 IoU 最高的未消费框配对；
    支持校验器数 ≥k 则记为一个共识位置。
    Returns: [(representative_box, {validator_name: box}), ...]
    """
    candidates = sorted(
        (
            (name, idx, box)
            for name, boxes in validators.items()
            for idx, box in enumerate(boxes)
        ),
        key=lambda x: x[2][4],
        reverse=True,
    )
    consumed: dict[str, set[int]] = {name: set() for name in validators}
    positions: list[tuple[Box, dict[str, Box]]] = []
    for name_s, idx_s, box_s in candidates:
        if idx_s in consumed[name_s]:
            continue
        supporters: dict[str, Box] = {name_s: box_s}
        consumed[name_s].add(idx_s)
        for name_o, boxes_o in validators.items():
            if name_o == name_s:
                continue
            best_idx: int | None = None
            best_iou = iou_thr
            for idx_o, box_o in enumerate(boxes_o):
                if idx_o in consumed[name_o]:
                    continue
                iov = xyxy_iou(box_s[:4], box_o[:4])
                if iov >= best_iou:
                    best_iou = iov
                    best_idx = idx_o
            if best_idx is not None:
                supporters[name_o] = boxes_o[best_idx]
                consumed[name_o].add(best_idx)
        if len(supporters) >= k:
            positions.append((box_s, supporters))
    return positions


def pick_by_priority(supporters: dict[str, Box], priority: list[str]) -> Box | None:
    """按优先级返回第一个存在的校验器框（标注框回退选框用）。"""
    for name in priority:
        if name in supporters:
            return supporters[name]
    return None


class ScoreResult(NamedTuple):
    """score_sample 的返回: 争议分 + 共识标注框 + 分项计数。"""

    score: float
    boxes: list[Box]
    fp_count: int
    fn_count: int


def score_sample(
    target_boxes: list[Box],
    validators: dict[str, list[Box]],
    weights: dict[str, float],
    iou_thr: float,
    k: int,
) -> ScoreResult:
    """N 模型加权争议分 + 共识标注框。

    fp = Σ target框(认同票<k): (W_total − W_认同)/W_total  (疑似 target 误检)
    fn = Σ 共识漏检位置: W_认同/W_total                   (target 漏, 校验器共识)
    score = fp + fn; 0=全一致; 越大越争议。
    boxes = 共识位置按 weights 降序回退选框(供 L1 自动标注, RF-DETR>GDINO>YOLOE)。
    """
    priority = sorted(weights.keys(), key=lambda n: weights[n], reverse=True)
    w_total = sum(weights.values()) or 1.0

    fp = 0.0
    fp_count = 0
    for tbox in target_boxes:
        w_agree = 0.0
        votes = 0
        for vname, vboxes in validators.items():
            if any(xyxy_iou(tbox[:4], vb[:4]) >= iou_thr for vb in vboxes):
                votes += 1
                w_agree += weights.get(vname, 0.0)
        if votes < k:
            fp += (w_total - w_agree) / w_total
            fp_count += 1

    fn = 0.0
    fn_count = 0
    boxes: list[Box] = []
    for pos_box, supporters in find_consensus_positions(validators, iou_thr, k):
        covered = any(xyxy_iou(pos_box[:4], tb[:4]) >= iou_thr for tb in target_boxes)
        if not covered:
            w_agree = sum(weights.get(n, 0.0) for n in supporters)
            fn += w_agree / w_total
            fn_count += 1
        picked = pick_by_priority(supporters, priority)
        if picked is not None:
            boxes.append(picked)

    return ScoreResult(score=fp + fn, boxes=boxes, fp_count=fp_count, fn_count=fn_count)


def to_yolo_label(boxes: list[Box], cls_id: int = 0) -> str:
    """归一化 xyxy Box 列表 → YOLO 标注行（cls cx cy w h），每行一框。

    Box 坐标已归一化（caller 从 ultralytics boxes.xyxyn 取），无需图像尺寸。
    """
    lines: list[str] = []
    for box in boxes:
        x1, y1, x2, y2, _conf = box
        ax1, ax2 = min(x1, x2), max(x1, x2)
        ay1, ay2 = min(y1, y2), max(y1, y2)
        # clamp 到 [0,1]：防 boxes.xyxyn 边界微负/>1 污染训练标注
        cx = max(0.0, min(1.0, (ax1 + ax2) / 2))
        cy = max(0.0, min(1.0, (ay1 + ay2) / 2))
        w = max(0.0, min(1.0, ax2 - ax1))
        h = max(0.0, min(1.0, ay2 - ay1))
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
