"""dedup_gt_boxes 单测：扫描/合并纯函数 + 报告语义。"""
from jxl.bin.dedup_gt_boxes import YoloBox, boxes_to_text, dedup_boxes, find_dup_pairs, parse_label


def _b(cls: int, cx: float, cy: float, w: float, h: float) -> YoloBox:
    return YoloBox(cls, cx, cy, w, h)


def test_parse_and_roundtrip() -> None:
    boxes = parse_label("0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.1 0.1\n")
    assert len(boxes) == 2
    assert boxes[0] == YoloBox(0, 0.5, 0.5, 0.2, 0.3)
    assert parse_label(boxes_to_text(boxes)) == boxes


def test_exact_dup_pair_detected() -> None:
    boxes = [_b(0, 0.362, 0.126, 0.342, 0.321), _b(0, 0.362, 0.125, 0.342, 0.322)]
    pairs = find_dup_pairs(boxes)
    assert pairs == [(0, 1)]
    cleaned, drops = dedup_boxes(boxes)
    assert drops == [1] and cleaned == [boxes[0]]


def test_cross_class_not_merged() -> None:
    boxes = [_b(0, 0.5, 0.5, 0.4, 0.4), _b(1, 0.5, 0.5, 0.4, 0.4)]
    assert find_dup_pairs(boxes) == []


def test_below_threshold_kept() -> None:
    boxes = [_b(0, 0.3, 0.5, 0.2, 0.2), _b(0, 0.42, 0.5, 0.2, 0.2)]  # IoU~0.45
    assert find_dup_pairs(boxes) == []


def test_chain_keeps_first_of_each_pair() -> None:
    # box0≈box1≈box2 三连近重复：pair(0,1),(0,2),(1,2) -> 只留 box0
    boxes = [_b(0, 0.5, 0.5, 0.4, 0.4), _b(0, 0.501, 0.5, 0.4, 0.4),
             _b(0, 0.502, 0.5, 0.4, 0.4)]
    cleaned, drops = dedup_boxes(boxes)
    assert drops == [1, 2] and cleaned == [boxes[0]]
