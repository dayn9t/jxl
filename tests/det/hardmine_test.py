"""hardmine 纯函数单测: 框级 IoU 匹配 + 难例分类 + YOLO 标注生成。"""
from __future__ import annotations

from jxl.det.hardmine import (
    SampleClass,
    classify_sample,
    find_consensus_positions,
    greedy_match,
    pick_by_priority,
    score_sample,
    to_yolo_label,
    xyxy_iou,
)

IOU_THR = 0.3
WEIGHTS = {"yoloe": 0.25, "gdino": 0.35, "rfdetr": 0.4}
PRIORITY = ["rfdetr", "gdino", "yoloe"]


def test_xyxy_iou_identical() -> None:
    assert xyxy_iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)) == 1.0


def test_xyxy_iou_disjoint() -> None:
    assert xyxy_iou((0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)) == 0.0


def test_xyxy_iou_partial() -> None:
    # 交 0.4*0.4=0.16 / 并 (0.25+0.25-0.16)=0.34
    iou = xyxy_iou((0.0, 0.0, 0.5, 0.5), (0.1, 0.1, 0.6, 0.6))
    assert abs(iou - 0.16 / 0.34) < 1e-9


def test_xyxy_iou_contained() -> None:
    # 小框完全在大框内: IoU = 小框面积 / 大框面积 = 0.04 / 1.0
    iou = xyxy_iou((0.0, 0.0, 1.0, 1.0), (0.4, 0.4, 0.6, 0.6))
    assert abs(iou - 0.04) < 1e-9


def test_greedy_match_all_matched() -> None:
    a = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.6, 0.6, 0.9, 0.9, 0.8)]
    b = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.6, 0.6, 0.9, 0.9, 0.8)]
    matched, ua, ub = greedy_match(a, b, IOU_THR)
    assert len(matched) == 2
    assert ua == []
    assert ub == []


def test_greedy_match_unmatched_both() -> None:
    a = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.8, 0.8, 0.95, 0.95, 0.8)]
    b = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.8, 0.1, 0.95, 0.3, 0.7)]
    matched, ua, ub = greedy_match(a, b, IOU_THR)
    assert len(matched) == 1
    assert ua == [1]
    assert ub == [1]


def test_greedy_match_below_threshold() -> None:
    a = [(0.0, 0.0, 0.1, 0.1, 0.9)]
    b = [(0.9, 0.9, 1.0, 1.0, 0.9)]
    matched, ua, ub = greedy_match(a, b, IOU_THR)
    assert matched == []
    assert ua == [0]
    assert ub == [0]


def test_greedy_match_empty_inputs() -> None:
    matched, ua, ub = greedy_match([], [], IOU_THR)
    assert matched == []
    assert ua == []
    assert ub == []


def test_to_yolo_label_basic() -> None:
    boxes = [(0.1, 0.2, 0.3, 0.4, 0.95)]
    assert to_yolo_label(boxes, cls_id=0) == "0 0.200000 0.300000 0.200000 0.200000"


def test_to_yolo_label_swapped_coords_clamped() -> None:
    # x1>x2 / y1>y2 规整（防 w/h 负）
    boxes = [(0.3, 0.4, 0.1, 0.2, 0.5)]
    assert to_yolo_label(boxes, cls_id=0) == "0 0.200000 0.300000 0.200000 0.200000"


def test_to_yolo_label_empty() -> None:
    assert to_yolo_label([], cls_id=0) == ""


def test_to_yolo_label_multi_box() -> None:
    boxes = [(0.0, 0.0, 0.5, 1.0, 0.9), (0.5, 0.5, 1.0, 1.0, 0.8)]
    out = to_yolo_label(boxes, cls_id=0)
    assert out == "0 0.250000 0.500000 0.500000 1.000000\n0 0.750000 0.750000 0.500000 0.500000"


def test_classify_drop_empty() -> None:
    assert classify_sample([], [], IOU_THR) == SampleClass.DROP_EMPTY


def test_classify_negative_person_false_positive() -> None:
    # YOLOE 无框、person 有框 → 误检负样本
    assert classify_sample([(0.1, 0.1, 0.5, 0.5, 0.9)], [], IOU_THR) == SampleClass.NEGATIVE


def test_classify_positive_person_missed() -> None:
    # YOLOE 有框、person 无框 → 漏检正样本
    assert classify_sample([], [(0.1, 0.1, 0.5, 0.5, 0.9)], IOU_THR) == SampleClass.POSITIVE


def test_classify_drop_agree() -> None:
    box = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    assert classify_sample(box, box, IOU_THR) == SampleClass.DROP_AGREE


def test_classify_positive_extra_yoloe() -> None:
    # YOLOE 多出框（漏检位置）→ 分歧正样本
    person = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    yoloe = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.7, 0.7, 0.9, 0.9, 0.8)]
    assert classify_sample(person, yoloe, IOU_THR) == SampleClass.POSITIVE


def test_classify_positive_extra_person() -> None:
    # person 多出框、但 YOLOE 仍有人 → 正样本（YOLOE 框）
    yoloe = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    person = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.7, 0.7, 0.9, 0.9, 0.8)]
    assert classify_sample(person, yoloe, IOU_THR) == SampleClass.POSITIVE


def test_classify_positive_completely_disjoint() -> None:
    # 双方各有框但彼此完全不相干（IoU=0 全未配对）→ 分歧正样本
    person = [(0.1, 0.1, 0.3, 0.3, 0.9)]
    yoloe = [(0.7, 0.7, 0.9, 0.9, 0.8)]
    assert classify_sample(person, yoloe, IOU_THR) == SampleClass.POSITIVE


def test_to_yolo_label_clamps_out_of_range() -> None:
    # 坐标越界（<0 / >1）→ clamp 到 [0,1]
    boxes = [(-0.1, -0.1, 1.2, 1.2, 0.9)]
    assert to_yolo_label(boxes, cls_id=0) == "0 0.550000 0.550000 1.000000 1.000000"


def test_find_consensus_positions_all_agree() -> None:
    validators = {
        "yoloe": [(0.1, 0.1, 0.5, 0.5, 0.9)],
        "gdino": [(0.1, 0.1, 0.5, 0.5, 0.95)],
        "rfdetr": [(0.12, 0.12, 0.52, 0.52, 0.99)],
    }
    positions = find_consensus_positions(validators, IOU_THR, 2)
    assert len(positions) == 1
    assert set(positions[0][1].keys()) == {"yoloe", "gdino", "rfdetr"}


def test_find_consensus_positions_split() -> None:
    validators = {
        "yoloe": [(0.1, 0.1, 0.3, 0.3, 0.9)],
        "gdino": [(0.1, 0.1, 0.3, 0.3, 0.95)],
        "rfdetr": [(0.7, 0.7, 0.9, 0.9, 0.99)],
    }
    positions = find_consensus_positions(validators, IOU_THR, 2)
    assert len(positions) == 1
    assert set(positions[0][1].keys()) == {"yoloe", "gdino"}


def test_find_consensus_positions_below_k() -> None:
    validators = {
        "yoloe": [(0.1, 0.1, 0.2, 0.2, 0.9)],
        "gdino": [(0.4, 0.4, 0.5, 0.5, 0.95)],
        "rfdetr": [(0.7, 0.7, 0.8, 0.8, 0.99)],
    }
    assert find_consensus_positions(validators, IOU_THR, 2) == []


def test_find_consensus_positions_empty() -> None:
    assert find_consensus_positions({"yoloe": [], "gdino": []}, IOU_THR, 2) == []


def test_pick_by_priority_first() -> None:
    s = {"yoloe": (0.1, 0.1, 0.5, 0.5, 0.9), "rfdetr": (0.12, 0.12, 0.52, 0.52, 0.99)}
    assert pick_by_priority(s, PRIORITY) == s["rfdetr"]


def test_pick_by_priority_fallback() -> None:
    s = {"yoloe": (0.1, 0.1, 0.5, 0.5, 0.9)}
    assert pick_by_priority(s, PRIORITY) == s["yoloe"]


def test_pick_by_priority_none() -> None:
    assert pick_by_priority({}, PRIORITY) is None


def test_score_sample_full_agreement() -> None:
    box = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    validators = {"yoloe": box, "gdino": box, "rfdetr": box}
    r = score_sample(box, validators, WEIGHTS, IOU_THR, 2)
    assert r.score == 0.0
    assert r.fp_count == 0
    assert r.fn_count == 0
    assert len(r.boxes) == 1


def test_score_sample_target_missed() -> None:
    validators = {
        "yoloe": [(0.1, 0.1, 0.5, 0.5, 0.9)],
        "gdino": [(0.1, 0.1, 0.5, 0.5, 0.95)],
        "rfdetr": [(0.12, 0.12, 0.52, 0.52, 0.99)],
    }
    r = score_sample([], validators, WEIGHTS, IOU_THR, 2)
    assert r.fn_count == 1
    assert abs(r.score - 1.0) < 1e-9  # 全员认同, W_agree/W_total=1.0
    assert r.boxes[0] == validators["rfdetr"][0]  # RF-DETR 优先


def test_score_sample_target_false_positive() -> None:
    target = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    validators = {"yoloe": [], "gdino": [], "rfdetr": []}
    r = score_sample(target, validators, WEIGHTS, IOU_THR, 2)
    assert r.fp_count == 1
    assert abs(r.score - 1.0) < 1e-9  # 无人认同, (W_total-0)/W_total=1.0
    assert r.boxes == []


def test_score_sample_partial_miss_one_validator() -> None:
    # target 漏, 但仅 yoloe+gdino 认同(2/3) → fn = (0.25+0.35)/1.0 = 0.6
    validators = {
        "yoloe": [(0.1, 0.1, 0.5, 0.5, 0.9)],
        "gdino": [(0.1, 0.1, 0.5, 0.5, 0.95)],
        "rfdetr": [],
    }
    r = score_sample([], validators, WEIGHTS, IOU_THR, 2)
    assert r.fn_count == 1
    assert abs(r.score - 0.6) < 1e-9
    assert r.boxes[0] == validators["gdino"][0]  # RF-DETR 缺→GDINO 回退
