"""hardmine 纯函数单测: 框级 IoU 匹配 + 难例分类 + YOLO 标注生成。"""
from __future__ import annotations

from jxl.det.hardmine import (
    SampleClass,
    classify_sample,
    greedy_match,
    to_yolo_label,
    xyxy_iou,
)

IOU_THR = 0.3


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
