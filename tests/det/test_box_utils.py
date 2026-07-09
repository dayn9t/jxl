"""box_utils.xyxy_iou 单测（抽自 hardmine / rmb_eval_grounding 共用几何）。"""

from jxl.det.box_utils import xyxy_iou


def test_iou_identical() -> None:
    assert xyxy_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint() -> None:
    assert xyxy_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_contain() -> None:
    assert 0.0 < xyxy_iou((0, 0, 10, 10), (2, 2, 8, 8)) < 1.0
