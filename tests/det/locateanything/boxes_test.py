"""LocateAnything 框后处理纯函数单测: 像素→归一化、钳制、IoU 去重.

模型无 confidence 输出, conf 恒 1.0 是协议约定(见 boxes.py docstring).
"""

from jxl.det.locateanything.boxes import dedup_boxes, normalize_boxes


def test_normalize_basic() -> None:
    """100x100 图, 像素框 (10,20,80,90) → 归一化 (0.1,0.2,0.8,0.9,conf=1.0)."""
    out = normalize_boxes([(10.0, 20.0, 80.0, 90.0)], 100, 100)
    assert out == [(0.1, 0.2, 0.8, 0.9, 1.0)]


def test_normalize_non_square() -> None:
    """非方形图: x 按 w、y 按 h 各自归一化."""
    out = normalize_boxes([(50.0, 100.0, 100.0, 200.0)], 200, 400)
    assert out == [(0.25, 0.25, 0.5, 0.5, 1.0)]


def test_normalize_clamps_out_of_range() -> None:
    """服务端像素坐标可能轻微越界(千分制取整误差), 钳制到 [0,1]."""
    out = normalize_boxes([(0.0, -5.0, 120.0, 100.0)], 100, 100)
    assert out == [(0.0, 0.0, 1.0, 1.0, 1.0)]


def test_normalize_empty() -> None:
    """无框(<box>none</box> 场景) → 空列表."""
    assert normalize_boxes([], 100, 100) == []


def test_normalize_drops_reversed_x() -> None:
    """乱序框(x1>x2)是自回归输出垃圾, 丢弃而非规整(规整会凭空造框)."""
    assert normalize_boxes([(70.0, 20.0, 30.0, 40.0)], 100, 100) == []


def test_normalize_drops_reversed_y() -> None:
    assert normalize_boxes([(10.0, 70.0, 40.0, 30.0)], 100, 100) == []


def test_normalize_drops_zero_area() -> None:
    """零面积框(x1==x2 或 y1==y2)丢弃, 避免产出 w=0 的退化 YOLO 标注."""
    assert normalize_boxes([(50.0, 50.0, 50.0, 50.0)], 100, 100) == []
    assert normalize_boxes([(0.0, 0.0, 0.0, 50.0)], 100, 100) == []


def test_normalize_keeps_valid_among_degenerate() -> None:
    """退化框只影响自身, 不影响同批的有效框."""
    out = normalize_boxes(
        [(70.0, 20.0, 30.0, 40.0), (10.0, 20.0, 80.0, 90.0)], 100, 100
    )
    assert out == [(0.1, 0.2, 0.8, 0.9, 1.0)]


def test_dedup_identical_boxes() -> None:
    """官方已知重复框问题: 完全相同的框只留一个."""
    boxes = [(0.1, 0.1, 0.5, 0.5, 1.0), (0.1, 0.1, 0.5, 0.5, 1.0)]
    assert dedup_boxes(boxes) == [(0.1, 0.1, 0.5, 0.5, 1.0)]


def test_dedup_keeps_legitimate_overlap() -> None:
    """真实重叠目标(如前后遮挡行人 IoU~0.5)不得被去重."""
    boxes = [
        (0.1, 0.1, 0.5, 0.5, 1.0),
        (0.3, 0.3, 0.7, 0.7, 1.0),
    ]
    assert len(dedup_boxes(boxes)) == 2


def test_dedup_greedy_chain() -> None:
    """贪心抑制对"保留集"比对: A 抑制 B 与 C, 不因 B 已被抑制而放过 C."""
    a = (0.0, 0.0, 1.0, 1.0, 1.0)
    b = (0.0, 0.0, 1.0, 0.9, 1.0)  # IoU(A,B)=0.9 → 抑制
    c = (0.0, 0.1, 1.0, 1.0, 1.0)  # IoU(A,C)=0.9 → 也被 A 抑制
    assert dedup_boxes([a, b, c], iou_thr=0.85) == [a]


def test_dedup_chain_keeps_far_box() -> None:
    """A 抑制 B, 但远离 A 的 C 保留(贪心按保留集逐一比对)."""
    a = (0.0, 0.0, 0.4, 0.4, 1.0)
    b = (0.01, 0.01, 0.41, 0.41, 1.0)  # IoU(A,B)≈0.86 → 抑制
    c = (0.6, 0.6, 0.9, 0.9, 1.0)  # 远离 → 保留
    assert dedup_boxes([a, b, c], iou_thr=0.85) == [a, c]


def test_dedup_empty_and_single() -> None:
    assert dedup_boxes([]) == []
    assert dedup_boxes([(0.1, 0.1, 0.2, 0.2, 1.0)]) == [(0.1, 0.1, 0.2, 0.2, 1.0)]


def test_dedup_preserves_order() -> None:
    """输出保持输入相对顺序(稳定), det_mine 依赖框序可复现."""
    boxes = [
        (0.5, 0.5, 0.8, 0.8, 1.0),
        (0.1, 0.1, 0.3, 0.3, 1.0),
        (0.51, 0.51, 0.81, 0.81, 1.0),  # ≈第一框 → 抑制
    ]
    assert dedup_boxes(boxes) == [boxes[0], boxes[1]]
