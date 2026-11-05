"""crop 契约测试——与 rs ulabel-utils/src/crop.rs 用例对齐（同名同义）。"""

from pathlib import Path

import pytest
from PIL import Image

from jxl.label.crop import (
    XYXY,
    be_uniform,
    crop_expanded,
    crop_rect_xyxy,
    expand_xyxy,
    save_expanded,
    xyxy_iou,
)
from jxl.label.meta import BorderExtend

ZERO = BorderExtend(left=0.0, right=0.0, top=0.0, bottom=0.0)


def test_be_uniform() -> None:
    be = be_uniform(0.1)
    assert (be.left, be.right, be.top, be.bottom) == (0.1, 0.1, 0.1, 0.1)
    assert be_uniform(0.0) == ZERO


def test_expand_uniform_per_axis() -> None:
    # 100x50 框: left/right 各 10（宽 10%），top/bottom 各 5（高 10%）
    out = expand_xyxy((10.0, 10.0, 110.0, 60.0), be_uniform(0.1), 1000, 1000)
    assert out == pytest.approx((0.0, 5.0, 120.0, 65.0))


def test_expand_negative_shrinks() -> None:
    # -0.25: 每边内缩 0.25×边长 → 100 框收到 [25,75]（半尺寸）
    out = expand_xyxy((0.0, 0.0, 100.0, 100.0), be_uniform(-0.25), 1000, 1000)
    assert out == pytest.approx((25.0, 25.0, 75.0, 75.0))


def test_expand_per_side_independent() -> None:
    # left=0.2 外延到 -20 被 clamp 到帧界 0；bottom=0.4 外延到 140（帧内不 clamp）
    be = BorderExtend(left=0.2, right=0.0, top=0.0, bottom=0.4)
    out = expand_xyxy((0.0, 0.0, 100.0, 100.0), be, 1000, 1000)
    assert out == pytest.approx((0.0, 0.0, 100.0, 140.0))


def test_expand_clamps_to_frame() -> None:
    out = expand_xyxy((0.0, 0.0, 100.0, 100.0), be_uniform(1.0), 100, 100)
    assert out == pytest.approx((0.0, 0.0, 100.0, 100.0))


def test_iou_ported_cases() -> None:
    b: XYXY = (0.1, 0.2, 0.5, 0.6)
    assert xyxy_iou(b, b) == 1.0
    assert xyxy_iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.9, 0.9)) == 0.0
    assert xyxy_iou((0.0, 0.0, 0.2, 0.2), (0.1, 0.0, 0.3, 0.2)) == pytest.approx(1 / 3)
    assert xyxy_iou((0.0, 0.0, 0.2, 0.2), (0.2, 0.0, 0.4, 0.2)) == 0.0
    assert xyxy_iou((0.1, 0.1, 0.1, 0.1), (0.1, 0.1, 0.1, 0.1)) == 0.0


def test_crop_rect_rounds_half_away_from_zero() -> None:
    # 2.5 → 3（半远离零，勿用内建 round——银行家舍入与 rs 差 1px）
    assert crop_rect_xyxy((2.5, 2.5, 7.5, 7.5), ZERO, 100, 100) == (3, 3, 5, 5)


def test_crop_rect_empty_returns_none() -> None:
    assert crop_rect_xyxy((0.0, 0.0, 100.0, 100.0), be_uniform(-0.6), 100, 100) is None


def test_crop_expanded_and_save(tmp_path: Path) -> None:
    im = Image.new("RGB", (10, 10), (200, 10, 10))
    cropped = crop_expanded(im, (2.0, 2.0, 8.0, 8.0), ZERO)
    assert cropped.size == (6, 6)
    dst = tmp_path / "sub" / "c.jpg"
    save_expanded(im, (2.0, 2.0, 8.0, 8.0), dst, ZERO)
    assert dst.exists() and dst.parent.is_dir()


def test_crop_expanded_empty_raises() -> None:
    im = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError, match="空框"):
        crop_expanded(im, (0.0, 0.0, 100.0, 100.0), be_uniform(-0.6))
