"""viz 纯函数单测: 缩放/画框/网格(合成小图, 无文件依赖)."""
from __future__ import annotations

from PIL import Image

from jxl.det.viz import RGB, draw_boxes, grid, scale_to_width


def test_scale_to_width_keeps_ratio() -> None:
    im = Image.new("RGB", (100, 50), (0, 0, 0))
    out = scale_to_width(im, 200)
    assert out.size == (200, 100)


def test_draw_boxes_marks_pixels() -> None:
    im = Image.new("RGB", (100, 100), (0, 0, 0))
    color: RGB = (255, 0, 0)
    out = draw_boxes(im, [(0.1, 0.1, 0.9, 0.9, 1.0)], color, width_px=3)
    # 框边缘像素变红
    assert out.getpixel((10, 10)) == color
    assert out.getpixel((50, 50)) == (0, 0, 0)  # 框内部未填


def test_grid_layout() -> None:
    tiles = [Image.new("RGB", (50, 30), (i, 0, 0)) for i in range(5)]
    out = grid(tiles, cols=2, tile_w=50)
    assert out.size == (100, 3 * 30)  # 3 行 x 2 列
