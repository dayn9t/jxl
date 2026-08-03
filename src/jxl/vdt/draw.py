"""vdt 演示可视化库（spec 2026-08-02-vdt-demo）。

Functional Core：纯绘制函数（输入 canvas + 跟踪数据 → 就地画）；命令式 TrailBuffer
仅在渲染循环内累积尾迹。零模型依赖，可独立单测。cli.render_video 是其消费者。
"""

from __future__ import annotations

import colorsys
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from jvi.geo.point2d import Point
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size
from jxl.det.d2d import D2dObject
from jxl.vdt.types import Keypoints


@dataclass(frozen=True, slots=True)
class DrawOpts:
    """绘制开关（不可变；cli 层从命令行选项构造）。"""

    box: bool = True
    skeleton: bool = True
    trail: bool = True
    hud: bool = True
    trail_len: int = 30


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """按 track_id 稳定生成 BGR 颜色（HSV 黄金角色相分布，相邻 id 色相差大）。"""
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.9)
    return (int(b * 255), int(g * 255), int(r * 255))  # cv2 用 BGR


# COCO 17 关键点骨架边（0-indexed；鼻0/左眼1/右眼2/左耳3/右耳4/左肩5/右肩6/
# 左肘7/右肘8/左腕9/右腕10/左髋11/右髋12/左膝13/右膝14/左踝15/右踝16）。
COCO_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),   # 头部
    (5, 7), (7, 9),                   # 左臂
    (6, 8), (8, 10),                  # 右臂
    (5, 11), (6, 12),                 # 肩-髋
    (11, 13), (13, 15),               # 左腿
    (12, 14), (14, 16),               # 右腿
    (5, 6), (11, 12),                 # 肩间 / 髋间
)

# ---------------------------------------------------------------------------
# 单测（pytest 按文件发现；零模型依赖）
# ---------------------------------------------------------------------------


def test_draw_opts_defaults() -> None:
    opts = DrawOpts()
    assert opts.box and opts.skeleton and opts.trail and opts.hud
    assert opts.trail_len == 30


def test_color_for_id_stable_and_distinct() -> None:
    assert color_for_id(7) == color_for_id(7)  # 稳定
    colors = [color_for_id(i) for i in range(20)]
    assert len(set(colors)) == 20  # 前 20 个 id 无碰撞


def test_coco_skeleton_edges_valid() -> None:
    assert len(COCO_SKELETON_EDGES) > 0
    flat = [v for e in COCO_SKELETON_EDGES for v in e]
    assert all(0 <= v < 17 for v in flat), "端点必须 ∈ [0,17)"
    assert len(set(COCO_SKELETON_EDGES)) == len(COCO_SKELETON_EDGES), "无重复边"
