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


def draw_track_box(
    canvas: np.ndarray, obj: D2dObject, color: tuple[int, int, int]
) -> None:
    """画检测框（归一化 rect → 像素）+ 左上角 ID 标签（带色块背景）。"""
    img_h, img_w = canvas.shape[:2]
    r = obj.rect.absolutize(Size.new(img_w, img_h)).round()
    x0, y0 = int(r.x), int(r.y)
    x1, y1 = int(r.x + r.width), int(r.y + r.height)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
    label = f"#{obj.id}"
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    y_lbl = max(y0, th + 2)
    cv2.rectangle(canvas, (x0, y_lbl - th - 4), (x0 + tw + 6, y_lbl + 2), color, -1)
    cv2.putText(
        canvas, label, (x0 + 2, y_lbl - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )


def draw_pose_skeleton(
    canvas: np.ndarray,
    kpts: Keypoints | None,
    color: tuple[int, int, int],
    kconf: float = 0.35,
) -> None:
    """画 COCO 17 点骨架：可见点（conf≥kconf）画圆点，两端皆可见的边画线。

    ``kpts is None`` 或全低置信 → 不画。
    """
    if kpts is None:
        return
    img_h, img_w = canvas.shape[:2]
    vis = [c >= kconf for c in kpts.conf]
    if not any(vis):
        return
    for i, p in enumerate(kpts.pts):
        if vis[i]:
            cv2.circle(canvas, (int(p.x * img_w), int(p.y * img_h)), 4, color, -1)
    for a, b in COCO_SKELETON_EDGES:
        if a < len(vis) and b < len(vis) and vis[a] and vis[b]:
            pa = (int(kpts.pts[a].x * img_w), int(kpts.pts[a].y * img_h))
            pb = (int(kpts.pts[b].x * img_w), int(kpts.pts[b].y * img_h))
            cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)


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


def test_draw_track_box_paints() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    ob = D2dObject(id=5, cls=0, conf=1.0, rect=Rect.new(0.1, 0.1, 0.2, 0.2))
    draw_track_box(canvas, ob, (0, 255, 0))
    assert int(canvas.sum()) > 0


def test_draw_pose_skeleton_none_no_change() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    draw_pose_skeleton(canvas, None, (0, 255, 0))
    assert int(canvas.sum()) == 0


def test_draw_pose_skeleton_low_conf_no_paint() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    kpts = Keypoints(pts=[Point(x=0.5, y=0.5)] * 17, conf=[0.1] * 17)
    draw_pose_skeleton(canvas, kpts, (0, 255, 0))
    assert int(canvas.sum()) == 0


def test_draw_pose_skeleton_visible_paints() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    kpts = Keypoints(pts=[Point(x=0.5, y=0.5)] * 17, conf=[0.9] * 17)
    draw_pose_skeleton(canvas, kpts, (0, 255, 0))
    assert int(canvas.sum()) > 0
