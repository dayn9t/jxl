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
    id: bool = True
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


def _box_rect_px(
    canvas: np.ndarray, obj: D2dObject
) -> tuple[int, int, int, int]:
    """归一化 rect → 像素框四角 ``(x0, y0, x1, y1)``（框绘制与 ID 标签定位共享）。"""
    img_h, img_w = canvas.shape[:2]
    r = obj.rect.absolutize(Size.new(img_w, img_h)).round()
    return int(r.x), int(r.y), int(r.x + r.width), int(r.y + r.height)


def draw_track_box(
    canvas: np.ndarray, obj: D2dObject, color: tuple[int, int, int]
) -> None:
    """画检测框（归一化 rect → 像素），不含 ID 标签（见 :func:`draw_track_id`）。"""
    x0, y0, x1, y1 = _box_rect_px(canvas, obj)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)


def draw_track_id(
    canvas: np.ndarray, obj: D2dObject, color: tuple[int, int, int]
) -> None:
    """画左上角 ID 标签（带色块背景 + 黑字 ``#id``）。"""
    x0, y0, _, _ = _box_rect_px(canvas, obj)
    label = f"#{obj.id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
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


def _fade(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """颜色按 alpha 向黑衰减（alpha=1 原色，alpha=0 全黑）。"""
    return tuple(int(c * alpha) for c in color)  # type: ignore[return-value]


class TrailBuffer:
    """per-id 归一化中心点环形缓冲（命令式累积；``draw`` 是纯绘制）。

    有状态、仅程序内构造（非 pydantic）；渲染循环每帧 ``push``，``draw`` 画衰减尾迹。
    """

    def __init__(self, max_len: int) -> None:
        if max_len <= 0:
            raise ValueError(f"trail_len 必须 > 0，实际 {max_len}")
        self._max_len = max_len
        self._buf: dict[int, deque[Point]] = {}

    def push(self, track_id: int, center: Point) -> None:
        """记录某 id 本帧的归一化中心；超过 max_len 自动丢最旧。"""
        d = self._buf.get(track_id)
        if d is None:
            d = deque[Point](maxlen=self._max_len)
            self._buf[track_id] = d
        d.append(center)

    def draw(self, canvas: np.ndarray) -> None:
        """画所有 id 的尾迹：逐段按年龄衰减 alpha（旧 0.2→新 1.0）、线宽（旧 1→新 2）。"""
        img_h, img_w = canvas.shape[:2]
        for tid, pts in self._buf.items():
            color = color_for_id(tid)
            n = len(pts)
            if n < 2:
                continue
            for i in range(1, n):
                age = (n - 1 - i) / (n - 1)  # 0=最新段，1=最旧段
                alpha = 1.0 - 0.8 * age  # 新 1.0 → 旧 0.2
                thick = 2 if alpha > 0.6 else 1
                p0 = (int(pts[i - 1].x * img_w), int(pts[i - 1].y * img_h))
                p1 = (int(pts[i].x * img_w), int(pts[i].y * img_h))
                cv2.line(canvas, p0, p1, _fade(color, alpha), thick, cv2.LINE_AA)


def draw_hud(
    canvas: np.ndarray,
    frame_idx: int,
    ts_ms: int,
    n_objects: int,
    tracker_mode: str,
) -> None:
    """顶部信息条（半透明黑底白字）：帧号 / 时间 / 目标数 / 跟踪模式。"""
    text = f"f#{frame_idx}  {ts_ms / 1000:.1f}s  |  {n_objects} objs  |  {tracker_mode}"
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    x1, y1 = tw + 20, th + 16
    canvas[0:y1, 0:x1] = (canvas[0:y1, 0:x1] // 2)  # 局部变暗≈半透明黑底
    cv2.putText(
        canvas, text, (10, th + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )


def render_demo_frame(
    canvas: np.ndarray,
    objects: list[D2dObject],
    kpts: list[Keypoints | None],
    trails: TrailBuffer,
    frame_idx: int,
    ts_ms: int,
    tracker_mode: str,
    opts: DrawOpts,
) -> None:
    """按 opts 总装绘制一帧（顺序 skeleton→box→id→trail→hud；box 压在骨架上，
    ID 标签压在框上可读；HUD 顶层不被遮）。"""
    n_valid = 0
    for ob, kp in zip(objects, kpts, strict=False):
        if ob.id == 0:
            continue
        n_valid += 1
        color = color_for_id(ob.id)
        if opts.skeleton:
            draw_pose_skeleton(canvas, kp, color)
        if opts.box:
            draw_track_box(canvas, ob, color)
        if opts.id:
            draw_track_id(canvas, ob, color)
    if opts.trail:
        trails.draw(canvas)
    if opts.hud:
        draw_hud(canvas, frame_idx, ts_ms, n_valid, tracker_mode)


# ---------------------------------------------------------------------------
# 单测（pytest 按文件发现；零模型依赖）
# ---------------------------------------------------------------------------


def test_draw_opts_defaults() -> None:
    opts = DrawOpts()
    assert opts.box and opts.id and opts.skeleton and opts.trail and opts.hud
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


def test_draw_track_id_paints() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    ob = D2dObject(id=5, cls=0, conf=1.0, rect=Rect.new(0.1, 0.1, 0.2, 0.2))
    draw_track_id(canvas, ob, (0, 255, 0))
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


def test_trail_buffer_ring_drops_oldest() -> None:
    tb = TrailBuffer(3)
    for i in range(5):
        tb.push(1, Point(x=i / 10, y=i / 10))
    assert len(tb._buf[1]) == 3  # 环形：只保留最近 3
    assert tb._buf[1][-1] == Point(x=0.4, y=0.4)  # 最后 push 的是 i=4


def test_trail_buffer_invalid_len_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        TrailBuffer(0)


def test_trail_buffer_draw_no_crash() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    tb = TrailBuffer(3)
    tb.push(1, Point(x=0.5, y=0.5))
    tb.push(1, Point(x=0.51, y=0.5))
    tb.draw(canvas)  # ≥2 点才画线；不崩
    assert int(canvas.sum()) > 0


def test_draw_hud_paints_top_bar() -> None:
    canvas = np.zeros((100, 200, 3), np.uint8)
    draw_hud(canvas, 5, 1000, 3, "iou")
    assert int(canvas[:30].sum()) > 0  # 顶部条被画


def test_render_demo_frame_all_on_paints() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    ob = D2dObject(id=1, cls=0, conf=1.0, rect=Rect.new(0.1, 0.1, 0.2, 0.2))
    kpts = Keypoints(pts=[Point(x=0.5, y=0.5)] * 17, conf=[0.9] * 17)
    tb = TrailBuffer(3)
    tb.push(1, Point(x=0.5, y=0.5))
    tb.push(1, Point(x=0.51, y=0.5))
    render_demo_frame(canvas, [ob], [kpts], tb, 0, 0, "iou", DrawOpts())
    assert int(canvas.sum()) > 0


def test_render_demo_frame_all_off_no_paint() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    opts = DrawOpts(box=False, id=False, skeleton=False, trail=False, hud=False)
    render_demo_frame(canvas, [], [], TrailBuffer(3), 0, 0, "iou", opts)
    assert int(canvas.sum()) == 0


def test_render_demo_frame_id_toggle_changes_pixels() -> None:
    """opts.id 开关实际改变绘制：id=True 比 id=False 多画标签像素。"""

    def _render(opts: DrawOpts) -> int:
        canvas = np.zeros((100, 100, 3), np.uint8)
        ob = D2dObject(id=7, cls=0, conf=1.0, rect=Rect.new(0.1, 0.1, 0.2, 0.2))
        render_demo_frame(canvas, [ob], [None], TrailBuffer(3), 0, 0, "iou", opts)
        return int(canvas.sum())

    on = _render(DrawOpts(box=True, id=True, skeleton=False, trail=False, hud=False))
    off = _render(DrawOpts(box=True, id=False, skeleton=False, trail=False, hud=False))
    assert on > off  # id 开启时多画了标签


def test_render_demo_frame_skips_id0_sentinel() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    ob0 = D2dObject(id=0, cls=0, conf=1.0, rect=Rect.new(0.1, 0.1, 0.2, 0.2))
    opts = DrawOpts(box=True, skeleton=False, trail=False, hud=False)
    render_demo_frame(canvas, [ob0], [None], TrailBuffer(3), 0, 0, "iou", opts)
    assert int(canvas.sum()) == 0  # id=0 哨兵不画
