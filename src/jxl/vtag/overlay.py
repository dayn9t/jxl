"""vtag 中文事件标签叠加 —— Functional Core（spec §6.1）。

纯函数：``parse_event``（事件串解析）+ ``draw_tags``（右下角竖排红字白描边）。
零视频依赖，可独立单测；``cli`` 是其消费者（渲染循环外加载字体一次，传入）。

设计要点（j-design-principles）：
- Functional Core / 不可变优先：``EventSpec``/``TagOpts`` frozen+slots；``draw_tags``
  不改输入 ``canvas``（PIL 来回拷贝），``names`` 空 → 原样返回（零开销）。
- No Silent Degradation：``parse_event`` 遇非法输入即 ``ValueError``，不猜测不回退。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True, slots=True)
class EventSpec:
    """用户指定的一个事件时段（spec §6.1）。"""

    name: str
    start: float  # 秒
    end: float  # 秒，start < end


@dataclass(frozen=True, slots=True)
class TagOpts:
    """叠加视觉参数（spec §9；右下角硬默认，仅字号/字体可调）。

    闪烁：标签按 ``blink_on`` 秒显示 / ``blink_off`` 秒消失循环，基于视频绝对时间
    ``t`` 的相位（所有事件同步闪烁，spec 增补）。
    """

    font_path: Path
    font_size: int = 96
    color_rgb: tuple[int, int, int] = (255, 0, 0)  # 红填充
    stroke_rgb: tuple[int, int, int] = (255, 255, 255)  # 白描边
    stroke_width: int = 2
    margin: int = 20
    blink_on: float = 0.25  # 显示时长（秒）
    blink_off: float = 0.525  # 消失时长（秒）


def parse_event(s: str) -> EventSpec:
    """``'打架,2.0-5.0'`` → ``EventSpec('打架', 2.0, 5.0)``。

    格式 ``<名称>,<起>-<止>``（spec §7）：名称不含逗号；起/止为非负浮点秒；``起 < 止``。

    非法（无逗号 / 非数 / 非有限数 nan·inf / ``起>=止`` / 负值 / 名称空）→ ``ValueError``。
    负值由时间段拆分天然拒绝——非负数无 ``-`` 前缀，故时间段串恰含一个 ``-``；
    出现负号即拆出 ≠2 段 → 格式错。nan/inf 经 ``float()`` 解析成功但会穿透后续比较
    （nan 比较恒 False）致事件静默丢弃，故用 ``math.isfinite`` 显式拒绝（spec §10）。
    """
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"事件格式应为 '<名称>,<起>-<止>'，实际 {s!r}")
    name, time_str = parts
    if not name:
        raise ValueError(f"事件名称为空: {s!r}")

    time_parts = time_str.split("-")
    if len(time_parts) != 2:
        raise ValueError(f"时间段格式应为 '<起>-<止>'，实际 {time_str!r}")
    try:
        start = float(time_parts[0])
        end = float(time_parts[1])
    except ValueError as e:
        raise ValueError(f"时间段非数值: {time_str!r}") from e
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(f"时间段必须为有限数值: {time_str!r}")
    if start >= end:
        raise ValueError(f"起必须 < 止: 起={start} 止={end}")
    return EventSpec(name=name, start=start, end=end)


def blink_visible(t: float, opts: TagOpts) -> bool:
    """``t``（秒）是否落在闪烁的显示相位（on）。

    周期 = ``blink_on + blink_off``；``t % 周期 < blink_on`` → 显示。
    所有事件共享同一绝对时间相位（同步闪烁）。``blink_on <= 0`` → 恒不显示。
    """
    if opts.blink_on <= 0:
        return False
    period = opts.blink_on + opts.blink_off
    if period <= 0:
        return False
    return (t % period) < opts.blink_on


def draw_tags(
    canvas: np.ndarray,
    names: list[str],
    font: ImageFont.FreeTypeFont,
    opts: TagOpts,
) -> np.ndarray:
    """右下角竖排红字白描边叠加（spec §6.1/§9）。

    - ``names`` 为空 → 原样返回 ``canvas``（零开销，不转 PIL）。
    - 否则 cv2(BGR)→PIL(RGB)→``ImageDraw.text``（``fill=color_rgb``、
      ``stroke_width``/``stroke_fill=stroke_rgb``、``anchor='rb'``）→转回 BGR。
      不修改输入 ``canvas``（PIL 来回拷贝）。
    - 多事件竖排，``names[0]`` 最靠下；行间距随字号（spec §8/§9）。
    """
    if not names:
        return canvas

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    height, width = canvas.shape[:2]
    # 行间距随字号（spec §9）；下界 1 保证多事件总有可见间隙。
    gap = max(1, opts.font_size // 4)
    y_bottom = height - opts.margin  # names[0] 底边贴右下角
    for name in names:
        bbox = draw.textbbox(
            (0.0, 0.0), name, font=font, anchor="la", stroke_width=opts.stroke_width
        )
        text_h = int(bbox[3] - bbox[1])
        draw.text(
            (width - opts.margin, y_bottom),
            name,
            font=font,
            anchor="rb",
            fill=opts.color_rgb,
            stroke_width=opts.stroke_width,
            stroke_fill=opts.stroke_rgb,
        )
        y_bottom -= text_h + gap
    # cv2 无 stub（返回 Any）；np.asarray 既是运行时恒等又把类型窄化为 ndarray，
    # 满足 mypy warn_return_any（同 jxl/io/video.py 模式）。
    return np.asarray(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# 单测（pytest 按文件发现；零视频依赖，仅需系统 CJK 字体）
# 本模块是生产模块（被 cli import），pytest 仅 dev 可用——故 pytest 在各 test 函数内
# 惰性 import（同 jxl/vdt/draw.py 模式），生产 import 不触发。
# ---------------------------------------------------------------------------

from pathlib import Path as _Path  # noqa: E402

# spec §9 默认字体；测试需真实 CJK 字形（「打架/跌倒」）。
_CJK_FONT = _Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def _red_mask(arr: np.ndarray) -> np.ndarray:
    """BGR 数组的红色像素掩码（R>200 & G<50 & B<50）。"""
    return (arr[:, :, 2] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 0] < 50)


def _red_bands(arr: np.ndarray) -> list[tuple[int, int]]:
    """红色像素的连续行带（每带 = 一个标签的纵向占据区间），按 y 升序。"""
    rows = np.asarray(_red_mask(arr).any(axis=1))
    n = rows.shape[0]
    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for r in range(n):
        v = bool(rows[r])
        if v and not in_band:
            start = r
            in_band = True
        elif not v and in_band:
            bands.append((start, r - 1))
            in_band = False
    if in_band:
        bands.append((start, n - 1))
    return bands


# --- parse_event ---------------------------------------------------------


def test_parse_event_ok() -> None:
    assert parse_event("打架,2.0-5.0") == EventSpec(name="打架", start=2.0, end=5.0)


def test_parse_event_zero_start_ok() -> None:
    """0 是合法起点（非负，非「负值」）。"""
    e = parse_event("跌倒,0.0-1.5")
    assert e.start == 0.0
    assert e.end == 1.5
    assert e.name == "跌倒"


def test_parse_event_no_comma_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_event("打架")


def test_parse_event_extra_comma_raises() -> None:
    """名称含逗号（>1 逗号）→ 格式错。"""
    import pytest

    with pytest.raises(ValueError):
        parse_event("打,架,2.0-5.0")


def test_parse_event_non_numeric_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_event("打架,abc-5.0")
    with pytest.raises(ValueError):
        parse_event("打架,2.0-xyz")


def test_parse_event_non_finite_raises() -> None:
    """nan/inf 经 float() 解析成功但穿透比较致事件静默丢弃 → 显式拒绝（spec §10）。"""
    import pytest

    with pytest.raises(ValueError):
        parse_event("打架,nan-5.0")
    with pytest.raises(ValueError):
        parse_event("打架,2.0-nan")
    with pytest.raises(ValueError):
        parse_event("打架,inf-5.0")


def test_parse_event_start_ge_end_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_event("打架,5.0-2.0")
    with pytest.raises(ValueError):
        parse_event("打架,3.0-3.0")


def test_parse_event_negative_raises() -> None:
    """负值 → 时间段拆出 ≠2 段 → ValueError（spec §10 严格不回退）。"""
    import pytest

    with pytest.raises(ValueError):
        parse_event("打架,-1.0-5.0")
    with pytest.raises(ValueError):
        parse_event("打架,1.0--5.0")


def test_parse_event_empty_name_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_event(",2.0-5.0")


def test_tag_opts_defaults() -> None:
    opts = TagOpts(font_path=_CJK_FONT)
    assert opts.font_size == 96
    assert opts.color_rgb == (255, 0, 0)
    assert opts.stroke_rgb == (255, 255, 255)
    assert opts.stroke_width == 2
    assert opts.margin == 20
    assert opts.blink_on == 0.25
    assert opts.blink_off == 0.525


def test_blink_visible_phase() -> None:
    """闪烁相位：on(0.25) → off(0.525) 循环，周期 0.775s（基于绝对 t）。"""
    opts = TagOpts(font_path=_CJK_FONT)  # 默认 on=0.25 off=0.525
    assert blink_visible(0.0, opts) is True  # 相位 0 < 0.25 → 显示
    assert blink_visible(0.24, opts) is True  # 仍 on
    assert blink_visible(0.25, opts) is False  # 边界进入 off
    assert blink_visible(0.5, opts) is False  # off 中
    assert blink_visible(0.775, opts) is True  # 新周期开始（0%周期=0 < 0.25）
    assert blink_visible(1.0, opts) is True  # 1.0%0.775=0.225 < 0.25 → on
    # blink_on<=0 → 恒不显示
    off_opts = TagOpts(font_path=_CJK_FONT, blink_on=0.0)
    assert blink_visible(0.0, off_opts) is False


# --- draw_tags -----------------------------------------------------------


def test_draw_tags_empty_returns_identical() -> None:
    """names 空 → 原样返回同一对象（零开销，不转 PIL）。"""
    canvas = np.full((50, 50, 3), 128, np.uint8)
    font = ImageFont.truetype(_CJK_FONT, 48)
    opts = TagOpts(font_path=_CJK_FONT, font_size=48)
    out = draw_tags(canvas, [], font, opts)
    assert out is canvas
    assert np.array_equal(out, canvas)


def test_draw_tags_with_event_paints_bottom_right() -> None:
    """有事件 → 右下角 ROI 出红（R>200 & G<50 & B<50），左上角 ROI 不变。"""
    canvas = np.full((200, 300, 3), 128, np.uint8)
    top_left_before = canvas[:60, :60].copy()
    font = ImageFont.truetype(_CJK_FONT, 48)
    opts = TagOpts(font_path=_CJK_FONT, font_size=48)
    out = draw_tags(canvas, ["打架"], font, opts)

    red = _red_mask(out)
    # 右下角 ROI 存在红色像素
    assert int(red[-80:, -160:].sum()) > 0
    # 左上角 ROI 无红色像素
    assert int(red[:60, :60].sum()) == 0
    # 左上角 ROI 像素未变（PIL 来回拷贝不扰动未绘制区）
    assert np.array_equal(out[:60, :60], top_left_before)
    # 输入 canvas 未被修改
    assert np.array_equal(canvas[:60, :60], top_left_before)


def test_draw_tags_does_not_mutate_input() -> None:
    """Functional Core：draw_tags 不就地改输入 canvas。"""
    canvas = np.full((200, 300, 3), 128, np.uint8)
    before = canvas.copy()
    font = ImageFont.truetype(_CJK_FONT, 48)
    opts = TagOpts(font_path=_CJK_FONT, font_size=48)
    draw_tags(canvas, ["打架"], font, opts)
    assert np.array_equal(canvas, before)


def test_draw_tags_multi_event_vertical_nonoverlap() -> None:
    """多事件竖排：两个标签的 y 区间不重叠（首个事件最靠下）。"""
    canvas = np.full((300, 300, 3), 128, np.uint8)
    font = ImageFont.truetype(_CJK_FONT, 48)
    opts = TagOpts(font_path=_CJK_FONT, font_size=48)
    out = draw_tags(canvas, ["打架", "跌倒"], font, opts)

    bands = _red_bands(out)
    assert len(bands) == 2, f"应有 2 条红色行带，实际 {bands}"
    # 升序两带不重叠且存在间隙（上带底 < 下带顶）
    assert bands[0][1] < bands[1][0], f"两标签 y 重叠: {bands}"


def test_draw_tags_first_event_at_bottom() -> None:
    """names[0] 最靠下：单事件底边 ≈ 高度 - margin；再加一个事件后底边不变。"""
    font = ImageFont.truetype(_CJK_FONT, 48)
    opts = TagOpts(font_path=_CJK_FONT, font_size=48)
    canvas = np.full((300, 300, 3), 128, np.uint8)

    one = draw_tags(canvas.copy(), ["打架"], font, opts)
    two = draw_tags(canvas.copy(), ["打架", "跌倒"], font, opts)

    one_bands = _red_bands(one)
    two_bands = _red_bands(two)
    assert len(one_bands) == 1
    assert len(two_bands) == 2
    # names[0]（打架）在两种情形下都处于最下方 → 最底带的底边一致
    assert one_bands[0][1] == two_bands[-1][1]
