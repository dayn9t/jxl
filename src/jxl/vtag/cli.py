"""vtag Typer CLI —— 视频事件标签叠加（spec §6.2/§7/§8）。

imperative shell：参数校验 → 事件解析 → 视频逐帧 → ``draw_tags`` → ``VideoWriter``。
Functional Core（``parse_event``/``draw_tags``）在 ``overlay.py``，本模块仅编排与 IO。

No Silent Degradation（spec §10）：视频/字体不存在、``--event`` 缺失或非法、事件时段
越界、``--font-size`` 非正、视频无帧、``VideoWriter``/字体加载失败 → ``typer.BadParameter``
报错退出，绝不静默回退（如降级到 cv2.putText）。

console script：``vtag = "jxl.vtag.cli:app"``（单命令 app，直接 ``vtag <video> --event …``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from PIL import ImageFont

from jxl.io.video import VideoIoError, VideoReader, VideoWriter
from jxl.vtag.overlay import EventSpec, TagOpts, blink_visible, draw_tags, parse_event

app = typer.Typer(help="视频事件标签叠加（右下角中文红字）")

_DEFAULT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
"""spec §9 默认字体（系统 Noto Sans CJK，index 0；``--font`` 覆盖）。"""


def _default_out(video: Path) -> Path:
    """``<input>_tagged.<ext>``（spec §7）。"""
    return video.with_name(f"{video.stem}_tagged{video.suffix}")


def _resolve_font(font: Path | None) -> Path:
    """返回字体路径并校验存在；``--font`` 覆盖默认（spec §9/§10）。"""
    path = font if font is not None else _DEFAULT_FONT
    if not path.is_file():
        raise typer.BadParameter(f"字体文件不存在: {path}")
    return path


def _parse_events(raw_events: list[str]) -> list[EventSpec]:
    """解析 ``--event`` 串列表 → ``EventSpec`` 列表（spec §7/§10）。

    空 / 格式错 / ``起>=止`` / 负值 → ``BadParameter``。
    """
    if not raw_events:
        raise typer.BadParameter("至少需要一个 --event")
    events: list[EventSpec] = []
    for raw in raw_events:
        try:
            events.append(parse_event(raw))
        except ValueError as e:
            raise typer.BadParameter(f"--event {raw!r} 非法: {e}") from e
    return events


def _validate_event_ranges(events: list[EventSpec], duration_s: float) -> None:
    """事件时段不得超出视频时长（spec §10：不静默裁剪）。"""
    for e in events:
        if e.end > duration_s:
            raise typer.BadParameter(
                f"--event {e.name!r} 时段 {e.start}-{e.end}s 超出视频时长 {duration_s:.2f}s"
            )


def _load_font(opts: TagOpts) -> ImageFont.FreeTypeFont:
    """循环外加载字体一次（spec §8）。失败即报错，不回退 cv2.putText（No Silent Degradation）。"""
    try:
        return ImageFont.truetype(opts.font_path, opts.font_size)
    except (OSError, ValueError) as e:
        # OSError: 文件不可读/不存在；ValueError: font_size<=0（PIL 抛）。两者均 → BadParameter。
        raise typer.BadParameter(f"字体加载失败: {opts.font_path} ({e})") from e


@app.command()
def main(
    video: Annotated[Path, typer.Argument(help="输入视频")],
    event: Annotated[
        list[str], typer.Option("--event", help="'名称,起秒-止秒'，可重复（≥1）")
    ] = [],  # noqa: B006 — typer/click repeatable-option idiom；click 不就地修改默认值
    out: Annotated[
        Path | None, typer.Option("--out", help="输出（默认 <input>_tagged.<ext>）")
    ] = None,
    font_size: Annotated[int, typer.Option("--font-size", help="字号（默认 48）")] = 48,
    font: Annotated[
        Path | None, typer.Option("--font", help="字体文件（默认 Noto Sans CJK）")
    ] = None,
) -> None:
    """在每个事件时段的画面右下角叠加红色中文标签，写出 ``_tagged`` 视频（spec §8）。

    数据流：``events=[parse_event(e) ...]`` → 循环外加载字体一次 → ``VideoReader``
    逐帧 → ``active=[e.name for e in events if e.start <= t < e.end]``（``t=ts_ms/1000``）
    → ``draw_tags`` → ``VideoWriter.write``。
    """
    if not video.is_file():
        raise typer.BadParameter(f"视频不存在: {video}")
    if font_size <= 0:
        raise typer.BadParameter(f"--font-size 必须 > 0，实际 {font_size}")

    font_path = _resolve_font(font)
    events = _parse_events(event)
    out_path = out if out is not None else _default_out(video)
    opts = TagOpts(font_path=font_path, font_size=font_size)

    try:
        reader = VideoReader(str(video))
    except VideoIoError as e:
        raise typer.BadParameter(str(e)) from e

    try:
        with reader:
            _validate_event_ranges(events, reader.duration_ms / 1000.0)
            font_obj = _load_font(opts)
            with VideoWriter(out_path, reader.fps, reader.size) as writer:
                for _frame_idx, ts_ms, frame in reader:
                    t = ts_ms / 1000.0
                    active = [e.name for e in events if e.start <= t < e.end]
                    # 闪烁：仅显示相位画（spec 增补）；off 相位或无事件 → draw_tags([]) 原样返回。
                    visible = active if blink_visible(t, opts) else []
                    writer.write(draw_tags(frame, visible, font_obj, opts))
    except VideoIoError as e:
        raise typer.BadParameter(str(e)) from e


if __name__ == "__main__":
    app()


# ---------------------------------------------------------------------------
# 单测（pytest 按文件发现；端到端用 _make_synthetic_video 合成短视频，零真实视频依赖）
# 本模块是生产 console script 入口，pytest 仅 dev 可用——故 pytest 在各 test 函数内惰性
# import（同 jxl/vdt/draw.py 模式），生产 import 不触发。typer.testing（CliRunner）属
# typer runtime 依赖，可顶层 import。
# ---------------------------------------------------------------------------

from pathlib import Path as _Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from jxl.vtag.overlay import EventSpec as _EventSpec  # noqa: E402


def _make_video(path: _Path, fps: float = 5.0, frames: int = 10) -> None:
    """合成 320x240 短视频（委托 vdt 测试 helper；矩形仅在顶部，右下角留空便于断言）。"""
    from jxl.vdt.decoder import _make_synthetic_video

    _make_synthetic_video(str(path), fps=fps, frames=frames, size=(320, 240))


def _read_frames(path: _Path) -> list[np.ndarray]:
    """读回 mp4 全部帧（BGR ndarray）。"""
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(np.asarray(frame))
    finally:
        cap.release()
    return frames


def _has_red_bottom_right(frame: np.ndarray) -> bool:
    """右下角 ROI 是否存在红色标签像素（R>200 & G<50 & B<50，BGR 序）。"""
    h, w = frame.shape[:2]
    roi = frame[max(0, h - 80):, max(0, w - 160):]
    red = (roi[:, :, 2] > 200) & (roi[:, :, 1] < 50) & (roi[:, :, 0] < 50)
    return int(red.sum()) > 0


# --- helper 纯函数 --------------------------------------------------------


def test_default_out_appends_tagged() -> None:
    assert _default_out(_Path("/tmp/s.mp4")) == _Path("/tmp/s_tagged.mp4")
    assert _default_out(_Path("a/b/c.mkv")) == _Path("a/b/c_tagged.mkv")


def test_resolve_font_default_exists() -> None:
    """默认字体存在（系统 Noto CJK）—— _resolve_font(None) 返回默认路径（spec §9）。"""
    assert _resolve_font(None) == _DEFAULT_FONT


def test_resolve_font_missing_raises_bad_parameter() -> None:
    import pytest

    with pytest.raises(typer.BadParameter):
        _resolve_font(_Path("/nonexistent/font.ttf"))


def test_parse_events_ok() -> None:
    """parse_event 集成：_parse_events 正确解析多个事件串（spec §7）。"""
    events = _parse_events(["甲,0.5-1.5", "乙,1.0-2.0"])
    assert events == [
        _EventSpec(name="甲", start=0.5, end=1.5),
        _EventSpec(name="乙", start=1.0, end=2.0),
    ]


def test_parse_events_empty_raises_bad_parameter() -> None:
    import pytest

    with pytest.raises(typer.BadParameter):
        _parse_events([])


def test_parse_events_invalid_raises_bad_parameter() -> None:
    import pytest

    with pytest.raises(typer.BadParameter):
        _parse_events(["无逗号"])


def test_validate_event_ranges_out_of_bounds_raises() -> None:
    """事件 end 超出视频时长 → BadParameter（spec §10 不静默裁剪）。"""
    import pytest

    events = [_EventSpec(name="甲", start=0.0, end=10.0)]
    with pytest.raises(typer.BadParameter):
        _validate_event_ranges(events, duration_s=2.0)


def test_validate_event_ranges_within_bounds_ok() -> None:
    events = [_EventSpec(name="甲", start=0.5, end=1.5)]
    _validate_event_ranges(events, duration_s=2.0)  # 不抛


# --- CLI（CliRunner；单命令 app 直接调用，无子命令前缀） --------------------
# typer 把 BadParameter 包成 SystemExit(2)，故校验 exit_code == 2（精确于 !=0）。


def test_app_help_exit_zero() -> None:
    """console script 注册：vtag --help 正常，命令与 --event 选项可见。"""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "event" in result.stdout


def test_main_missing_event_bad_parameter(tmp_path: _Path) -> None:
    """无 --event → BadParameter（exit 2；spec §10：至少一个）。"""
    video = tmp_path / "s.mp4"
    _make_video(video)
    runner = CliRunner()
    result = runner.invoke(app, [str(video)])
    assert result.exit_code == 2


def test_main_invalid_event_bad_parameter(tmp_path: _Path) -> None:
    """--event 格式错 → BadParameter（exit 2）。"""
    video = tmp_path / "s.mp4"
    _make_video(video)
    runner = CliRunner()
    result = runner.invoke(app, [str(video), "--event", "无逗号"])
    assert result.exit_code == 2


def test_main_video_not_found_bad_parameter(tmp_path: _Path) -> None:
    """视频不存在 → BadParameter（exit 2）。"""
    runner = CliRunner()
    result = runner.invoke(
        app, [str(tmp_path / "nope.mp4"), "--event", "甲,0.5-1.5"]
    )
    assert result.exit_code == 2


def test_main_font_not_found_bad_parameter(tmp_path: _Path) -> None:
    """--font 不存在 → BadParameter（exit 2）。"""
    video = tmp_path / "s.mp4"
    _make_video(video)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [str(video), "--event", "甲,0.5-1.5", "--font", "/nonexistent/font.ttf"],
    )
    assert result.exit_code == 2


def test_main_font_size_non_positive_bad_parameter(tmp_path: _Path) -> None:
    """--font-size <= 0 → BadParameter（exit 2；覆盖 ValueError 逃逸缺口）。"""
    video = tmp_path / "s.mp4"
    _make_video(video)
    runner = CliRunner()
    result = runner.invoke(
        app, [str(video), "--event", "甲,0.5-1.5", "--font-size", "0"]
    )
    assert result.exit_code == 2


def test_main_event_out_of_bounds_bad_parameter(tmp_path: _Path) -> None:
    """事件 end 超视频时长 → BadParameter（exit 2；补 CLI 端到端覆盖 spec §10）。"""
    video = tmp_path / "s.mp4"
    _make_video(video, fps=5.0, frames=10)  # duration 2.0s
    runner = CliRunner()
    result = runner.invoke(
        app, [str(video), "--event", "甲,0.5-3.0"]  # end=3.0 > 2.0
    )
    assert result.exit_code == 2


def test_main_end_to_end(tmp_path: _Path) -> None:
    """端到端：合成视频 + 2 事件 → 可读 mp4；时段内帧 vs 时段外帧像素有差（spec §11）。"""
    video = tmp_path / "s.mp4"
    _make_video(video, fps=5.0, frames=10)  # ts=0..1800ms（duration 2.0s）
    out = tmp_path / "s_tagged.mp4"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            str(video),
            "--event", "甲,0.5-1.5",
            "--event", "乙,0.5-1.5",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, f"CLI 失败: {result.output}"

    assert out.is_file() and out.stat().st_size > 0
    frames = _read_frames(out)
    assert len(frames) >= 5, f"预期 ~10 帧，实得 {len(frames)}"

    tagged = [f for f in frames if _has_red_bottom_right(f)]
    untagged = [f for f in frames if not _has_red_bottom_right(f)]
    assert len(tagged) > 0, "应有事件时段内的叠加帧"
    assert len(untagged) > 0, "应有事件时段外的非叠加帧"
    # 时段内帧 vs 时段外帧的右下角 ROI 像素确有差异
    assert not np.array_equal(tagged[0][-80:, -160:], untagged[0][-80:, -160:])
