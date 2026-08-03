"""视频解码器（``OcvDecoder``）—— vdt 管线第一阶。

薄适配 ``jxl.io.video.VideoReader``：按 ``DecodeCfg.fps`` 等间隔抽帧，发射
``(frame_idx, ts_ms, BGR image)``。``frame_idx`` 是**采样计数器**（0 起连续递增，
非源帧索引），供下游 IoU 跟踪按连续帧关联；``ts_ms`` 是**源视频真实时间戳**
（ReID 的 ttl/motion_radius 按秒，必须用源时间，spec §4 Decoder 行）。

解码/采样逻辑由共享层 ``VideoReader`` 单一提供（spec §3 单一数据源——解码/编码逻辑
仅此处一份）；本类仅做 ``VideoIoError → DecodeError`` 适配 + ``fps``/``duration_ms``/
``size`` 转发。公共接口（构造签名、``__iter__`` yield 元组、属性）不变 → pipeline/cli
调用方无需改。

有状态、仅程序内构造（持 ``VideoReader`` → ``cv2.VideoCapture``，不可序列化、迭代器
只能消费一次）。
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from jxl.io.video import VideoIoError, VideoReader
from jxl.vdt.types import DecodeCfg, DecodeError


class OcvDecoder:
    """opencv 视频解码器适配器——委托 ``VideoReader``，可配置 fps 采样。

    有状态、仅程序内构造——持 ``VideoReader``（不可序列化）；``__iter__`` 是一次性
    视频流语义，二次迭代 ``raise DecodeError``。

    属性：
        fps: 采样帧率（= ``cfg.fps``），供 Aggregator 记录到 ``Tracks.fps``。
        duration_ms: 源视频时长（ms），供 Aggregator 记录到 ``Tracks.duration_ms``。
        size: ``(width, height)``。
    """

    def __init__(self, video_path: str, cfg: DecodeCfg) -> None:
        """打开 ``video_path`` 并按 ``cfg.fps`` 采样。

        Args:
            video_path: 视频文件路径。
            cfg: 解码配置（``fps`` 为目标采样帧率）。

        Raises:
            DecodeError: 视频打不开 / 源 fps 非正 / 帧数非正 / 尺寸非法 /
                sample_fps 非正（No Silent Degradation；由 ``VideoReader`` 检测，
                此处适配为 vdt 域的 ``DecodeError``）。
        """
        try:
            self._reader = VideoReader(video_path, sample_fps=cfg.fps)
        except VideoIoError as e:
            raise DecodeError(str(e)) from e
        self._video_path = video_path

    @property
    def fps(self) -> float:
        """采样帧率（= ``cfg.fps``），供 Aggregator 记录到 ``Tracks.fps``。"""
        return self._reader.fps

    @property
    def duration_ms(self) -> int:
        """源视频时长（ms），供 Aggregator 记录到 ``Tracks.duration_ms``。"""
        return self._reader.duration_ms

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)``。"""
        return self._reader.size

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        """转发 ``VideoReader`` 迭代，yield ``(frame_idx, ts_ms, BGR ndarray)``。

        ``frame_idx`` 为采样计数器（0 起连续）；``ts_ms`` 由源帧索引换算（源真实时间）。
        迭代器只能消费一次；``VideoReader`` 的「抽帧得 0 帧 / 重复迭代」``VideoIoError``
        适配为 ``DecodeError``。
        """
        try:
            yield from self._reader
        except VideoIoError as e:
            raise DecodeError(str(e)) from e


# ---------------------------------------------------------------------------
# 单测（自包含：cv2.VideoWriter 合成短视频，零真实模型/视频依赖）
# ---------------------------------------------------------------------------


def _make_synthetic_video(
    path: str, *, fps: float = 25.0, frames: int = 25, size: tuple[int, int] = (64, 64)
) -> None:
    """合成纯色背景 + 逐帧移动矩形的短视频（供解码器测试）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    w, h = size
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise AssertionError(f"无法构造合成视频: {path}")
    try:
        for i in range(frames):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            x0 = (i * 2) % (w - 10)
            cv2.rectangle(canvas, (x0, 10), (x0 + 10, 30), (0, 255, 0), -1)
            writer.write(canvas)
    finally:
        writer.release()


def test_full_fps_sampling() -> None:
    """fps=25（=源）采样 → frame_idx 连续 0..N、ts_ms 单调、duration_ms>0。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        video = f"{tmp}/s.mp4"
        _make_synthetic_video(video, fps=25.0, frames=25)

        dec = OcvDecoder(video, DecodeCfg(fps=25.0))
        items = list(dec)

        assert dec.duration_ms == round(25 / 25.0 * 1000)  # 1000ms
        assert [idx for idx, _, _ in items] == list(range(len(items)))
        ts_list = [ts for _, ts, _ in items]
        assert ts_list == sorted(ts_list), "ts_ms 应单调递增"
        assert ts_list[0] == 0, "首帧 ts_ms 应 ≈0"
        assert len(items) >= 20, f"预期 ~25 帧（容许编码损耗），实得 {len(items)}"
        for _, _, img in items:
            assert img.ndim == 3 and img.shape[2] == 3


def test_downsample_fps() -> None:
    """fps=5 采样 25fps/25帧视频 → 抽帧数 ≈5、ts_ms 间隔 ≈200ms。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        video = f"{tmp}/s.mp4"
        _make_synthetic_video(video, fps=25.0, frames=25)

        dec = OcvDecoder(video, DecodeCfg(fps=5.0))
        items = list(dec)

        # stride = round(25/5) = 5；range(0,25,5) = [0,5,10,15,20] → 5 帧（允许编码 ±1）。
        assert abs(len(items) - 5) <= 1, f"预期 ~5 采样帧，实得 {len(items)}"
        ts_list = [ts for _, ts, _ in items]
        if len(ts_list) >= 2:
            gap = ts_list[1] - ts_list[0]
            assert abs(gap - 200) <= 20, f"ts_ms 间隔应 ≈200ms，实得 {gap}ms"
        assert dec.fps == 5.0


def test_invalid_path_raises() -> None:
    """打不开的路径 → DecodeError（No Silent Degradation）。"""
    import pytest

    with pytest.raises(DecodeError):
        OcvDecoder("/nonexistent/video.mp4", DecodeCfg(fps=10.0))


def test_double_iteration_raises() -> None:
    """迭代器只能消费一次（视频流语义）。"""
    import tempfile

    import pytest

    with tempfile.TemporaryDirectory() as tmp:
        video = f"{tmp}/s.mp4"
        _make_synthetic_video(video, fps=10.0, frames=10)
        dec = OcvDecoder(video, DecodeCfg(fps=10.0))
        first = list(dec)
        assert len(first) > 0
        with pytest.raises(DecodeError):
            list(dec)
