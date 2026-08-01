"""视频解码器（``OcvDecoder``）—— vdt 管线第一阶。

按 ``DecodeCfg.fps`` 对源视频等间隔抽帧，发射 ``(frame_idx, ts_ms, BGR image)``。
``frame_idx`` 是**采样计数器**（0 起连续递增，非源帧索引），供下游 IoU 跟踪按连续
帧关联；``ts_ms`` 是**源视频真实时间戳**（ReID 的 ttl/motion_radius 按秒，必须用源
时间，spec §4 Decoder 行）。

有状态、仅程序内构造（持 ``cv2.VideoCapture``，不可序列化、迭代器只能消费一次）。
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from jxl.vdt.types import DecodeCfg, DecodeError


class OcvDecoder:
    """opencv ``VideoCapture`` 解码器，可配置 fps 采样。

    有状态、仅程序内构造——持 ``cv2.VideoCapture``（不可序列化）；``__iter__`` 是
    一次性视频流语义，二次迭代 ``raise DecodeError``。

    属性：
        fps: 采样帧率（= ``cfg.fps``），供 Aggregator 记录到 ``Tracks.fps``。
        duration_ms: 源视频时长（ms），供 Aggregator 记录到 ``Tracks.duration_ms``。
    """

    fps: float
    duration_ms: int

    def __init__(self, video_path: str, cfg: DecodeCfg) -> None:
        """打开 ``video_path`` 并按 ``cfg.fps`` 计算采样步长。

        Args:
            video_path: 视频文件路径。
            cfg: 解码配置（``fps`` 为目标采样帧率）。

        Raises:
            DecodeError: 视频打不开 / 源 fps 非正 / 帧数非正（No Silent Degradation）。
        """
        cap = cv2.VideoCapture(video_path)
        # cv2 无 stub，isOpened 返回 bool；打不开立即失败，不静默回退。
        if not cap.isOpened():
            cap.release()
            raise DecodeError(f"无法打开视频: {video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0.0:
            cap.release()
            raise DecodeError(f"视频 fps 非正（损坏？）: {video_path}, fps={source_fps}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise DecodeError(
                f"视频帧数非正（损坏？）: {video_path}, frames={frame_count}"
            )

        self._cap = cap
        self._video_path = video_path
        self._source_fps = source_fps
        self._frame_count = frame_count
        # round(source_fps / cfg.fps)：每 N 源帧取一帧；下界 1（采样不得超源 fps）。
        self._sample_stride = max(1, round(source_fps / cfg.fps))
        self._consumed = False

        self.fps = cfg.fps
        self.duration_ms = round(frame_count / source_fps * 1000)

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        """按 ``sample_stride`` 抽帧，yield ``(frame_idx, ts_ms, BGR ndarray)``。

        ``frame_idx`` 为采样计数器（0 起连续）；``ts_ms`` 由源帧索引换算（源真实时间）。
        到达尾部或读帧失败即停止；迭代结束（含提前 break）``cap.release()``。
        迭代器只能消费一次。

        实现用**顺序 ``grab()`` + 条件 ``retrieve()``**，而非 ``cap.set(POS_FRAMES)`` seek：
        mp4v 等稀疏关键帧格式下，``set`` 每次 seek 会从最近 keyframe 全量重解码，实测
        750 帧需 244s；顺序 ``grab``（仅解封装不解码）跳过非采样帧、仅对采样帧 ``retrieve``
        解码，同 750 帧仅 1.5s（~160×）。对任意 stride 均快且正确。
        """
        if self._consumed:
            raise DecodeError(f"OcvDecoder 迭代器已消费，不可重复迭代: {self._video_path}")
        self._consumed = True

        cap = self._cap
        source_fps = self._source_fps
        stride = self._sample_stride
        try:
            frame_idx = 0
            src_idx = 0
            # grab() 推进一帧（仅解封装，廉价）；每逢采样位置 retrieve() 解码该帧。
            # 以 grab() 返回 False 作 EOF（比 CAP_PROP_FRAME_COUNT 更可靠——后者对部分编码不准）。
            while cap.grab():
                if src_idx % stride == 0:
                    ret, frame = cap.retrieve()
                    if not ret or frame is None:
                        break
                    ts_ms = round(src_idx / source_fps * 1000)
                    # frame 来自 cv2（无 stub→Any），asarray 既是运行时恒等（已是 ndarray）
                    # 又把类型窄化为 ndarray，满足 mypy 严格返回类型。
                    yield (frame_idx, ts_ms, np.asarray(frame))
                    frame_idx += 1
                src_idx += 1
            # spec §9：抽帧得 0 帧（损坏/不可解码，构造期 CAP_PROP_FRAME_COUNT 对部分
            # 编码不准、可能误报正）→ raise，不静默产出空 Tracks（No Silent Degradation）。
            if frame_idx == 0:
                raise DecodeError(
                    f"视频抽帧得 0 帧（损坏或不可解码）: {self._video_path}"
                )
        finally:
            cap.release()


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
