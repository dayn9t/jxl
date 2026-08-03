"""共享视频 IO 层 —— ``VideoReader`` / ``VideoWriter``（spec §4）。

vdt 与 vtag 共用的零业务依赖视频读写。提炼自 ``jxl/vdt/decoder.py`` 的 ``OcvDecoder``
（grab/retrieve 优化，避免 ``cap.set`` 重解码）与 ``jxl/vdt/cli.py`` ``render_video``
的内联 ``cv2.VideoWriter``。

**不依赖任何业务子包**（spec §3：单一数据源——解码/编码逻辑仅此处一份）。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


class VideoIoError(Exception):
    """视频 IO 错误。

    覆盖：打不开 / 源 fps 非正 / 帧数非正 / 尺寸非法 / sample_fps 非正 /
    抽帧得 0 帧 / ``VideoWriter`` 打开失败（No Silent Degradation，严格不回退）。
    """


class VideoReader:
    """逐帧视频解码器（spec §4）。

    ``sample_fps=None`` 读全部帧（源 fps）；否则按 ``sample_fps`` 等间隔采样。

    沿用 ``OcvDecoder`` 的**顺序 ``grab()`` + 条件 ``retrieve()``** 优化——mp4v 等
    稀疏关键帧格式下 ``cap.set(POS_FRAMES)`` 每次 seek 会从最近 keyframe 全量重解码
    （实测 750 帧 244s），而顺序 ``grab``（仅解封装不解码）跳过非采样帧、仅对采样帧
    ``retrieve`` 解码（同 750 帧 1.5s，~160×）。对任意 stride 均快且正确。

    有状态、仅程序内构造——持 ``cv2.VideoCapture``（不可序列化）；``__iter__`` 是
    一次性视频流语义，二次迭代 ``raise VideoIoError``。

    属性：
        fps: 配置的输出帧率（``sample_fps``；``None`` 时为源 fps）——与 ``OcvDecoder.fps``
            语义一致，供 ``VideoWriter`` 以同速率写出。
        size: ``(width, height)``。
        duration_ms: 源视频时长（ms），``round(frame_count / source_fps * 1000)``——
            单一数据源暴露视频元数据，供 ``OcvDecoder`` 透传到 ``Tracks.duration_ms``。
    """

    def __init__(self, path: str, sample_fps: float | None = None) -> None:
        """打开 ``path`` 并按 ``sample_fps`` 计算采样步长。

        Args:
            path: 视频文件路径。
            sample_fps: 目标采样帧率；``None`` 读全帧（源 fps）。不得超源 fps（超则
                stride 退化为 1，等价全帧）。

        Raises:
            VideoIoError: ``sample_fps`` 非正 / 打不开 / 源 fps 非正 / 帧数非正 / 尺寸非法。
        """
        if sample_fps is not None and sample_fps <= 0.0:
            raise VideoIoError(f"sample_fps 非正: {sample_fps}")

        cap = cv2.VideoCapture(path)
        # cv2 无 stub，isOpened 返回 bool；打不开立即失败，不静默回退（No Silent Degradation）。
        if not cap.isOpened():
            cap.release()
            raise VideoIoError(f"无法打开视频: {path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0.0:
            cap.release()
            raise VideoIoError(f"视频 fps 非正（损坏？）: {path}, fps={source_fps}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise VideoIoError(
                f"视频帧数非正（损坏？）: {path}, frames={frame_count}"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise VideoIoError(f"视频尺寸非法: {path}, {width}x{height}")

        effective_fps = source_fps if sample_fps is None else float(sample_fps)
        # round(source_fps / sample_fps)：每 N 源帧取一帧；下界 1（采样不得超源 fps）。
        stride = 1 if sample_fps is None else max(1, round(source_fps / sample_fps))

        self._path = path
        self._cap = cap
        self._source_fps = source_fps
        self._frame_count = frame_count
        self._stride = stride
        self._sample_fps = effective_fps
        self._size = (width, height)
        self._consumed = False

    @property
    def fps(self) -> float:
        """配置的输出帧率（``sample_fps`` 或源 fps）。"""
        return self._sample_fps

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)``。"""
        return self._size

    @property
    def duration_ms(self) -> int:
        """源视频时长（ms）：``round(frame_count / source_fps * 1000)``。"""
        return round(self._frame_count / self._source_fps * 1000)

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # 迭代器正常消费完已在 finally release；此处覆盖未迭代 / 提前 ``with`` 退出的情形。
        # cv2 重复 release 是安全 no-op。
        self._cap.release()

    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:
        """按 ``stride`` 抽帧，yield ``(frame_idx, ts_ms, BGR ndarray)``。

        ``frame_idx`` 为采样计数器（0 起连续递增）；``ts_ms`` 为源真实时间戳。
        以 ``grab()`` 返回 False 作 EOF（比 ``CAP_PROP_FRAME_COUNT`` 更可靠——后者对
        部分编码不准）。抽帧得 0 帧 → raise（No Silent Degradation）。迭代器只能消费一次。

        取舍（No Silent Degradation 边界）：``grab()`` 成功后若 ``retrieve()`` 失败（流
        中段损坏帧）会 ``break`` 静默结束、仅返回已抽到的部分帧——这与干净 EOF 无法可靠
        区分（mp4v 尾包/部分编码下 ``retrieve()`` 亦返回 False），故不 raise；全损
        （``frame_idx==0``）仍 raise。调用方若需严格，应在调用侧比对预期帧数。
        """
        if self._consumed:
            raise VideoIoError(
                f"VideoReader 迭代器已消费，不可重复迭代: {self._path}"
            )
        self._consumed = True

        cap = self._cap
        source_fps = self._source_fps
        stride = self._stride
        try:
            frame_idx = 0
            src_idx = 0
            # grab() 推进一帧（仅解封装，廉价）；每逢采样位置 retrieve() 解码该帧。
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
            # 抽帧得 0 帧（损坏/不可解码，构造期 FRAME_COUNT 对部分编码不准、可能误报正）
            # → raise，不静默产出空流（No Silent Degradation）。
            if frame_idx == 0:
                raise VideoIoError(
                    f"视频抽帧得 0 帧（损坏或不可解码）: {self._path}"
                )
        finally:
            cap.release()


class VideoWriter:
    """mp4v 视频编码器（spec §4），context manager 自动 release。

    ``isOpened`` 失败立即抛错（No Silent Degradation）；``__exit__`` release。
    """

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        """以 mp4v fourcc 打开 ``path``。

        Args:
            path: 输出视频路径（父目录自动创建）。
            fps: 输出帧率。
            size: ``(width, height)``。

        Raises:
            VideoIoError: fps 非正 / 尺寸非法 / ``VideoWriter`` 打开失败。
        """
        if fps <= 0.0:
            raise VideoIoError(f"fps 非正: {fps}")
        w, h = size
        if w <= 0 or h <= 0:
            raise VideoIoError(f"size 非法: {size}")

        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            writer.release()
            raise VideoIoError(f"VideoWriter 打开失败: {path}")

        self._path = path
        self._writer = writer

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._writer.release()

    def write(self, frame: np.ndarray) -> None:
        """写入一帧 BGR ndarray。"""
        self._writer.write(frame)


# ---------------------------------------------------------------------------
# 单测（自包含：jxl.vdt.decoder._make_synthetic_video 合成短视频，零真实视频依赖；
# _make_synthetic_video 仅在测试函数内 lazy import，本模块生产代码不依赖 vdt）
# ---------------------------------------------------------------------------


def test_reader_full_frames() -> None:
    """sample_fps=None 读全帧：帧数 ≈ 合成帧数、fps=源 fps、size 正确、frame_idx 连续。"""
    import tempfile

    from jxl.vdt.decoder import _make_synthetic_video

    with tempfile.TemporaryDirectory() as tmp:
        video = f"{tmp}/s.mp4"
        _make_synthetic_video(video, fps=25.0, frames=25, size=(64, 48))

        with VideoReader(video) as r:
            items = list(r)
            assert r.fps == 25.0
            assert r.size == (64, 48)
            assert len(items) >= 20, f"预期 ~25 帧（容许编码损耗），实得 {len(items)}"
            assert [idx for idx, _, _ in items] == list(range(len(items)))
            ts_list = [ts for _, ts, _ in items]
            assert ts_list == sorted(ts_list), "ts_ms 应单调递增"
            assert ts_list[0] == 0, "首帧 ts_ms 应 ≈0"
            for _, _, img in items:
                assert img.ndim == 3 and img.shape[2] == 3


def test_reader_sample_fps() -> None:
    """sample_fps=5 采样 25fps/25帧 → ~5 帧、ts_ms 间隔 ≈200ms、fps=5。"""
    import tempfile

    from jxl.vdt.decoder import _make_synthetic_video

    with tempfile.TemporaryDirectory() as tmp:
        video = f"{tmp}/s.mp4"
        _make_synthetic_video(video, fps=25.0, frames=25)

        r = VideoReader(video, sample_fps=5.0)
        items = list(r)
        assert r.fps == 5.0
        # stride = round(25/5) = 5；range(0,25,5) → [0,5,10,15,20] = 5 帧（允许编码 ±1）。
        assert abs(len(items) - 5) <= 1, f"预期 ~5 采样帧，实得 {len(items)}"
        ts_list = [ts for _, ts, _ in items]
        if len(ts_list) >= 2:
            gap = ts_list[1] - ts_list[0]
            assert abs(gap - 200) <= 20, f"ts_ms 间隔应 ≈200ms，实得 {gap}ms"


def test_reader_invalid_path_raises() -> None:
    """打不开的路径 → VideoIoError（No Silent Degradation）。"""
    import pytest

    with pytest.raises(VideoIoError):
        VideoReader("/nonexistent/video.mp4")


def test_reader_invalid_sample_fps_raises() -> None:
    """sample_fps<=0 → VideoIoError（构造期即拒，不进入 cv2）。"""
    import tempfile

    import pytest

    from jxl.vdt.decoder import _make_synthetic_video

    with tempfile.TemporaryDirectory() as tmp:
        video = f"{tmp}/s.mp4"
        _make_synthetic_video(video, fps=10.0, frames=5)
        with pytest.raises(VideoIoError):
            VideoReader(video, sample_fps=0.0)


def test_reader_double_iteration_raises() -> None:
    """迭代器只能消费一次（视频流语义）。"""
    import tempfile

    import pytest

    from jxl.vdt.decoder import _make_synthetic_video

    with tempfile.TemporaryDirectory() as tmp:
        video = f"{tmp}/s.mp4"
        _make_synthetic_video(video, fps=10.0, frames=10)
        r = VideoReader(video)
        first = list(r)
        assert len(first) > 0
        with pytest.raises(VideoIoError):
            list(r)


def test_writer_roundtrip_readable() -> None:
    """VideoWriter 写 N 帧 → cv2.VideoCapture 读回 isOpened + 帧数一致。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(f"{tmp}/w.mp4")
        n = 10
        with VideoWriter(out, fps=10.0, size=(64, 48)) as w:
            for _ in range(n):
                w.write(np.zeros((48, 64, 3), dtype=np.uint8))
        assert out.is_file() and out.stat().st_size > 0

        cap = cv2.VideoCapture(str(out))
        try:
            assert cap.isOpened()
            got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert abs(got - n) <= 1, f"预期 ~{n} 帧（容许编码损耗），实得 {got}"


def test_writer_creates_parent_dir() -> None:
    """VideoWriter 自动创建父目录。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub" / "deep" / "w.mp4"
        with VideoWriter(out, fps=10.0, size=(32, 32)) as w:
            w.write(np.zeros((32, 32, 3), dtype=np.uint8))
        assert out.is_file() and out.stat().st_size > 0


def test_writer_invalid_fps_raises() -> None:
    """fps<=0 → VideoIoError。"""
    import pytest

    with pytest.raises(VideoIoError):
        VideoWriter(Path("/tmp/x.mp4"), fps=0.0, size=(64, 48))


def test_writer_invalid_size_raises() -> None:
    """size 含非正维度 → VideoIoError。"""
    import pytest

    with pytest.raises(VideoIoError):
        VideoWriter(Path("/tmp/x.mp4"), fps=10.0, size=(0, 48))


def test_reader_writer_pipeline() -> None:
    """端到端：VideoReader 读合成视频 → VideoWriter 转写 → 读回帧数一致。"""
    import tempfile

    from jxl.vdt.decoder import _make_synthetic_video

    with tempfile.TemporaryDirectory() as tmp:
        src = f"{tmp}/src.mp4"
        dst = Path(f"{tmp}/dst.mp4")
        _make_synthetic_video(src, fps=10.0, frames=10, size=(64, 48))

        with VideoReader(src) as r, VideoWriter(dst, r.fps, r.size) as w:
            count = 0
            for _, _, frame in r:
                w.write(frame)
                count += 1
            assert count >= 8, f"预期 ~10 采样帧，实得 {count}"

        cap = cv2.VideoCapture(str(dst))
        try:
            assert cap.isOpened()
            got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert got >= 8, f"预期 ~10 帧，实得 {got}"
