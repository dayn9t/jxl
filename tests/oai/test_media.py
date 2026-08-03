from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from jxl.oai.media import (
    build_user_content,
    image_data_url,
    sample_timestamps,
    sample_video_frames,
)


def _write_synth_video(path: Path, frames: int = 30, fps: float = 10.0) -> None:
    """合成一个 frames/fps 秒的纯色渐变视频, 供抽帧测试使用 (无二进制依赖)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
    for i in range(frames):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, 1] = (i * 8) % 256
        writer.write(frame)
    writer.release()


def test_image_data_url_jpeg(tmp_path: Path) -> None:
    p = tmp_path / "x.jpg"
    raw = b"\xff\xd8\xff\xe0fake-jpeg"
    p.write_bytes(raw)

    url = image_data_url(p)

    assert url.startswith("data:image/jpeg;base64,")
    payload = url.split("base64,", 1)[1]
    assert base64.b64decode(payload) == raw


def test_image_data_url_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        image_data_url(tmp_path / "x.bmp")


def test_sample_timestamps_multiple() -> None:
    assert sample_timestamps(10.0, 5) == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_sample_timestamps_single() -> None:
    assert sample_timestamps(10.0, 1) == [0.0]


def test_sample_timestamps_zero_raises() -> None:
    with pytest.raises(ValueError):
        sample_timestamps(1.0, 0)


def test_sample_video_frames_count_and_format(tmp_path: Path) -> None:
    p = tmp_path / "synth.mp4"
    _write_synth_video(p, frames=30, fps=10.0)  # 3.0s

    urls = sample_video_frames(p, n=4)

    assert len(urls) == 4
    for u in urls:
        assert u.startswith("data:image/jpeg;base64,")
        base64.b64decode(u.split("base64,", 1)[1])  # 解码不抛错即可


def test_sample_video_frames_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        sample_video_frames(tmp_path / "nope.mp4", n=4)


def test_build_user_content_structure() -> None:
    blocks = build_user_content(["u1", "u2"], "问题?")

    assert len(blocks) == 3
    assert blocks[0] == {"type": "image_url", "image_url": {"url": "u1"}}
    assert blocks[1] == {"type": "image_url", "image_url": {"url": "u2"}}
    assert blocks[2] == {"type": "text", "text": "问题?"}
