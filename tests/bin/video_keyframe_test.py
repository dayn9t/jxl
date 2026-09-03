"""video_keyframe 纯函数单测: I 帧下采样选择."""
from __future__ import annotations

import pytest

from jxl.bin.video_keyframe import select_frames


def test_select_frames_every_4() -> None:
    # 300 I 帧 4:1 → 保留 0,4,8,...,296 共 75
    assert select_frames(300, 4) == list(range(0, 300, 4))


def test_select_frames_fewer_than_every() -> None:
    assert select_frames(3, 4) == [0]


def test_select_frames_invalid() -> None:
    with pytest.raises(ValueError):
        select_frames(10, 0)
