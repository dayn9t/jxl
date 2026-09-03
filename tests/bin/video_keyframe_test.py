"""video_keyframe 纯函数单测: I 帧下采样选择 + 相对路径输出目录."""

from __future__ import annotations

from pathlib import Path

import pytest

from jxl.bin.video_keyframe import rel_stem_dir, select_frames


def test_select_frames_every_4() -> None:
    # 300 I 帧 4:1 → 保留 0,4,8,...,296 共 75
    assert select_frames(300, 4) == list(range(0, 300, 4))


def test_select_frames_fewer_than_every() -> None:
    assert select_frames(3, 4) == [0]


def test_select_frames_invalid() -> None:
    with pytest.raises(ValueError):
        select_frames(10, 0)


def test_rel_stem_dir_same_name_cross_camera(tmp_path: Path) -> None:
    # n001 实测结构 video/{1,2}/{0}/{date}/HH-MM-SS.mkv: 相机 1/2 同日期同名 mkv
    # → 输出子目录不同(防帧互相覆盖), 跨日期同理由日期段防撞
    cam1 = tmp_path / "1" / "0" / "2026-06-23" / "08-00-02.mkv"
    cam2 = tmp_path / "2" / "0" / "2026-06-23" / "08-00-02.mkv"
    assert rel_stem_dir(tmp_path, cam1) == Path("1/0/2026-06-23/08-00-02")
    assert rel_stem_dir(tmp_path, cam2) == Path("2/0/2026-06-23/08-00-02")


def test_rel_stem_dir_stem_keeps_inner_dots(tmp_path: Path) -> None:
    # HH-MM-SS.xxx.mkv → 仅去最后 .mkv 后缀, 保留 HH-MM-SS.xxx
    mkv = tmp_path / "1" / "0" / "d" / "08-00-02.000.mkv"
    assert rel_stem_dir(tmp_path, mkv) == Path("1/0/d/08-00-02.000")
