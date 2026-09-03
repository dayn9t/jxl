"""video_keyframe 单测: I 帧下采样选择 + 相对路径输出目录 + ffmpeg 失败可重试语义."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jxl.bin import video_keyframe
from jxl.bin.video_keyframe import extract_one, rel_stem_dir, select_frames


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


# ---- ffmpeg 失败语义(extract_one 返回 int|None, 注入假 ffmpeg 不依赖真二进制) ----


def _fake_ffmpeg(
    rc: int, n_frames: int = 0
) -> Callable[[Path, Path], subprocess.CompletedProcess[str]]:
    """假 ffmpeg: 在 tmp_dir 落 n_frames 张伪帧后返回指定 rc."""

    def run(mkv: Path, tmp_dir: Path) -> subprocess.CompletedProcess[str]:
        for i in range(n_frames):
            (tmp_dir / f"f{i:06d}.jpg").write_bytes(b"jpg")
        return subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=rc, stderr="boom" if rc else ""
        )

    return run


def _mk_mkv(video_dir: Path, rel: str) -> Path:
    mkv = video_dir / rel
    mkv.parent.mkdir(parents=True, exist_ok=True)
    mkv.write_bytes(b"x")
    return mkv


def test_extract_one_ffmpeg_failure_returns_none(tmp_path: Path) -> None:
    # rc!=0 → None(不写断点的信号), 与真 0 I 帧(0)显式区分, 不静默丢帧
    _mk_mkv(tmp_path, "a.mkv")
    n = extract_one(
        tmp_path / "a.mkv", tmp_path / "out", 4, tmp_path, ffmpeg=_fake_ffmpeg(rc=1)
    )
    assert n is None


def test_extract_one_zero_iframe_success_returns_zero(tmp_path: Path) -> None:
    # rc=0 且 0 I 帧 → 0(合法负语义: kept=0 状态行照写)
    _mk_mkv(tmp_path, "a.mkv")
    n = extract_one(
        tmp_path / "a.mkv", tmp_path / "out", 4, tmp_path, ffmpeg=_fake_ffmpeg(rc=0)
    )
    assert n == 0


def test_extract_one_success_keeps_every_nth(tmp_path: Path) -> None:
    # 5 I 帧 every=2 → 保留 0/2/4 共 3 帧, 输出按 i//every 重编号
    _mk_mkv(tmp_path, "a.mkv")
    n = extract_one(
        tmp_path / "a.mkv",
        tmp_path / "out",
        2,
        tmp_path,
        ffmpeg=_fake_ffmpeg(rc=0, n_frames=5),
    )
    assert n == 3
    assert sorted(p.name for p in (tmp_path / "out" / "a").glob("*.jpg")) == [
        "00000.jpg",
        "00001.jpg",
        "00002.jpg",
    ]


def test_run_failure_no_state_line_and_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run: 失败 mkv 不写 _state.jsonl(重跑可重试) + 整体 exit 1 + 汇总失败清单."""
    vdir = tmp_path / "v"
    _mk_mkv(vdir, "a.mkv")
    _mk_mkv(vdir, "b.mkv")
    monkeypatch.setattr(video_keyframe, "run_ffmpeg", _fake_ffmpeg(rc=1))
    r = CliRunner().invoke(video_keyframe.app, [str(vdir), str(tmp_path / "out")])
    assert r.exit_code == 1
    assert not (tmp_path / "out" / "_state.jsonl").exists()
    assert "a.mkv" in r.stderr and "b.mkv" in r.stderr  # 失败清单(相对路径)


def test_run_zero_iframe_success_writes_state_exit_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run: 真 0 I 帧属成功 — kept=0 状态行照写, 整体 exit 0."""
    vdir = tmp_path / "v"
    _mk_mkv(vdir, "a.mkv")
    monkeypatch.setattr(video_keyframe, "run_ffmpeg", _fake_ffmpeg(rc=0))
    r = CliRunner().invoke(video_keyframe.app, [str(vdir), str(tmp_path / "out")])
    assert r.exit_code == 0
    state = (tmp_path / "out" / "_state.jsonl").read_text(encoding="utf-8")
    assert '"kept":0' in state


def test_run_mixed_success_failure_only_success_in_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run: 混合成败 — 状态行只含成功 mkv, exit 1, 重跑时仅失败者 todo."""
    vdir = tmp_path / "v"
    _mk_mkv(vdir, "a.mkv")
    _mk_mkv(vdir, "b.mkv")

    def half_fail(mkv: Path, tmp_dir: Path) -> subprocess.CompletedProcess[str]:
        return _fake_ffmpeg(rc=1 if mkv.name == "b.mkv" else 0, n_frames=3)(
            mkv, tmp_dir
        )

    monkeypatch.setattr(video_keyframe, "run_ffmpeg", half_fail)
    out = tmp_path / "out"
    r = CliRunner().invoke(video_keyframe.app, [str(vdir), str(out)])
    assert r.exit_code == 1
    state = (out / "_state.jsonl").read_text(encoding="utf-8")
    assert "a.mkv" in state and "b.mkv" not in state
    assert sorted(p.name for p in (out / "a").glob("*.jpg")) == ["00000.jpg"]
