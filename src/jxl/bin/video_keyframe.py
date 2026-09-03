#!/usr/bin/env python3
"""视频 I 帧抽取 + N:1 下采样(n001 关键帧提取).

ffmpeg skip_frame nokey 解码仅 I 帧(快), 落盘序号 idx%every==0 保留.
断点续跑: 仅成功 mkv 记录 _state.jsonl, 重跑跳过; ffmpeg 失败不记录(重跑自动重试),
整体失败 exit 1(不与成功混淆).
输出/状态键均按 mkv 相对 video_dir 路径(mkv 名跨日期/跨相机重名, 相对路径防撞):
帧 out_dir/<相对路径去后缀>/<NNNNN>.jpg, 状态键 = 相对路径(含 .mkv).

用法: video_keyframe <video_dir> <out_dir> [--every 4] [--jobs 4]
"""

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import orjson
import typer

app = typer.Typer(add_completion=False, help="视频 I 帧 N:1 抽取.")

_STATE_NAME = "_state.jsonl"


def select_frames(count: int, every: int) -> list[int]:
    """前 count 个解码序号保留 idx % every == 0."""
    if every < 1:
        raise ValueError(f"every 须 >=1: {every}")
    return list(range(0, count, every))


def rel_stem_dir(video_dir: Path, mkv: Path) -> Path:
    """mkv → 帧输出子目录: 相对 video_dir 路径去 .mkv 后缀(保源结构, 同名不撞)."""
    return mkv.relative_to(video_dir).with_suffix("")


def _state_key(video_dir: Path, mkv: Path) -> str:
    """状态判重键: mkv 相对 video_dir 路径(n001 mkv 名跨日期/跨相机重名, 全相对路径防撞)."""
    return str(mkv.relative_to(video_dir))


def run_ffmpeg(mkv: Path, tmp_dir: Path) -> subprocess.CompletedProcess[str]:
    """ffmpeg skip_frame nokey 解码 mkv 全部 I 帧到 tmp_dir/f%06d.jpg(成败由 caller 判)."""
    # -nostats -loglevel error: 静默; scale 保持原分辨率
    return subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-loglevel",
            "error",
            "-skip_frame",
            "nokey",
            "-i",
            str(mkv),
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            str(tmp_dir / "f%06d.jpg"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def extract_one(
    mkv: Path,
    out_dir: Path,
    every: int,
    video_dir: Path,
    ffmpeg: Callable[[Path, Path], subprocess.CompletedProcess[str]] | None = None,
) -> int | None:
    """单 mkv: 抽全部 I 帧到临时目录 → 按序保留 every:1 → out_dir 保源相对结构.

    返回保留数(rc=0 且 0 I 帧时合法为 0); ffmpeg 失败返回 None — caller 不写断点,
    重跑自动重试该 mkv(None 与真 0 I 帧显式区分, 不静默丢帧).
    ffmpeg 可注入(单测假 rc, 不依赖真 ffmpeg).
    """
    if every < 1:
        raise ValueError(f"every 须 >=1: {every}")
    stem_dir = out_dir / rel_stem_dir(video_dir, mkv)
    stem_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="keyframe-") as tmp:
        tmp_dir = Path(tmp)
        r = (ffmpeg or run_ffmpeg)(mkv, tmp_dir)
        if r.returncode != 0:
            typer.secho(
                f"ffmpeg 失败 {mkv.name}: {r.stderr[:300]}",
                fg=typer.colors.RED,
                err=True,
            )
            return None
        frames = sorted(tmp_dir.glob("f*.jpg"))
        keep = select_frames(len(frames), every)
        for i in keep:
            dst = stem_dir / f"{i // every:05d}.jpg"
            shutil.copy2(frames[i], dst)
        return len(keep)


@app.command()
def run(
    video_dir: Annotated[Path, typer.Argument(help="视频目录(递归找 mkv)")],
    out_dir: Annotated[Path, typer.Argument(help="帧输出目录")],
    every: Annotated[int, typer.Option("--every", help="I 帧 N:1 采样")] = 4,
    jobs: Annotated[int, typer.Option("--jobs", help="并行 ffmpeg 数")] = 4,
) -> None:
    """I 帧 N:1 抽取, 状态断点续跑."""
    video_root = video_dir.resolve()
    mkvs = sorted(video_root.rglob("*.mkv"))
    if not mkvs:
        typer.secho(f"无 mkv: {video_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    state = out_dir / _STATE_NAME
    state.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if state.exists():
        done = {
            str(orjson.loads(line)["mkv"])
            for line in state.read_text(encoding="utf-8").splitlines()
            if line
        }
    todo = [m for m in mkvs if _state_key(video_root, m) not in done]
    typer.secho(
        f"mkv {len(mkvs)} done {len(done)} todo {len(todo)}",
        fg=typer.colors.CYAN,
    )

    failed: list[str] = []  # 相对路径失败清单(list.append 线程安全)

    def work(m: Path) -> int | None:
        n = extract_one(m, out_dir, every, video_root)
        if n is None:
            # 失败不写状态行: 断点只记成功, 下次重跑自动重试该 mkv
            failed.append(_state_key(video_root, m))
            return None
        row = {"mkv": _state_key(video_root, m), "rel": str(m.parent), "kept": n}
        with state.open("a", encoding="utf-8") as f:
            f.write(orjson.dumps(row).decode() + "\n")
        return n

    total = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for n in ex.map(work, todo):
            if n is not None:
                total += n
    if failed:
        typer.secho(
            f"ffmpeg 失败 {len(failed)}/{len(todo)} mkv(未写断点, 重跑本命令自动重试):\n"
            + "\n".join(f"  {k}" for k in sorted(failed)),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    typer.secho(f"保留 {total} 帧 → {out_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
