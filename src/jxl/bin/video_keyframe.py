#!/usr/bin/env python3
"""视频 I 帧抽取 + N:1 下采样(n001 关键帧提取).

ffmpeg skip_frame nokey 解码仅 I 帧(快), 落盘序号 idx%every==0 保留.
断点续跑: 已处理 mkv 记录 _state.jsonl, 重跑跳过.
命名 {mkv_stem}_{i:05d}.jpg 保源可溯(mkv 名含日期时间在父目录, 记入状态行).

用法: video_keyframe <video_dir> <out_dir> [--every 4] [--jobs 4]
"""

import shutil
import subprocess
import tempfile
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


def _state_key(mkv: Path) -> str:
    """状态判重键: 父目录名/文件名(n001 mkv 名跨日期重名, 需日期防撞)."""
    return f"{mkv.parent.name}/{mkv.name}"


def extract_one(mkv: Path, out_dir: Path, every: int) -> int:
    """单 mkv: 抽全部 I 帧到临时目录 → 按序保留 every:1 → out_dir. 返回保留数."""
    if every < 1:
        raise ValueError(f"every 须 >=1: {every}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="keyframe-") as tmp:
        tmp_dir = Path(tmp)
        # -nostats -loglevel error: 静默; scale 保持原分辨率
        r = subprocess.run(
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
        if r.returncode != 0:
            typer.secho(
                f"ffmpeg 失败 {mkv.name}: {r.stderr[:300]}",
                fg=typer.colors.RED,
                err=True,
            )
            return 0
        frames = sorted(tmp_dir.glob("f*.jpg"))
        keep = select_frames(len(frames), every)
        for i in keep:
            dst = out_dir / f"{mkv.stem}_{i // every:05d}.jpg"
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
    mkvs = sorted(video_dir.rglob("*.mkv"))
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
    todo = [m for m in mkvs if _state_key(m) not in done]
    typer.secho(
        f"mkv {len(mkvs)} done {len(done)} todo {len(todo)}",
        fg=typer.colors.CYAN,
    )

    def work(m: Path) -> int:
        n = extract_one(m, out_dir, every)
        row = {"mkv": _state_key(m), "rel": str(m.parent), "kept": n}
        with state.open("a", encoding="utf-8") as f:
            f.write(orjson.dumps(row).decode() + "\n")
        return n

    total = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for n in ex.map(work, todo):
            total += n
    typer.secho(f"保留 {total} 帧 → {out_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
