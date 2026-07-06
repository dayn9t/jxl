#!/home/jiang/py/jxl/.venv/bin/python
"""从 mkv 视频提取编码关键帧（I-frame）→ 图片目录。

ffmpeg 无损提取所有 I 帧（select=eq(pict_type,I)），全量不抽样。
输出扁平 jpg，命名 {video_stem}_{frame_idx:06d}.jpg。

用法:
    mkv_keyframes <src_dir> <dst_dir>
"""

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, help="mkv → 编码关键帧(I-frame)提取。")

_MKV_EXT = ".mkv"


def extract_keyframes(src_dir: Path, dst_dir: Path) -> tuple[list[Path], list[Path]]:
    """递归找 mkv → ffmpeg 提取 I 帧 → 扁平 jpg。返回 (succeeded, failed)。"""
    if not shutil.which("ffmpeg"):
        typer.secho("未找到 ffmpeg，请先安装。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    mkvs = sorted(src_dir.rglob(f"*{_MKV_EXT}"))
    if not mkvs:
        typer.secho(f"未找到 mkv: {src_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    dst_dir.mkdir(parents=True, exist_ok=True)
    succeeded: list[Path] = []
    failed: list[Path] = []
    for mkv in mkvs:
        out_pattern = str(dst_dir / f"{mkv.stem}_%06d.jpg")
        cmd = [
            "ffmpeg", "-i", str(mkv),
            "-vf", r"select=eq(pict_type\,I)",
            "-vsync", "vfr",
            "-q:v", "2",
            out_pattern,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)  # noqa: S603
        except subprocess.TimeoutExpired:
            failed.append(mkv)
            typer.secho(f"ffmpeg 超时 {mkv.name}", fg=typer.colors.YELLOW, err=True)
            continue
        if result.returncode != 0:
            failed.append(mkv)
            typer.secho(
                f"ffmpeg 失败 {mkv.name}: {result.stderr[-300:]}",
                fg=typer.colors.YELLOW,
                err=True,
            )
        else:
            succeeded.append(mkv)
    return succeeded, failed


@app.command()
def main(
    src_dir: Annotated[Path, typer.Argument(help="mkv 源目录（递归）")],
    dst_dir: Annotated[Path, typer.Argument(help="输出图片目录")],
) -> None:
    """递归抽取所有 mkv 的编码关键帧到扁平 jpg 目录。全失败时非零退出。"""
    succeeded, failed = extract_keyframes(src_dir, dst_dir)
    typer.secho(
        f"成功 {len(succeeded)} / 失败 {len(failed)} → {dst_dir}",
        fg=typer.colors.GREEN if succeeded else typer.colors.RED,
    )
    if not succeeded:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
