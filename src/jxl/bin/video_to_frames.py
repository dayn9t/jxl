#!/home/jiang/py/jxl/.venv/bin/python
"""Sample frames from videos into a dataset folder.

When you film banknotes from many angles under different lighting, this turns
each clip into still frames you can later annotate. Samples one frame every
``--every-sec`` seconds and writes JPEGs into the output directory.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import imageio.v3 as iio
import typer

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(add_completion=False, help="Sample frames from videos into a dataset folder.")

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@app.command()
def run(
    src: Path = typer.Option(..., help="Video file, or a directory of videos."),
    dst: Path = typer.Option(..., help="Output directory for sampled .jpg frames."),
    every_sec: float = typer.Option(1.0, min=0.1, help="Sample one frame every N seconds."),
    max_per_video: int = typer.Option(0, min=0, help="Cap frames per video (0 = no cap)."),
    prefix: str = typer.Option("frame", help="Output filename prefix."),
) -> None:
    """Walk each video and write sampled frames as JPEG."""
    dst.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    else:
        videos = [src]
    if not videos:
        typer.secho(f"No videos found at {src}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    written = 0
    for vid in videos:
        meta = iio.immeta(vid)
        fps = float(meta.get("fps") or 25.0)
        if fps <= 0:
            fps = 25.0
        step = max(1, round(fps * every_sec))
        tag = vid.stem
        kept = 0
        for idx, frame in enumerate(iio.imiter(vid)):
            if idx % step != 0:
                continue
            out_path = dst / f"{prefix}_{tag}_{written:06d}.jpg"
            iio.imwrite(out_path, frame, extension=".jpeg", quality=90)
            written += 1
            kept += 1
            if max_per_video and kept >= max_per_video:
                break

    typer.secho(f"Wrote {written} frames from {len(videos)} video(s) to {dst}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
