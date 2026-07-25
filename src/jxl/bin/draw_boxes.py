#!/usr/bin/env python3
"""把 grounding 检测框画回原图，供视觉 / VLM 复查定位准确性。

读 ground_notes / yolo_ground 产出的 ndjson，把检测框(label+conf)画到图上，
保存到 review 目录。可对多后端各画一份做交叉对比。
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from PIL import Image, ImageDraw

app = typer.Typer(add_completion=False, help="画 grounding 框回图，供复查。")


@app.command()
def run(
    grounding: Path = typer.Option(..., "--grounding", help="grounding ndjson。"),
    src: Path = typer.Option(Path("assets/rmb_yolo/images/valid"), "--src", help="图片根（image 字段相对此）。"),
    out: Path = typer.Option(Path("assets/grounding_review"), "--out"),
    limit: int = typer.Option(20, "--limit", help="最多画多少张。"),
) -> None:
    rows = [json.loads(l) for l in grounding.read_text(encoding="utf-8").splitlines() if l.strip()]
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in rows:
        if "error" in r:
            continue
        img_path = src / r["image"]
        if not img_path.exists():
            continue
        im = Image.open(img_path).convert("RGB")
        w_im, h_im = im.size
        dr = ImageDraw.Draw(im)
        for d in r.get("detections", []):
            x1, y1, x2, y2 = d["bbox"]
            dr.rectangle([x1 * w_im, y1 * h_im, x2 * w_im, y2 * h_im], outline=(255, 0, 0), width=3)
            dr.text((x1 * w_im, max(0, y1 * h_im - 12)), f'{d["label"]} {d["conf"]:.2f}', fill=(255, 0, 0))
        im.save(out / (Path(r["image"]).stem + "_review.jpg"), quality=88)
        n += 1
        if n >= limit:
            break
    typer.secho(f"画框复查 {n} 张 → {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
