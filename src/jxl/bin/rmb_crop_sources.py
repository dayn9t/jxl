#!/home/jiang/py/jxl/.venv/bin/python
"""从 rmb_yolo train 按 bbox 抠出钱币前景，作可灵图生图的参考素材源。

按面额平衡抽样，去白底，输出紧致钱币图到 assets/sources/{denom}/。
即"根据标注的长方形截取前景"。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from PIL import Image

app = typer.Typer(add_completion=False, help="按 bbox 抠钱币前景作可灵参考源。")

CANON = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        c = img_dir / (stem + ext)
        if c.exists():
            return c
    return None


@app.command()
def run(
    root: Path = typer.Option(Path("assets/rmb_yolo"), "--root", help="rmb_yolo 根目录。"),
    split: str = typer.Option("train", "--split", help="train/valid。"),
    out: Path = typer.Option(Path("assets/sources"), "--out", help="输出目录。"),
    per_denom: int = typer.Option(40, "--per-denom", help="每面额抽多少张。"),
    padding: float = typer.Option(0.08, "--padding", help="bbox 外扩比例。"),
    seed: int = typer.Option(7, "--seed"),
    min_size: int = typer.Option(120, "--min-size", help="抠图最短边像素下限（太小的丢弃）。"),
) -> None:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    by_denom: dict[str, list[tuple[Path, list[float]]]] = {d: [] for d in CANON}
    for lbl in lbl_dir.glob("*.txt"):
        stem = lbl.stem
        img = find_image(img_dir, stem)
        if img is None:
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            if 0 <= cls < 6:
                by_denom[CANON[cls]].append((img, [float(p[1]), float(p[2]), float(p[3]), float(p[4])]))

    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    dropped = 0
    for d in CANON:
        items = by_denom[d]
        rng.shuffle(items)
        (out / d).mkdir(parents=True, exist_ok=True)
        n = 0
        for img, bbox in items:
            if n >= per_denom:
                break
            im = Image.open(img).convert("RGB")
            w_im, h_im = im.size
            cx, cy, w, h = bbox
            x1 = max(0.0, (cx - w / 2 - w * padding) * w_im)
            y1 = max(0.0, (cy - h / 2 - h * padding) * h_im)
            x2 = min(w_im, (cx + w / 2 + w * padding) * w_im)
            y2 = min(h_im, (cy + h / 2 + h * padding) * h_im)
            if x2 - x1 < min_size or y2 - y1 < min_size:
                dropped += 1
                continue
            crop = im.crop((int(x1), int(y1), int(x2), int(y2)))
            name = f"{img.stem}_{d}_{n}.jpg"
            crop.save(out / d / name, quality=92)
            manifest.append({"denom": d, "src": str(img), "bbox": bbox, "out": str(out / d / name)})
            n += 1

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.secho(f"抠出 {len(manifest)} 张钱币前景 → {out}（丢弃太小 {dropped}）", fg=typer.colors.GREEN)
    for d in CANON:
        typer.echo(f"  {d}: {sum(1 for m in manifest if m['denom'] == d)}")


if __name__ == "__main__":
    app()
