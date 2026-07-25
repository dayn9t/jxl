#!/usr/bin/env python3
"""画 YOLO 标注 bbox 回图网格，供样本审批/VLM 复查。

读 images/+labels/，随机抽 N 张，按 class 着色画 bbox + 面额标签，拼成网格输出。
用法:
  .venv/bin/python tools/preview_yolo.py --src assets/rmb_synth_v10 --out /tmp/preview.jpg --n 12
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

NAMES = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]
COLORS = ["red", "blue", "green", "orange", "purple", "brown"]


def main() -> None:
    ap = argparse.ArgumentParser(description="画 YOLO bbox 网格预览。")
    ap.add_argument("--src", required=True, help="数据集根(含 images/ labels/)")
    ap.add_argument("--out", required=True, help="输出网格 jpg")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--split", default="", help="子目录(如 train/val/test)，空=直接 images/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    imgdir = src / "images" / args.split if args.split else src / "images"
    lbldir = src / "labels" / args.split if args.split else src / "labels"
    imgs = sorted(imgdir.glob("*.jpg"))
    rng = random.Random(args.seed)
    rng.shuffle(imgs)
    imgs = imgs[:args.n]

    cw, ch = 360, 290
    rows = (len(imgs) + args.cols - 1) // args.cols
    canvas = Image.new("RGB", (cw * args.cols, ch * rows), "white")
    d = ImageDraw.Draw(canvas)
    for i, imgp in enumerate(imgs):
        im = Image.open(imgp).convert("RGB").resize((cw - 10, ch - 30))
        ld = ImageDraw.Draw(im)
        lbl = lbldir / f"{imgp.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                x, y, w, h = map(float, p[1:5])
                W, H = im.size
                ld.rectangle([W * (x - w / 2), H * (y - h / 2), W * (x + w / 2), H * (y + h / 2)],
                             outline=COLORS[c % 6], width=3)
                ld.text((W * (x - w / 2), H * (y - h / 2) - 12), NAMES[c], fill=COLORS[c % 6])
        x0, y0 = (i % args.cols) * cw + 5, (i // args.cols) * ch + 20
        canvas.paste(im, (x0, y0))
        d.text(((i % args.cols) * cw + 5, (i // args.cols) * ch + 2), imgp.stem[:28], fill="black")
    canvas.save(args.out)
    print(f"{len(imgs)} 张 → {args.out}")


if __name__ == "__main__":
    main()
