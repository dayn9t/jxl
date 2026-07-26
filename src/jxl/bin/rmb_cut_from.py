#!/usr/bin/env python3
"""从 rmb_yolo 按 bbox 抠取钞票前景：裁剪块 → YOLOE-seg 抠透明 → 按面额存 notes_cut。

用户决策：rmb_yolo 不再作训练/验证集，但从中抠取钞票前景子图作 Copy-Paste 素材。
对每个合格 bbox 裁剪块(含 padding)，YOLOE-seg("banknote")抠透明背景，按标注 class 存面额目录。

过滤：bad_bbox 黑名单 + 面积过大/过小 + 横条(纵横比>10) + 单边>0.98。
面额不均(1/5/10yuan 各~300, 20/50/100yuan 各1000-2000)，小面额靠可灵变体补充。

用法:
  /home/jiang/cc/py/jxl/.venv/bin/python tools/cut_from_rmb.py --limit 50   # 小规模验证
  /home/jiang/cc/py/jxl/.venv/bin/python tools/cut_from_rmb.py              # 全量
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLOE

MODEL = "/home/jiang/cc/py/jxl/models/yoloe-11l-seg.pt"
CANON = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]
ID2D = dict(enumerate(CANON))


def collect_items(root: Path, split: str, bad: set[str],
                  area_min: float, area_max: float, ar_max: float) -> list[tuple]:
    """收集所有合格 bbox → (img_path, class, cx,cy,w,h, stem, idx)。"""
    imgdir = root / "images" / split
    lbldir = root / "labels" / split
    items: list[tuple] = []
    for img in sorted(imgdir.glob("*.jpg")):
        if img.stem in bad:
            continue
        lbl = lbldir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        for i, line in enumerate(lbl.read_text(encoding="utf-8").splitlines()):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            cx, cy, w, h = map(float, p[1:5])
            area = w * h
            short = min(w, h)
            ar = max(w, h) / short if short > 0 else 999.0
            if not (area_min <= area <= area_max):
                continue
            if w > 0.98 or h > 0.98:
                continue
            if ar > ar_max:
                continue
            items.append((img, c, cx, cy, w, h, img.stem, i))
    return items


def crop_block(im: Image.Image, cx: float, cy: float, w: float, h: float, pad: float) -> Image.Image:
    W, H = im.size
    x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
    x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
    x1 -= (x2 - x1) * pad
    y1 -= (y2 - y1) * pad
    x2 += (x2 - x1) * pad
    y2 += (y2 - y1) * pad
    return im.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))


def main() -> None:
    ap = argparse.ArgumentParser(description="从 rmb_yolo 抠取钞票前景透明 PNG。")
    ap.add_argument("--root", default="assets/rmb_yolo")
    ap.add_argument("--out", default="assets/notes_cut")
    ap.add_argument("--bad", default="assets/rmb_bad_bbox.json")
    ap.add_argument("--splits", default="train,valid", help="抠取的 split(逗号分隔)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--pad", type=float, default=0.10, help="bbox 外扩比例")
    ap.add_argument("--area-min", type=float, default=0.005)
    ap.add_argument("--area-max", type=float, default=0.90)
    ap.add_argument("--ar-max", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="每 split 取前 N 个 bbox(0=全部)")
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()

    model = YOLOE(MODEL)
    model.set_classes(["banknote"], model.get_text_pe(["banknote"]))

    bad_path = Path(args.bad)
    if bad_path.exists():
        with bad_path.open(encoding="utf-8") as f:
            bad: set[str] = set(json.load(f))
    else:
        bad = set()
    out = Path(args.out)
    for d in CANON:
        (out / d).mkdir(parents=True, exist_ok=True)

    total, ok = 0, 0
    per_denom: dict[str, int] = dict.fromkeys(CANON, 0)
    for raw_split in args.splits.split(","):
        split = raw_split.strip()
        items = collect_items(Path(args.root), split, bad, args.area_min, args.area_max, args.ar_max)
        if args.limit:
            items = items[:args.limit]
        print(f"[{split}] 合格 bbox: {len(items)}")
        total += len(items)

        # 裁剪块
        blocks: list[tuple[Image.Image, tuple[int, str, int]]] = []
        for img, c, cx, cy, w, h, stem, i in items:
            im = Image.open(img).convert("RGB")
            blocks.append((crop_block(im, cx, cy, w, h, args.pad), (c, stem, i)))

        # 批量 YOLOE-seg
        for b in range(0, len(blocks), args.batch):
            chunk = blocks[b:b + args.batch]
            res_list = model.predict([np.array(bl[0]) for bl in chunk], conf=args.conf, verbose=False)
            for (block, (c, stem, i)), r in zip(chunk, res_list, strict=False):
                if r.masks is None or len(r.masks) == 0:
                    continue
                masks = r.masks.data
                best = int(masks.sum(dim=(1, 2)).argmax())
                mask = masks[best].cpu().numpy().astype("uint8") * 255
                W, H = block.size
                if mask.shape != (H, W):
                    mask = np.asarray(Image.fromarray(mask).resize((W, H), Image.NEAREST))
                rgba = block.convert("RGBA")
                rgba.putalpha(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(1.5)))
                d = ID2D[c]
                rgba.save(out / d / f"rmb_{stem}_{i}.png")
                ok += 1
                per_denom[d] += 1
            print(f"  {b + len(chunk)}/{len(blocks)}  累计召回 {ok}")

    print(f"\n抠图成功 {ok}/{total} (召回率 {ok / total * 100:.1f}%) → {out}")
    print("各面额:", {d: per_denom[d] for d in CANON})


if __name__ == "__main__":
    main()
