#!/home/jiang/py/jxl/.venv/bin/python
"""YOLOE-seg 抠钱币：精确掩码去白底，输出透明 PNG（供 Copy-Paste）。

用 jxl venv（ultralytics 8.4.75 + yoloe-11l-seg.pt）。set_classes(["banknote"]) 开放词汇分割。
比阈值裁白边干净得多（精确掩码，无黑/白边残留）。

用法:
  /home/jiang/py/jxl/.venv/bin/python tools/seg_cut.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLOE

MODEL = "/home/jiang/py/jxl/models/yoloe-11l-seg.pt"
CANON = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOE-seg 抠钱币透明 PNG。")
    ap.add_argument("--src", default="assets/sources_selected")
    ap.add_argument("--out", default="assets/notes_cut")
    ap.add_argument("--classes", default="banknote")
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    model = YOLOE(MODEL)
    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    model.set_classes(names, model.get_text_pe(names))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for d in CANON:
        sd = Path(args.src) / d
        if not sd.exists():
            continue
        files = sorted(sd.glob("*.jpg"))
        if not files:
            continue
        (out / d).mkdir(parents=True, exist_ok=True)
        res_list = model.predict([str(f) for f in files], conf=args.conf, verbose=False)
        for f, r in zip(files, res_list, strict=False):
            im = Image.open(f).convert("RGBA")
            W, H = im.size
            if r.masks is None or len(r.masks) == 0:
                continue  # 未分割到，跳过
            masks = r.masks.data  # tensor N x h x w (0/1)
            areas = masks.sum(dim=(1, 2))
            best = int(areas.argmax())
            mask_small = masks[best].cpu().numpy().astype("uint8") * 255
            mh, mw = mask_small.shape
            if (mw, mh) != (W, H):
                mask_full = np.asarray(Image.fromarray(mask_small).resize((W, H), Image.NEAREST))
            else:
                mask_full = mask_small
            mask_im = Image.fromarray(mask_full).filter(ImageFilter.GaussianBlur(1.5))  # 羽化边缘软化锯齿
            im.putalpha(mask_im)
            im.save(out / d / (f.stem + ".png"))
            manifest.append({"denom": d, "src": str(f), "out": str(out / d / (f.stem + ".png"))})

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"抠图 {len(manifest)} → {out}")


if __name__ == "__main__":
    main()
