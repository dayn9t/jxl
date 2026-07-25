#!/usr/bin/env python3
"""背景图统一处理成 960×960：
- 宽高比≈1:1 → resize 960×960
- 宽高比≈2:1 或 1:2 → 切两半(允许重叠) → 各 resize 960×960
- 其他比例 → center crop 到 1:1 → resize 960×960
"""
import argparse
from pathlib import Path

from PIL import Image


def split_or_resize(img: Image.Image, overlap: float = 0.1) -> list[Image.Image]:
    W, H = img.size
    long_over_short = max(W, H) / min(W, H)
    out: list[Image.Image] = []
    if long_over_short > 1.6:  # 长边/短边>1.6 → 切两半(长边切,允许重叠)
        if W >= H:  # 横图，竖切
            half = W // 2
            ov = int(half * overlap)
            out.append(img.crop((0, 0, half + ov, H)).resize((960, 960), Image.LANCZOS))
            out.append(img.crop((half - ov, 0, W, H)).resize((960, 960), Image.LANCZOS))
        else:  # 竖图，横切
            half = H // 2
            ov = int(half * overlap)
            out.append(img.crop((0, 0, W, half + ov)).resize((960, 960), Image.LANCZOS))
            out.append(img.crop((0, half - ov, W, H)).resize((960, 960), Image.LANCZOS))
    else:  # ~1:1 或轻微比例 → resize 960×960
        out.append(img.resize((960, 960), Image.LANCZOS))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="背景图统一成 960×960。")
    ap.add_argument("--src", required=True, help="源背景目录(递归)")
    ap.add_argument("--dst", required=True, help="输出 960×960 目录")
    args = ap.parse_args()
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in Path(args.src).rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            for im in split_or_resize(Image.open(p).convert("RGB")):
                im.save(dst / f"bg960_{n:05d}.jpg", quality=88)
                n += 1
        except OSError:
            continue
    print(f"处理 {n} 张 960×960 → {args.dst}")


if __name__ == "__main__":
    main()
