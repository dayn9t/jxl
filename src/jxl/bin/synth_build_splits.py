#!/home/jiang/py/jxl/.venv/bin/python
"""合成数据随机划分 train/val/test（全合成，同分布公平评估，解决 domain shift）。

输入合成图目录(images/+labels/)，随机按比例划分，复制到 YOLO 标准布局，生成 data_abs.yaml。
val/test 与 train 同分布 → 公平评估（不再用 rmb_yolo 白底验证集）。

用法:
  .venv/bin/python tools/build_synth_splits.py --src assets/rmb_synth_v10 --dst assets/rmb_synth_v10_split
"""
import argparse
import random
import shutil
from pathlib import Path

NAMES = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]


def main() -> None:
    ap = argparse.ArgumentParser(description="合成数据随机划分 train/val/test。")
    ap.add_argument("--src", required=True, help="合成图目录(含 images/ labels/)")
    ap.add_argument("--dst", required=True, help="输出数据集目录")
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    imgs = sorted((src / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"无合成图: {src}/images/")
    rng = random.Random(args.seed)
    rng.shuffle(imgs)
    n = len(imgs)
    r = [float(x) for x in args.ratios.split(",")]
    n_train = int(n * r[0]); n_val = int(n * r[1])
    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train + n_val],
        "test": imgs[n_train + n_val:],
    }

    dst = Path(args.dst)
    for split in splits:
        for sub in ("images", "labels"):
            (dst / sub / split).mkdir(parents=True, exist_ok=True)

    for split, files in splits.items():
        for img in files:
            lbl = src / "labels" / f"{img.stem}.txt"
            shutil.copy(img, dst / "images" / split / img.name)
            if lbl.exists():
                shutil.copy(lbl, dst / "labels" / split / lbl.name)
        print(f"  {split}: {len(files)}")

    (dst / "data_abs.yaml").write_text(
        f"path: {dst.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 6\nnames: {NAMES}\n",
        encoding="utf-8")
    print(f"→ {dst}/data_abs.yaml  (总计 {n} 图)")


if __name__ == "__main__":
    main()
