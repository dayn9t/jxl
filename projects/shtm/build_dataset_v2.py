#!/usr/bin/env python3
"""Build SHTM detector V2 training set.

Sources:
- base corpus  : ~/ws/trash/cabin/samples (8,349 jpg+txt, manual 5-class labels)
                 split by dedup stems: train 6730 / val 808 / test 811
- s1 increment : ~/ws/trash/tmp_s1_yolo (VLabel export of s1_relabel_v1,
                 4,471 frames / 11,771 boxes) -> ALL into train

Output: ~/ws/trash/cabin/dataset_v2 (images/labels via symlink; rsync -L to sgcc0)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

WS = Path.home() / "ws" / "trash"
SAMPLES_IMG = WS / "cabin" / "samples" / "images"
SAMPLES_LBL = WS / "cabin" / "samples" / "labels"
S1_IMG = WS / "tmp_s1_yolo" / "images"
S1_LBL = WS / "tmp_s1_yolo" / "labels"
DEDUP = Path("/home/jiang/cc/py/jxl/projects/shtm/dedup")
OUT = WS / "cabin" / "dataset_v2"

CLASS_NAMES = ["opening", "lid", "dump", "person", "can"]


def read_stems(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def place(split: str, img_src: Path, lbl_src: Path) -> tuple[int, Counter]:
    img_dir = OUT / split / "images"
    lbl_dir = OUT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    link(img_src, img_dir / img_src.name)
    link(lbl_src, lbl_dir / lbl_src.name)
    boxes = Counter()
    n = 0
    for line in lbl_src.read_text().splitlines():
        cid = int(line.split()[0])
        boxes[cid] += 1
        n += 1
    assert n == sum(boxes.values())
    return n, boxes


def main() -> None:
    train_stems = read_stems(DEDUP / "train_stems.txt")
    val_stems = read_stems(DEDUP / "val_stems.txt")
    test_stems = read_stems(DEDUP / "test_stems.txt")

    # --- reconcile stems vs corpus before writing anything ---
    available = {p.stem for p in SAMPLES_IMG.glob("*.jpg")}
    for name, stems in (("train", train_stems), ("val", val_stems), ("test", test_stems)):
        missing = [s for s in stems if s not in available]
        if missing:
            sys.exit(f"FATAL: {name} has {len(missing)} stems missing from samples, e.g. {missing[:3]}")
    overlap = (set(train_stems) & set(val_stems)) | (set(train_stems) & set(test_stems)) | (set(val_stems) & set(test_stems))
    if overlap:
        sys.exit(f"FATAL: split overlap: {len(overlap)} stems, e.g. {sorted(overlap)[:3]}")
    covered = set(train_stems) | set(val_stems) | set(test_stems)
    extra = available - covered
    if extra:
        sys.exit(f"FATAL: {len(extra)} corpus stems not covered by any split, e.g. {sorted(extra)[:3]}")

    s1_imgs = sorted(S1_IMG.glob("*.jpg"))
    s1_stems = {p.stem for p in s1_imgs}
    clash = s1_stems & available
    if clash:
        sys.exit(f"FATAL: s1 stems clash with samples: {len(clash)}, e.g. {sorted(clash)[:3]}")
    for p in s1_imgs:
        if not (S1_LBL / f"{p.stem}.txt").exists():
            sys.exit(f"FATAL: s1 image without label: {p.name}")

    # --- build ---
    stats: dict[str, dict] = {}
    for split, stems in (("train", train_stems), ("val", val_stems), ("test", test_stems)):
        frames = 0
        boxes: Counter = Counter()
        for s in stems:
            n, b = place(split, SAMPLES_IMG / f"{s}.jpg", SAMPLES_LBL / f"{s}.txt")
            frames += 1
            boxes += b
        stats[split] = {"frames": frames, "boxes": boxes, "src": "samples"}

    for p in s1_imgs:
        n, b = place("train", p, S1_LBL / f"{p.stem}.txt")
        stats["train"]["frames"] += 1
        stats["train"]["boxes"] += b

    # --- data.yaml ---
    yaml_text = (
        "# SHTM detector V2: samples(dedup split) + s1_relabel_v1(all in train)\n"
        f"path: {OUT}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (OUT / "data.yaml").write_text(yaml_text)

    # --- report ---
    print(f"{'split':<6}{'frames':>8}{'boxes':>8}  " + "  ".join(f"{n}:{v:>6}" for n, v in zip(CLASS_NAMES, [0] * 5)))
    total_f = total_b = 0
    grand: Counter = Counter()
    for split in ("train", "val", "test"):
        st = stats[split]
        by_cls = "  ".join(f"{CLASS_NAMES[i]:<7}{st['boxes'].get(i, 0):>6}" for i in range(5))
        print(f"{split:<6}{st['frames']:>8}{sum(st['boxes'].values()):>8}  {by_cls}")
        total_f += st["frames"]
        total_b += sum(st["boxes"].values())
        grand += st["boxes"]
    by_cls = "  ".join(f"{CLASS_NAMES[i]:<7}{grand.get(i, 0):>6}" for i in range(5))
    print(f"{'TOTAL':<6}{total_f:>8}{total_b:>8}  {by_cls}")
    print(f"\ndata.yaml written -> {OUT}/data.yaml")


if __name__ == "__main__":
    main()
