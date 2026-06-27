#!/home/jiang/py/jxl/.venv/bin/python
"""YOLO-aware data augmentation for banknote detection.

Reads a YOLO detection dataset (``images/`` + ``labels/`` with paired
``.jpg``/``.txt`` files, where each label line is
``class x_center y_center width height`` in normalized coords) and writes
N bbox-safe augmented copies per source image into a destination root.

All transforms go through albumentations ``BboxParams(format="yolo")`` so the
bounding boxes are warped in lockstep with the pixels and boxes that fall
below ``min_visibility`` are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import albumentations as A
import numpy as np
import typer
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

app = typer.Typer(add_completion=False, help="YOLO-aware augmentation for banknote detection.")


@dataclass(frozen=True, slots=True)
class Box:
    """One YOLO annotation: class id + normalized cx/cy/w/h."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float


def read_labels(label_path: Path) -> list[Box]:
    """Parse a YOLO ``.txt`` label file; missing file => no boxes."""
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        boxes.append(Box(int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def write_labels(label_path: Path, boxes: Sequence[Box]) -> None:
    """Write boxes back as a YOLO ``.txt`` label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{b.cls} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes)
    label_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as an HxWx3 uint8 ndarray."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def save_image_rgb(path: Path, arr: np.ndarray) -> None:
    """Save an HxWx3 uint8 ndarray as JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path, quality=95)


def build_pipeline(seed: int) -> A.Compose:
    """Bbox-safe augmentation pipeline (mild — preserves denomination cues)."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.4),
            A.Rotate(limit=15, border_mode=0, value=0, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CLAHE(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["classes"],
            min_visibility=0.3,
        ),
        seed=seed,
    )


def find_pairs(src: Path, image_ext: str) -> list[tuple[Path, Path]]:
    """Pair every image under src with its YOLO label file.

    Supports both the standard ``images/``+``labels/`` split and side-by-side
    ``img.jpg``/``img.txt`` layouts.
    """
    img_root = src / "images" if (src / "images").is_dir() else src
    lbl_root = src / "labels" if (src / "labels").is_dir() else img_root
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(img_root.rglob(f"*{image_ext}")):
        rel = img.relative_to(img_root)
        candidate = lbl_root / rel.with_suffix(".txt")
        label = candidate if candidate.exists() else img.with_suffix(".txt")
        pairs.append((img, label))
    return pairs


@app.command()
def run(
    src: Path = typer.Option(..., help="Source YOLO dataset root (images/+labels/ or side-by-side)."),
    dst: Path = typer.Option(..., help="Destination YOLO dataset root."),
    n: int = typer.Option(3, min=1, help="Augmented copies per source image."),
    seed: int = typer.Option(42, help="Base RNG seed (run is deterministic)."),
    image_ext: str = typer.Option(".jpg", help="Image extension to scan."),
) -> None:
    """Generate N bbox-safe augmented copies of every labeled image under src."""
    dst_img = dst / "images"
    dst_lbl = dst / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(seed)
    pairs = find_pairs(src, image_ext)
    if not pairs:
        typer.secho(f"No images (*{image_ext}) found under {src}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    total = 0
    skipped = 0
    for img_path, lbl_path in pairs:
        boxes = read_labels(lbl_path)
        if not boxes:
            skipped += 1
            continue
        arr = load_image_rgb(img_path)
        bboxes = [[b.xc, b.yc, b.w, b.h] for b in boxes]
        classes = [b.cls for b in boxes]
        stem = img_path.stem
        for k in range(n):
            out = pipeline(image=arr, bboxes=bboxes, classes=classes)
            out_img = out["image"]
            out_boxes = out["bboxes"]
            out_classes = out["classes"]
            # If every box was cropped out, keep the originals rather than emit an empty label.
            if not out_boxes:
                out_boxes, out_classes = bboxes, classes
            name = f"{stem}_aug{k}"
            save_image_rgb(dst_img / f"{name}{image_ext}", out_img)
            write_labels(
                dst_lbl / f"{name}.txt",
                [Box(int(c), *map(float, b)) for c, b in zip(out_classes, out_boxes, strict=False)],
            )
            total += 1

    typer.secho(
        f"Augmented {total} images (+labels) into {dst} | skipped {skipped} unlabeled",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
