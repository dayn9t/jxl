#!/usr/bin/env python3
"""Convert foreign detection datasets (COCO / VOC) into YOLO .txt labels.

Outputs one ``.txt`` per image with lines ``class cx cy w h`` (all normalized),
plus a ``classes.txt`` mapping class id -> name. Roboflow YOLOv8 exports and
pankaj's darknet .txt are already YOLO — this tool is for COCO/VOC sources.

Stdlib only (json + xml.etree); no extra dependencies.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Convert COCO/VOC annotations to YOLO txt.")


def _yolo_line(x: float, y: float, w: float, h: float, iw: float, ih: float) -> str:
    """Absolute xywh (top-left) + image size -> normalized YOLO line body."""
    cx = (x + w / 2.0) / iw
    cy = (y + h / 2.0) / ih
    return f"{cx:.6f} {cy:.6f} {w / iw:.6f} {h / ih:.6f}"


@app.command(name="coco")
def coco(
    coco_json: Path = typer.Option(..., help="instances/annotations JSON (COCO format)."),
    label_dir: Path = typer.Option(..., help="Output dir for .txt label files."),
) -> None:
    """Convert a COCO detection JSON into per-image YOLO .txt labels."""
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    cats_sorted = sorted(data["categories"], key=lambda c: c["id"])
    cat_to_id = {c["id"]: i for i, c in enumerate(cats_sorted)}
    names = [c["name"] for c in cats_sorted]

    by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    label_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img in data["images"]:
        anns = by_image.get(img["id"], [])
        iw, ih = float(img["width"]), float(img["height"])
        lines = []
        for a in anns:
            x, y, w, h = (float(v) for v in a["bbox"])
            lines.append(f"{cat_to_id[a['category_id']]} {_yolo_line(x, y, w, h, iw, ih)}")
        stem = Path(str(img["file_name"])).stem
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

    classes_file = label_dir.parent / "classes.txt"
    classes_file.write_text("\n".join(names) + "\n", encoding="utf-8")
    typer.secho(f"Wrote {written} YOLO labels to {label_dir} | {len(names)} classes -> {classes_file}", fg=typer.colors.GREEN)


@app.command(name="voc")
def voc(
    ann_dir: Path = typer.Option(..., help="Dir of VOC .xml files."),
    label_dir: Path = typer.Option(..., help="Output dir for .txt label files."),
    names: Path = typer.Option(Path("classes.txt"), help="class names (one per line); created if missing."),
) -> None:
    """Convert VOC .xml annotations into YOLO .txt labels."""
    class_list: list[str]
    if names.exists():
        class_list = [ln.strip() for ln in names.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        # Auto-collect class names from all xml files (sorted) and persist.
        found: set[str] = set()
        for xf in ann_dir.glob("*.xml"):
            root = ET.parse(xf).getroot()
            for obj in root.findall("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text:
                    found.add(name_el.text.strip())
        class_list = sorted(found)
        names.write_text("\n".join(class_list) + "\n", encoding="utf-8")
    class_index = {n: i for i, n in enumerate(class_list)}

    label_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for xf in ann_dir.glob("*.xml"):
        root = ET.parse(xf).getroot()
        size = root.find("size")
        iw = float(size.find("width").text) if size is not None and size.find("width") is not None else 0.0
        ih = float(size.find("height").text) if size is not None and size.find("height") is not None else 0.0
        if iw <= 0 or ih <= 0:
            continue
        lines = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None or not name_el.text:
                continue
            cls = class_index.get(name_el.text.strip())
            if cls is None:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            x = float(bnd.find("xmin").text)
            y = float(bnd.find("ymin").text)
            w = float(bnd.find("xmax").text) - x
            h = float(bnd.find("ymax").text) - y
            lines.append(f"{cls} {_yolo_line(x, y, w, h, iw, ih)}")
        (label_dir / f"{xf.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

    typer.secho(f"Wrote {written} YOLO labels to {label_dir} | {len(class_list)} classes", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
