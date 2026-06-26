#!/home/jiang/py/jxl/.venv/bin/python
"""COCO -> YOLO 格式转换(按类别筛, 如 person).

读 COCO instances_*.json, 筛指定类别(默认 person), bbox [x,y,w,h] 绝对像素左上角
-> YOLO 归一化中心 xywh, 输出 images/+labels/. 多 category 可指定.

用法:
    coco_to_yolo /path/instances_train2017.json /path/train2017 /path/out
    coco_to_yolo ... --categories person --categories car
"""

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import orjson
import typer
from loguru import logger

# typer CLI 惯用模式
app = typer.Typer(help="COCO -> YOLO 转换(按类别)")

IMG_EXT = ".jpg"


@app.command()
def main(
    ann_json: Annotated[Path, typer.Argument(help="COCO instances_*.json")],
    img_dir: Annotated[Path, typer.Argument(help="COCO 图片目录(train2017/等)")],
    out_dir: Annotated[Path, typer.Argument(help="输出 YOLO 目录(images/+labels/)")],
    categories: Annotated[list[str], typer.Option(help="目标类别名(默认 person)")] = ["person"],
) -> None:
    """筛 COCO 类别 转 YOLO, 复制对应图片."""
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    logger.info("读 {} ...", ann_json.name)
    data = orjson.loads(ann_json.read_bytes())

    # 类别名 -> YOLO class id(按指定顺序)
    name2yid = {n: i for i, n in enumerate(categories)}
    cat2yid = {
        c["id"]: name2yid[c["name"]]
        for c in data["categories"]
        if c["name"] in name2yid
    }
    assert cat2yid, f"类别 {categories} 不在 {ann_json.name}"
    logger.info("类别映射: {}", {c["name"]: c["id"] for c in data["categories"] if c["name"] in name2yid})

    imgs = {im["id"]: im for im in data["images"]}
    by_img: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for ann in data["annotations"]:
        yid = cat2yid.get(ann["category_id"])
        if yid is not None:
            by_img[ann["image_id"]].append((yid, ann["bbox"]))

    n_img = n_box = 0
    missing = 0
    for img_id, boxes in by_img.items():
        im = imgs[img_id]
        iw, ih = float(im["width"]), float(im["height"])
        src = img_dir / im["file_name"]
        if not src.is_file():
            missing += 1
            continue
        lines = []
        for yid, bbox in boxes:
            x, y, w, h = bbox
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(iw, x + w), min(ih, y + h)
            cw, ch = x2 - x1, y2 - y1
            if cw <= 0 or ch <= 0:
                continue
            cx, cy = (x1 + cw / 2) / iw, (y1 + ch / 2) / ih
            lines.append(f"{yid} {cx:.6f} {cy:.6f} {cw / iw:.6f} {ch / ih:.6f}")
        if not lines:
            continue
        (out_lbl / f"{Path(im['file_name']).stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        shutil.copy(src, out_img / im["file_name"])
        n_img += 1
        n_box += len(lines)

    logger.info(
        "完成: {} 图 / {} 框 -> {}{}",
        n_img, n_box, out_dir, f" | 跳过 {missing} 缺图" if missing else "",
    )


if __name__ == "__main__":
    app()
