#!/usr/bin/env python3
"""从 YOLO 标注数据集截取前景 crop(任意类别 bbox, 原图分辨率, 紧贴 bbox, 无 padding).

遍历 samples/{images,labels}, 把指定类别(--person-class)的 bbox(归一化) 区域
按原图分辨率裁出为独立 jpg, 存到输出目录. 一图多目标产多 crop, 命名 <原图stem>_p<序号>.
用于后续分类器(年龄/性别/身份)训练.

典型用法:
    crop_foreground /path/to/samples /path/to/crops
    crop_foreground /path/to/samples /path/to/crops --limit 100   # 小批量验证
"""

from pathlib import Path
from typing import Annotated

import typer
from jcx.sys.fs import files_in
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from loguru import logger

# typer CLI 惯用模式: 参数校验异常消息豁免噪声规则

app = typer.Typer(help="截取前景 crop(任意类别 bbox, 原图分辨率, 紧贴 bbox)")

IMG_EXT = ".jpg"
"""图像文件扩展名"""
LBL_EXT = ".txt"
"""YOLO 标注文件扩展名"""
YOLO_FIELDS = 5
"""YOLO 标注每行字段数(class cx cy w h)"""


@app.command()
def main(
    src_dir: Annotated[Path, typer.Argument(help="samples 目录(含 images/ + labels/)")],
    dst_dir: Annotated[Path, typer.Argument(help="输出 crops 目录")],
    person_class: Annotated[int, typer.Option(help="要截取目标的 YOLO 类别 id(默认 0)")] = 0,
    limit: Annotated[int, typer.Option(help="只处理前 N 张图(0=全部)")] = 0,
) -> None:
    """从 YOLO samples 截取每个目标 bbox 为独立 crop(原图分辨率, 不 resize)."""
    image_dir = src_dir / "images"
    label_dir = src_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise typer.BadParameter(f"samples 需含 images/ + labels/: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    images = files_in(image_dir, IMG_EXT)
    if limit > 0:
        images = images[:limit]
    logger.info("输入 {} 张图 -> {}", len(images), dst_dir)

    total = written = skipped = skipped_empty = 0
    for i, img_path in enumerate(images, 1):
        label_path = label_dir / (img_path.stem + LBL_EXT)
        if not label_path.is_file():
            skipped += 1
            continue

        image = ImageNda.load(img_path)
        idx = 0
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != YOLO_FIELDS:
                continue
            if int(float(parts[0])) != person_class:
                continue
            cx, cy, w, h = map(float, parts[1:])
            # 归一化 bbox -> clip 到 [0,1] 防越界
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)
            rw, rh = x2 - x1, y2 - y1
            if rw <= 0 or rh <= 0:
                continue
            rect = Rect(x=x1, y=y1, width=rw, height=rh)  # 归一化, roi() 自动转像素
            crop = image.roi(rect)
            ch, cw = crop.data().shape[0], crop.data().shape[1]
            if cw < 1 or ch < 1:
                # 极小 bbox 经 roi round 后尺寸为 0, 无法保存, 跳过(远景小目标)
                skipped_empty += 1
                continue
            crop.save(dst_dir / f"{img_path.stem}_p{idx}{IMG_EXT}")
            idx += 1
            written += 1

        total += idx
        if i % 2000 == 0:
            logger.info("进度 {}/{} 图, crop={}", i, len(images), written)

    logger.info(
        "完成: 图={} crop={} 跳过(无label={}, 空/极小crop={})",
        len(images),
        written,
        skipped,
        skipped_empty,
    )


if __name__ == "__main__":
    app()
