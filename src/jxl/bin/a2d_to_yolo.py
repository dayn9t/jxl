#!/usr/bin/env python3
"""A2dResult 格式数据集 -> YOLO(darknet).

处理旧版 img_label/sam_label 产出的 A2dResult(检测结果)数据集,
将其转换为 YOLO 训练格式(images/ + labels/). 与 jxl_sample(A2dImageLabel
格式)并存, 专门处理历史 A2dResult 数据, 且不依赖 meta 文件.

典型用法:
    a2d_to_yolo /path/to/dataset /path/to/yolo_out
    a2d_to_yolo /path/to/dataset /path/to/yolo_out --meta-id 0
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer
from jcx.sys.fs import files_in, make_subdir
from jcx.text.txt_json import load_json
from loguru import logger

from jxl.det.a2d import A2dObject, A2dResult
from jxl.label.a2d.dd import A2dObjectLabel, A2dObjectLabels
from jxl.label.darknet.darknet_set import darknet_export_objects

# typer CLI 惯用模式: 参数校验异常消息豁免噪声规则

app = typer.Typer(help="A2dResult 格式数据集 -> YOLO(darknet)")

IMG_EXT = ".jpg"
"""图像文件扩展名"""


def a2d_object_to_label(ob: A2dObject) -> A2dObjectLabel:
    """A2dObject(检测结果) -> A2dObjectLabel(标注格式)."""
    return A2dObjectLabel.new(
        id_=ob.id, category=ob.cls, confidence=ob.conf, polygon=ob.rect.vertexes()
    )


def a2d_result_to_objects(res: A2dResult) -> A2dObjectLabels:
    """A2dResult.objects -> A2dObjectLabel 列表."""
    return [a2d_object_to_label(ob) for ob in res.objects]


@app.command()
def main(
    src_dir: Annotated[
        Path, typer.Argument(help="来源目录(含 image/ + a2d_m{meta_id}/)")
    ],
    dst_dir: Annotated[Path, typer.Argument(help="目的 YOLO 目录")],
    meta_id: Annotated[int, typer.Option(help="元数据ID")] = 0,
) -> None:
    """把 A2dResult 数据集导出为 YOLO(images/ + labels/)."""
    if not src_dir.is_dir():
        raise typer.BadParameter(f"来源目录不存在: {src_dir}")

    image_dir = src_dir / "image"
    label_dir = src_dir / f"a2d_m{meta_id}"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {label_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    labels_out = make_subdir(dst_dir, "labels", remake=True)
    images_out = make_subdir(dst_dir, "images", remake=True)

    json_files = files_in(label_dir, ".json")
    logger.info("源: {} | 样本数: {}", src_dir, len(json_files))

    total = 0
    missing = 0
    for i, jf in enumerate(json_files, 1):
        res = load_json(jf, A2dResult).unwrap()
        objects = a2d_result_to_objects(res)
        darknet_export_objects(objects, labels_out / f"{jf.stem}.txt")
        src_image = image_dir / f"{jf.stem}{IMG_EXT}"
        if src_image.is_file():
            shutil.copy(src_image, images_out / src_image.name)
        else:
            missing += 1
        total += len(objects)
        if i % 1000 == 0:
            logger.info("进度: {}/{}", i, len(json_files))

    logger.info(
        "完成! 样本: {} | 目标数: {} | 缺图: {} | 输出: {}",
        len(json_files),
        total,
        missing,
        dst_dir,
    )


if __name__ == "__main__":
    app()
