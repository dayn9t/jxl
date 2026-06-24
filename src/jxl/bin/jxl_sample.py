#!/opt/ias/env/bin/python
"""样本生成程序: 把标注数据集导出为 YOLO(darknet) 训练样本."""

# typer CLI 惯用模式: bool Option 默认值与参数校验异常消息, 豁免相关噪声规则
# ruff: noqa: FBT002, TRY003, EM101, EM102

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from jxl.label.darknet.darknet_set import darknet_dump_labels
from jxl.label.io import dump_label_prop, load_image_label_pairs
from jxl.label.meta import find_meta


class SampleFormat(StrEnum):
    """样本来源标注格式."""

    HOP = "hop"
    """人工标注(hop_m{meta_id})"""
    A2D = "a2d"
    """自动标注(img_label/sam_label 产物, a2d_m{meta_id})"""


app = typer.Typer(help="样本生成程序: 标注数据集 -> YOLO 训练样本")


@app.command()
def main(  # noqa: PLR0913
    src_dir: Annotated[Path, typer.Argument(help="来源标注目录")],
    dst_dir: Annotated[Path, typer.Argument(help="目的样本目录")],
    meta_id: Annotated[int, typer.Argument(help="元数据ID")],
    label_format: Annotated[
        SampleFormat, typer.Option("--format", help="标注格式: hop/a2d")
    ] = SampleFormat.HOP,
    category: Annotated[str | None, typer.Option(help="指定类别")] = None,
    prop: Annotated[str | None, typer.Option(help="属性值")] = None,
    prefix: Annotated[str, typer.Option(help="样本文件前缀")] = "",
    keep_dst_dir: Annotated[
        bool, typer.Option("--keep-dst-dir", help="保留目标目录, 不重建")
    ] = False,
    crop_roi: Annotated[
        bool, typer.Option("--crop-roi", help="裁剪ROI, 只对检测任务有效")
    ] = False,
) -> None:
    """把标注数据集导出为 YOLO(darknet) 训练样本."""
    if not src_dir.is_dir():
        raise typer.BadParameter(f"数据来源目录不存在: {src_dir}")

    meta = find_meta(meta_id, src_dir).unwrap()

    logger.info("加载目录: {}", src_dir)

    labels = load_image_label_pairs(src_dir, meta_id, label_format)

    if not labels:
        raise RuntimeError("未加载到标注样本, 检查 --format 与 meta_id")

    if prop:
        if category is None:
            raise typer.BadParameter("使用 --prop 时必须指定 --category")
        cat_id = meta.cat_meta(name=category).id
        total = dump_label_prop(labels, dst_dir, cat_id, prop, keep_dst_dir, prefix)
    else:
        total = darknet_dump_labels(labels, dst_dir, crop_roi, keep_dst_dir)
    logger.info("样本({})生成完毕!", total)


if __name__ == "__main__":
    app()
