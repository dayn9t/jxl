#!/usr/bin/env python
"""
元数据集管理工具，用于导入、添加和管理样本。
"""

from pathlib import Path
from typing import Annotated, Optional, List

import typer
from jcx.text.txt_json import load_json
from loguru import logger
from jvi.image.image_nda import ImageNda

from jxl.label.meta_dataset import MetaDataset

app = typer.Typer(help="元数据集管理工具")


@app.command("import")
def import_samples(
    folder: Annotated[Path, typer.Argument(help="数据集根目录")],
    format_name: Annotated[str, typer.Argument(help="格式名称，例如'a2d'")],
    meta_id: Annotated[int, typer.Argument(help="元数据ID")] = 0,
    image_files: Annotated[
        List[Path], typer.Option(help="要导入的图像文件列表")
    ] = None,
    label_files: Annotated[
        List[Path], typer.Option(help="要导入的标注文件列表")
    ] = None,
    image_dir: Annotated[
        Optional[Path], typer.Option(help="要批量导入的图像目录")
    ] = None,
    label_dir: Annotated[
        Optional[Path], typer.Option(help="要批量导入的标注目录")
    ] = None,
    move: Annotated[bool, typer.Option(help="是否移动文件而不是复制")] = False,
) -> None:
    """
    导入样本到数据集，可以指定单个文件列表或整个目录进行批量导入。
    """
    dataset = MetaDataset(folder, format_name, meta_id)

    # 导入单个文件
    if image_files and label_files:
        if len(image_files) != len(label_files):
            logger.error("图像文件和标注文件数量不匹配")
            raise typer.Exit(1)

        for img_file, lbl_file in zip(image_files, label_files):
            logger.info(f"导入样本: {img_file.name}")
            dataset.import_sample(img_file, lbl_file, move)

    # 批量导入目录
    elif image_dir and label_dir:
        if not image_dir.is_dir() or not label_dir.is_dir():
            logger.error("指定的图像目录或标注目录不存在")
            raise typer.Exit(1)

        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        for img_file in image_files:
            # 查找对应的标注文件
            lbl_file = label_dir / img_file.with_suffix(".json").name
            if not lbl_file.exists():
                logger.warning(f"找不到对应的标注文件: {lbl_file}")
                continue

            logger.info(f"导入样本: {img_file.name}")
            dataset.import_sample(img_file, lbl_file, move)

    else:
        logger.error("请提供图像和标注文件列表，或者指定图像和标注目录")
        raise typer.Exit(1)


@app.command("add")
def add_sample(
    folder: Annotated[Path, typer.Argument(help="数据集根目录")],
    format_name: Annotated[str, typer.Argument(help="格式名称，例如'a2d'")],
    name: Annotated[str, typer.Argument(help="样本名称")],
    image_file: Annotated[Path, typer.Argument(help="图像文件路径")],
    label_file: Annotated[Path, typer.Argument(help="标注文件路径")],
    meta_id: Annotated[int, typer.Option(help="元数据ID")] = 0,
) -> None:
    """
    添加单个样本到数据集。
    """
    dataset = MetaDataset(folder, format_name, meta_id)

    if not image_file.exists():
        logger.error(f"图像文件不存在: {image_file}")
        raise typer.Exit(1)

    if not label_file.exists():
        logger.error(f"标注文件不存在: {label_file}")
        raise typer.Exit(1)

    # 加载图像和标注
    image = ImageNda.load(image_file)
    label = load_json(label_file).unwrap()

    # 添加样本
    logger.info(f"添加样本: {name}")
    dataset.add_sample(name, image, label)
    logger.info(f"样本添加成功: {name}")


@app.command("validate")
def validate_dataset(
    folder: Annotated[Path, typer.Argument(help="数据集根目录")],
    format_name: Annotated[str, typer.Argument(help="格式名称，例如'a2d'")],
    meta_id: Annotated[int, typer.Option(help="元数据ID")] = 0,
) -> None:
    """
    验证数据集结构是否有效。
    """
    dataset = MetaDataset(folder, format_name, meta_id)

    if dataset.valid():
        logger.info(f"数据集结构有效: {folder}")
        image_dir = folder / "image"
        label_dir = folder / dataset.meta_dir_name(format_name, meta_id)

        image_count = len(list(image_dir.glob("*.jpg"))) + len(
            list(image_dir.glob("*.png"))
        )
        label_count = len(list(label_dir.glob("*.json")))

        logger.info(f"图像数量: {image_count}")
        logger.info(f"标注数量: {label_count}")

        if image_count != label_count:
            logger.warning(
                f"图像和标注数量不匹配: 图像 {image_count}, 标注 {label_count}"
            )
    else:
        logger.error(f"数据集结构无效: {folder}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
