#!/usr/bin/env python3
"""图像目录自动标注工具 - 基于 YOLOE 开放词汇检测.

该工具参考 sam_label.py 设计，但支持图像目录输入而非视频文件。
使用 YOLOE 模型的开放词汇能力，可通过文本提示指定任意目标类别。

典型用法:
    # 标注人头部
    img_label /path/to/images /path/to/output head --conf-thr 0.3

    # 标注多个类别
    img_label /path/to/images /path/to/output "head,person" --conf-thr 0.3
"""

from pathlib import Path
from typing import Annotated

import typer
from jcx.sys.fs import find
from jvi.image.image_nda import ImageNda
from loguru import logger

from jxl.det.d2d import D2dOpt
from jxl.det.yolo.d2d_yoloe import D2dYoloE
from jxl.label.a2d.dd import A2dImageLabel
from jxl.label.meta_dataset import MetaDataset
from jxl.yolo.util import yolo_set_weights_dir

app = typer.Typer(help="图像目录自动标注工具 - 基于 YOLOE 开放词汇检测")


def process_image_dir(
    src_dir: Path,
    dst_dir: Path,
    names: list[str],
    model_name: str = "yoloe-11l-seg.pt",
    weights_dir: Path | None = None,
    conf_thr: float = 0.3,
    iou_thr: float = 0.5,
    meta_id: int = 0,
    verbose: bool = False,
) -> tuple[int, int]:
    """处理图像目录，使用 YOLOE 模型进行目标检测并保存标注.

    Args:
        src_dir: 输入图像目录（支持 .jpg, .jpeg, .png, .bmp）
        dst_dir: 输出数据集目录
        names: 目标类别名称列表
        model_name: YOLOE 模型文件名
        weights_dir: 模型权重文件目录，None 使用默认目录
        conf_thr: 目标检测的最小置信度阈值
        iou_thr: 非极大值抑制的 IOU 阈值
        meta_id: 元数据 ID
        verbose: 是否显示详细信息

    Returns:
        (成功处理的图像数, 总图像数)

    """
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    logger.info("src_dir: {}", src_dir)
    logger.info("dst_dir: {}", dst_dir)

    weights_dir = weights_dir or Path("/home/jiang/cc/py/jxl/models")
    yolo_set_weights_dir(str(weights_dir))

    # 查找所有图像文件
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files: list[Path] = []
    for ext in exts:
        image_files.extend(find(src_dir, ext))
    image_files.sort()

    if not image_files:
        logger.warning("未找到图像文件: {}", src_dir)
        return 0, 0

    logger.info("找到 {} 张图像", len(image_files))

    # 初始化检测器
    opt = D2dOpt(conf_thr=conf_thr, iou_thr=iou_thr)
    model_file = Path(weights_dir, model_name)
    logger.info("model: {}", model_file)
    logger.info("names: {}", names)
    logger.info("conf_thr: {} iou_thr: {}", conf_thr, iou_thr)

    model = D2dYoloE(model_file, opt, names)

    # 初始化数据集管理器
    dataset = MetaDataset(dst_dir, "a2d", meta_id)

    success = 0
    total = len(image_files)

    for i, image_file in enumerate(image_files, 1):
        try:
            image = ImageNda.load(image_file)
            d2d_ret = model.detect(image)

            if not d2d_ret.objects:
                logger.info("[{}/{}] {} - 未检测到目标", i, total, image_file.name)
                continue

            a2d_ret = A2dImageLabel.from_d2d(d2d_ret)

            # 使用原文件名（不含扩展名）作为样本名
            name = image_file.stem
            logger.info(
                "[{}/{}] {} - 检测到 {} 个目标",
                i,
                total,
                name,
                len(a2d_ret.objects),
            )

            dataset.add_sample(name, image, a2d_ret)
            success += 1

        except (OSError, ValueError, RuntimeError, KeyError, AttributeError) as e:
            # 批处理:单张图像失败(I/O 解码 / 模型推理 / 标注转换 / 写入)记录后
            # 继续处理其余图像,不静默吞错(logger.error 显式上报);
            # verbose 模式向上抛出以便定位。
            logger.error("[{}/{}] {} - 处理失败: {}", i, total, image_file.name, e)
            if verbose:
                raise

    logger.info(
        "完成! 成功: {}/{}, 输出目录: {}",
        success,
        total,
        dst_dir,
    )
    return success, total


@app.command()
def main(
    src_dir: Annotated[Path, typer.Argument(help="输入图像目录")],
    dst_dir: Annotated[Path, typer.Argument(help="输出数据集目录")],
    names: Annotated[str, typer.Argument(help="目标类别名称, 多个类别用逗号分隔")],
    model_name: Annotated[
        str, typer.Option(help="YOLOE 模型文件名")
    ] = "yoloe-11l-seg.pt",
    weights_dir: Annotated[Path | None, typer.Option(help="模型权重文件目录")] = None,
    conf_thr: Annotated[float, typer.Option(help="置信度阈值")] = 0.3,
    iou_thr: Annotated[float, typer.Option(help="IOU 阈值")] = 0.5,
    meta_id: Annotated[int, typer.Option(help="元数据 ID")] = 0,
    verbose: Annotated[bool, typer.Option(help="显示详细信息")] = False,
) -> None:
    """对图像目录进行批量自动标注.

    该工具遍历指定目录中的所有图像文件，使用 YOLOE 开放词汇检测模型
    识别目标，并将检测结果保存为标注数据集。

    示例:
        # 标注人头部
        img_label /path/to/images /path/to/output head

        # 标注头部和人体，提高置信度阈值
        img_label /path/to/images /path/to/output "head,person" --conf-thr 0.5

        # 使用自定义模型和权重目录
        img_label /path/to/images /path/to/output head \\
            --model-name yoloe-11l-seg.pt \\
            --weights-dir /path/to/weights
    """
    name_arr = [n.strip() for n in names.split(",")]
    process_image_dir(
        src_dir=src_dir,
        dst_dir=dst_dir,
        names=name_arr,
        model_name=model_name,
        weights_dir=weights_dir,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        meta_id=meta_id,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
