from pathlib import Path

import typer
from jvi.image.image_nda import ImageNda
from loguru import logger

from jxl.det.d2d import D2dResult, D2dOpt, D2dObject
from jxl.det.yolo.d2d_yolo import D2dYolo
from jxl.label.a2d.dd import A2dImageLabel
from jxl.label.darknet.darknet_set import DarknetSet, darknet_dump_labels


def iou_match(res: D2dResult, label: A2dImageLabel, iou_threshold: float) -> bool:
    """判断检测结果与标注之间的IOU匹配

    Args:
        res (D2dResult): 检测结果
        label (A2dImageLabel): Darknet标注
        iou_threshold (float): IOU阈值

    Returns:
        bool: 是否匹配
    """
    if len(res.objects) != len(label.objects):
        return False
    for ob1 in res.objects:
        matched = find_matched(label, iou_threshold, ob1)
        if not matched:
            return False
    return True


def find_matched(label: A2dImageLabel, iou_threshold: float, ob1: D2dObject) -> bool:
    """判断检测结果与标注之间的IOU匹配

    Args:
        label: 标注数据
        iou_threshold: IOU阈值
        ob1: 检测对象

    Returns:
        bool: 是否找到匹配的目标
    """
    matched = False
    for ob2 in label.objects:
        if ob1.rect.iou(ob2.rect()) >= iou_threshold:
            matched = True
            break
    return matched


def main(
        darknet_dir: Path = typer.Argument(..., help="Darknet样本集所在文件夹"),
        model_path: Path = typer.Argument(..., help="模型路径"),
        export_root: Path = typer.Argument(..., help="导出路径"),
        iou_threshold: float = typer.Option(0.5, help="IOU阈值，用于判断两个目标是否匹配"),
) -> None:
    if not model_path.exists():
        print(f"模型路径不存在: {model_path}")
        return

    det_opt = D2dOpt()
    detector = D2dYolo(model_path, det_opt)

    darknet_set = DarknetSet(darknet_dir)
    pairs = darknet_set.find_pairs()

    unmatched = []

    for i, (image_path, label) in enumerate(pairs):
        logger.debug(f"#{i}: {image_path.name} objects={len(label.objects)}")
        image = ImageNda.load(image_path)
        det = detector.detect(image)
        if not iou_match(det, label, iou_threshold):
            logger.info(f"#{i}: {image_path.name} unmatched: {len(label.objects)}/{len(det.objects)}")
            unmatched.append((image_path, label))

    darknet_dump_labels(unmatched, export_root)
    logger.info("total unmatched:", len(unmatched))


if __name__ == "__main__":
    typer.run(main)
