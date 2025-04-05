from pathlib import Path

import typer
from jvi.image.image_nda import ImageNda

from jxl.det.d2d import D2dResult, D2dOpt
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
        matched = False
        for ob2 in label.objects:
            if ob1.rect.iou(ob2.rect()) >= iou_threshold:
                matched = True
                break
        if not matched:
            return False
    return True


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

    for image_path, label in pairs:
        print(f"正在处理图像: {image_path.name}")
        image = ImageNda.load(image_path)
        det = detector.detect(image)
        if not iou_match(det, label, iou_threshold):
            unmatched.append((image_path, label))

    darknet_dump_labels(unmatched, export_root)
    print("unmatched:", len(unmatched))


if __name__ == "__main__":
    typer.run(main)
