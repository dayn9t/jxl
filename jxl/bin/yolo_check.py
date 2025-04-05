import typer
from pathlib import Path
from typing import List
from jxl.label.darknet.darknet_dir import DarknetImageLabel
from jxl.det.d2d import Detector2D, D2dResult
from jxl.model.types import ModelInfo
from jvi.image.image_nda import ImageNda
from jxl.label.a2d.dd import A2dImageLabel, A2dObjectLabel
from jcx.time.dt import now_iso_str


def check_darknet_labels_with_detector(
    label_path: Path, detector: Detector2D, image: ImageNda, iou_threshold: float = 0.5
) -> List[str]:
    """
    检查Darknet标注文件中的样本，并挑选出与Detector2D输出不符的样本。

    Args:
        label_path (Path): 标注文件的路径。
        detector (Detector2D): 2D目标检测器实例。
        image (ImageNda): 输入图像。
        iou_threshold (float): IOU阈值，用于判断两个目标是否匹配。

    Returns:
        List[str]: 包含标注与检测器输出不符的样本的行。
    """

    errors = []
    result = DarknetImageLabel.load(label_path)
    if result.is_err():
        print(f"无法加载文件: {label_path}")
        return errors

    label = result.unwrap()
    detection_result: D2dResult = detector.detect(image)

    for obj in label.objects:
        match_found = False
        label_rect = obj.rect()
        for det_obj in detection_result.objects:
            if obj.class_id == det_obj.cls:
                det_rect = det_obj.rect
                if iou(label_rect, det_rect) >= iou_threshold:
                    match_found = True
                    break
        if not match_found:
            errors.append(obj.to_str())

    return errors


def to_label(detection_result: D2dResult) -> A2dImageLabel:
    """
    将检测结果转换为 A2dImageLabel 格式。

    Args:
        detection_result (D2dResult): 检测器输出的结果。

    Returns:
        A2dImageLabel: 转换后的标注信息。
    """
    objects = []
    for det_obj in detection_result.objects:
        obj = A2dObjectLabel.new(
            id_=det_obj.id,
            category=det_obj.cls,
            confidence=det_obj.conf,
            polygon=det_obj.rect.vertexes(),
        )
        objects.append(obj)

    return A2dImageLabel.new(
        user_agent="yolo_check",
        objects=objects,
        date=now_iso_str(),
        last_modified=now_iso_str(),
    )


def main(
    label_file: Path = typer.Argument(..., help="Darknet标注文件的路径"),
    image_file: Path = typer.Argument(..., help="对应的图像文件路径"),
    model_info: str = typer.Argument(..., help="模型信息字符串"),
    model_root: Path = typer.Argument(..., help="模型根目录路径"),
    iou_threshold: float = typer.Option(0.5, help="IOU阈值，用于判断两个目标是否匹配"),
) -> None:
    """
    检查Darknet标注文件中的样本与检测器输出是否匹配。

    Args:
        label_file (Path): Darknet标注文件的路径
        image_file (Path): 对应的图像文件路径
        model_info (str): 模型信息字符串
        model_root (Path): 模型根目录路径
        iou_threshold (float): IOU阈值，用于判断两个目标是否匹配
    """
    detector = Detector2D.new(ModelInfo.parse(model_info), model_root)
    image = ImageNda.load(image_file)

    errors = check_darknet_labels_with_detector(
        label_file, detector, image, iou_threshold
    )
    if errors:
        print("发现以下标注与检测器输出不符的样本:")
        for error in errors:
            print(error)
    else:
        print("所有标注均与检测器输出匹配。")


if __name__ == "__main__":
    typer.run(main)
