from pathlib import Path
from typing import List, Tuple

from jcx.sys.fs import make_subdir
from jvi.drawing.color import YOLO_GRAY
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from jvi.image.proc import get_roi_image

from jxl.label.a2d.dd import (
    A2dImageLabelPairs,
    A2dObjectLabels,
)
from jxl.label.a2d.label_set import A2dLabelSet, LabelFormat
from jxl.label.darknet.darknet_dir import DarknetDir

DARKNET_EXT = ".txt"
"""Darknet标注文件扩展名"""


class DarknetSet(A2dLabelSet):
    """Darknet标注格式数据集"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return DarknetDir.valid_dir(folder)

    def __init__(self, folder: Path, meta_id: int = 0):
        super().__init__(folder, meta_id)

        self.darknet_dir = DarknetDir(folder)

    def __len__(self) -> int:
        return len(self.darknet_dir)

    def format(self) -> LabelFormat:
        """获取标注格式"""
        return LabelFormat.DARKNET

    def find_pairs(self, pattern: str = "") -> A2dImageLabelPairs:
        """查找匹配模式的标注对集"""

        pairs = self.darknet_dir.find_pairs(pattern)
        return [(path, label.to_label()) for path, label in pairs]

    def save(self, root: Path) -> None:
        self.darknet_dir.save()


def darknet_export_objects(objects: A2dObjectLabels, txt_file: Path) -> None:
    """保存标注对象集"""
    with open(txt_file, "w") as fp:
        for o in objects:
            c = o.prob_class.value
            if c < 0:  # TODO: 类别过滤
                continue
            f = o.rect().cs_list()
            fp.write("%d %f %f %f %f\n" % (c, f[0], f[1], f[2], f[3]))


def darknet_dump_labels(
    labels: A2dImageLabelPairs,
    folder: Path,
    crop_roi: bool = False,
    keep_dst_dir: bool = False,
) -> int:
    """保存darknet样本标注信息"""

    remake = not keep_dst_dir
    labels_dir = make_subdir(folder, "labels", remake)
    images_dir = make_subdir(folder, "images", remake)

    print("\n开始生成样本(%d)：" % len(labels))
    total = 0
    for image, label in labels:
        src: ImageNda = ImageNda.load(image)

        if crop_roi:
            rect, label = label.crop_by_roi(src.size())
            src = src.roi(rect)
        name = image.stem
        jpg = images_dir / f"{name}.jpg"
        txt = labels_dir / f"{name}.txt"
        darknet_export_objects(label.objects, txt)

        if label.roi == Rect.one().vertexes():
            dst = src
        else:
            dst = get_roi_image(src, label.roi, YOLO_GRAY)
        dst.save(jpg)
        total += len(label.objects)
    return total


def calc_iou(det_rect: Rect, label_rect: Rect) -> float:
    """计算检测框和标注框的IOU匹配度。

    Args:
        det_rect (Rect): 检测器输出的边界框
        label_rect (Rect): 标注的边界框

    Returns:
        float: IOU值，范围[0, 1]，值越大表示匹配度越高
    """
    # 计算交集区域
    x1 = max(det_rect.x, label_rect.x)
    y1 = max(det_rect.y, label_rect.y)
    x2 = min(det_rect.right(), label_rect.right())
    y2 = min(det_rect.bottom(), label_rect.bottom())

    # 如果没有交集，返回0
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # 计算交集面积
    intersection = (x2 - x1) * (y2 - y1)

    # 计算并集面积
    det_area = det_rect.width * det_rect.height
    label_area = label_rect.width * label_rect.height
    union = det_area + label_area - intersection

    # 计算IOU
    return intersection / union


def find_bbox_matches(
    det_rects: List[Rect], label_rects: List[Rect], iou_threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """查找检测框和标注框之间的匹配对。

    Args:
        det_rects (List[Rect]): 检测器输出的边界框列表
        label_rects (List[Rect]): 标注的边界框列表
        iou_threshold (float): IOU阈值，默认0.5

    Returns:
        List[Tuple[int, int, float]]: 匹配对列表，每个元素为(检测框索引, 标注框索引, IOU值)
    """
    matches = []
    for i, det_rect in enumerate(det_rects):
        for j, label_rect in enumerate(label_rects):
            iou = calc_iou(det_rect, label_rect)
            if iou >= iou_threshold:
                matches.append((i, j, iou))
    return matches
