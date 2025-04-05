from pathlib import Path

from jcx.sys.fs import make_subdir
from jvi.drawing.color import Color
from jvi.image.image_nda import ImageNda
from jvi.image.proc import get_roi_image

from jxl.label.a2d.dd import (
    A2dImageLabelPairs,
    A2dObjectLabels,
)
from jxl.label.a2d.label_set import A2dLabelSet, LabelFormat
from jxl.label.darknet.darknet_dir import DarknetDir
from jxl.label.meta import LabelMeta

DARKNET_EXT = ".txt"
"""Darknet标注文件扩展名"""


class DarknetSet(A2dLabelSet):
    """Darknet标注格式数据集"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return DarknetDir.valid_dir(folder)

    def __init__(self, folder: Path, meta_id: int = 0):
        super().__init__(LabelFormat.DARKNET, folder, meta_id)

        self.darknet_dir = DarknetDir(folder)

    def __len__(self) -> int:
        return len(self.darknet_dir)

    def find_pairs(self, pattern: str) -> A2dImageLabelPairs:
        """加载 darknet 格式的数据集"""

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
    meta: LabelMeta,
    crop_roi: bool = False,
    keep_dst_dir: bool = False,
    prefix: str = "",
) -> int:
    """保存darknet样本标注信息, TODO: meta 可能有更多的用处"""

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
        name = prefix + image.stem
        jpg = images_dir / f"{name}.jpg"
        txt = labels_dir / f"{name}.txt"
        darknet_export_objects(label.objects, txt)

        roi = label.roi().unwrap()
        dst = get_roi_image(src, roi, Color.parse(meta.sample.background))
        dst.save(jpg)
        total += len(label.objects)
    return total
