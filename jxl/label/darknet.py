from pathlib import Path

import numpy as np
from jcx.sys.fs import files_in, make_dir
from jvi.drawing.color import Color
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from jvi.image.proc import get_roi_image
from jxl.common import ProbValue
from jxl.label.info import (
    ImageLabelInfos,
    ImageLabelPairs,
    ImageLabelInfo,
    ObjectLabelInfos,
    ObjectLabelInfo,
)
from jxl.label.label_set import LabelSet, LabelFormat
from jxl.label.meta import LabelMeta

DARKNET_EXT = ".txt"  # 标注文件扩展名


def load_objects(txt_file: Path) -> ObjectLabelInfos:
    """加载标注对象集"""
    arr = np.loadtxt(str(txt_file))
    if len(arr.shape) == 1:
        if arr.shape[0] != 5:
            print("WARN: no label in %s" % txt_file)
            return []
        arr = [arr]
    # print('shape:', arr.shape, len(arr.shape))
    return [
        ObjectLabelInfo(
            id=0,
            prob_class=ProbValue(int(row[0]), 1.0),
            polygon=Rect.from_cs_list(*row[1:5]).vertexes(),
        )
        for row in arr
    ]


def save_objects(objects: ObjectLabelInfos, txt_file: Path) -> None:
    """保存标注对象集"""
    with open(txt_file, "w") as fp:
        for o in objects:
            c = o.prob_class.value
            if c < 0:  # TODO: 类别过滤
                continue
            f = o.rect().cs_list()
            fp.write("%d %f %f %f %f\n" % (c, f[0], f[1], f[2], f[3]))


class DarknetSet(LabelSet):
    """Darknet标注格式数据集"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return Path(folder / "images").is_dir() and Path(folder / "labels").is_dir()

    def __init__(self, folder: Path, meta_id: int = 0, pattern: str = "*"):
        super().__init__(LabelFormat.DARKNET, folder, meta_id, pattern)

    def __len__(self) -> int:
        assert self
        pass

    def load_pairs(self) -> ImageLabelPairs:
        """加载 darknet 格式的数据集"""
        image_dir = Path(self.folder, "images")
        label_dir = Path(self.folder, "labels")

        label_files = files_in(label_dir, DARKNET_EXT)
        print(f"darknet_load_labels: {len(label_files)}")

        pairs = []
        for label_file in label_files:
            labels = load_objects(label_file)
            image_file = (image_dir / label_file.name).with_suffix(".jpg")
            assert image_file.exists(), f"图像文件不存在: {image_file}"
            image_label = ImageLabelInfo.new("dark_loader", labels)
            pairs.append((image_file, image_label))
        return pairs

    def save(self, root: Path) -> ImageLabelInfos:
        pass


def img2label(image_file: Path) -> Path:
    """从图片文件获取标注文件"""
    label_file = (image_file.parent.parent / "labels" / image_file.name).with_suffix(
        ".txt"
    )
    assert image_file.exists()
    assert label_file.exists()
    return label_file


def darknet_dump_labels(
    labels: ImageLabelPairs,
    folder: Path,
    meta: LabelMeta,
    crop_roi: bool = False,
    keep_dst_dir: bool = False,
    prefix: str = "",
) -> int:
    """保存darknet样本标注信息, TODO: meta 可能有更多的用处"""

    remake = not keep_dst_dir
    labels_dir = make_dir(folder, "labels", remake)
    images_dir = make_dir(folder, "images", remake)

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
        save_objects(label.objects, txt)

        roi = label.roi().unwrap()
        dst = get_roi_image(src, roi, Color.parse(meta.sample.background))
        dst.save(jpg)
        total += len(label.objects)
    return total
