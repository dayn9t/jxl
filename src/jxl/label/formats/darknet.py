"""Darknet(YOLO) 训练格式 IO —— jxl label/darknet 层（2026-09-21 迁出、2026-11-05 回迁）。

外部训练格式 IO，服务于 A2d/标注数据集与 YOLO 训练集互转。
类型域声明：本模块处理【标注域】数据（A2dObjectLabel：多边形 + 属性表）；
darknet 的归一化中心矩形只是外部格式的存储形状，经 to_label() 转换入口即折算为
标注域类型，检测域类型（D2dObject 等）禁止进入本包。
未迁项（2026-09-21 迁出时裁决）：calc_iou/find_bbox_matches（检测框匹配辅助，
全仓无调用方，随迁移舍弃）。
"""

from pathlib import Path
from typing import Self

from jcx.sys.fs import files_in, make_subdir
from jcx.text.io import save_txt
from jcx.text.txt_json import load_txt
from jvi.drawing.color import YOLO_GRAY
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from jvi.image.proc import get_roi_image
from loguru import logger
from pydantic import BaseModel
from rustshed import Ok, Result, result_shortcut

from jxl.label.a2d import (
    A2dImageLabel,
    A2dImageLabelPairs,
    A2dObjectLabel,
    A2dObjectLabels,
)
from jxl.label.dataset import A2dLabelSet, LabelFormat

DARKNET_EXT = ".txt"
"""Darknet标注文件扩展名"""


class DarknetObjectLabel(BaseModel):
    """
    表示Darknet标注文件中的一行数据。
    """

    class_id: int
    """类别ID，表示对象的类别"""
    x_center: float
    """边界框中心的x坐标，归一化到[0, 1]范围"""
    y_center: float
    """边界框中心的y坐标，归一化到[0, 1]范围"""
    width: float
    """边界框的宽度，归一化到[0, 1]范围"""
    height: float
    """边界框的高度，归一化到[0, 1]范围"""

    @classmethod
    def from_str(cls, line: str) -> Self:
        """
        从字符串解析DarknetObjectLabel对象。

        Args:
            line (str): 包含标注信息的字符串，格式为 "class_id x_center y_center width height"

        Returns:
            DarknetObjectLabel: 解析后的对象实例。
        """
        parts = line.strip().split()
        if len(parts) != 5:
            logger.warning(f"Invalid line format: {line.strip()}")
            raise ValueError("Line must contain 5 space-separated values.")

        class_id, x_center, y_center, width, height = map(float, parts)
        return cls(
            class_id=int(class_id),
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        )

    def to_str(self) -> str:
        """
        将DarknetObjectLabel对象转换为字符串表示形式。

        Returns:
            str: 包含标注信息的字符串，格式为 "class_id x_center y_center width height"
        """
        return f"{self.class_id} {self.x_center} {self.y_center} {self.width} {self.height}"

    def rect(self) -> Rect:
        """
        将归一化的边界框转换为的矩形区域。

        Returns:
            Rect: 包含边界框的矩形对象。
        """
        return Rect(
            x=self.x_center - self.width / 2,
            y=self.y_center - self.height / 2,
            width=self.width,
            height=self.height,
        )

    def to_label(self) -> A2dObjectLabel:
        """
        将DarknetObjectLabel转换为A2dObjectLabel。

        Returns:
            A2dObjectLabel: 转换后的A2dObjectLabel对象。
        """
        rect = self.rect()
        return A2dObjectLabel.new(
            id_=0,  # ID将在后续设置
            category=self.class_id,
            confidence=1.0,  # Darknet格式不包含置信度信息，默认为1.0
            polygon=rect.vertexes(),
        )


class DarknetImageLabel(BaseModel):
    """
    表示Darknet一个标注文件中的信息。
    """

    objects: list[DarknetObjectLabel]
    """包含的对象列表"""

    @classmethod
    def from_str(cls, lines: str) -> Self:
        """
        从多行字符串解析DarknetLabel对象。

        Args:
            lines (str): 包含标注信息的多行字符串，每行格式为 "class_id x_center y_center width height"

        Returns:
            DarknetImageLabel: 解析后的对象实例。
        """
        objects = []
        for line in lines.strip().split("\n"):
            if line != "":
                ob = DarknetObjectLabel.from_str(line)
                objects.append(ob)
        return cls(objects=objects)

    def to_str(self) -> str:
        """
        将DarknetLabel对象转换为字符串表示形式。

        Returns:
            str: 包含所有标注信息的字符串，每行格式为 "class_id x_center y_center width height"
        """
        return "\n".join([obj.to_str() for obj in self.objects])

    @classmethod
    @result_shortcut
    def load(cls, path: Path) -> Result[Self, Exception]:
        """
        从指定路径加载Darknet标注文件。

        Args:
            path (StrPath): 标注文件的路径。

        Returns:
            DarknetImageLabel: 包含标注对象的实例。
        """
        text = load_txt(path).Q
        return Ok(cls.from_str(text))

    def save(self, path: Path) -> Result[bool, Exception]:
        """
        将DarknetLabel对象保存到指定路径的文件中。

        Args:
            path (StrPath): 保存文件的路径。

        Returns:
            Result[bool, Exception]: 保存成功返回True，否则返回异常。
        """

        text = self.to_str()
        return save_txt(text, path)

    def to_label(self) -> A2dImageLabel:
        """
        将DarknetLabel转换为A2dImageLabel。

        Returns:
            A2dImageLabel: 转换后的A2dImageLabel对象。
        """
        # 将每个Darknet对象转换为A2d对象
        objects = []
        for i, ob in enumerate(self.objects):
            a2d_ob = ob.to_label()
            a2d_ob.id = i + 1  # 设置连续的ID，从1开始
            objects.append(a2d_ob)

        # 创建A2dImageLabel对象
        return A2dImageLabel(user_agent="darknet", objects=objects)


type DarknetImageLabelPair = tuple[Path, DarknetImageLabel]
"""Darknet图像与标注信息对"""

type DarknetImageLabelPairs = list[DarknetImageLabelPair]
"""Darknet图像与标注信息对集"""


class DarknetDir:
    """Darknet标注格式数据集目录"""

    @classmethod
    def valid_dir(cls, folder: Path, _meta_id: int = 0) -> bool:
        """检验路径是否是本格式的数据集"""
        return Path(folder / "images").is_dir() and Path(folder / "labels").is_dir()

    def __init__(self, folder: Path) -> None:
        self.pairs = darknet_load_pairs(folder)

    def __len__(self) -> int:
        return len(self.pairs)

    def get_pairs(self) -> DarknetImageLabelPairs:
        """获取图像和标注对列表。

        Returns:
            DarknetImageLabelPairs: 图像和标注对列表。
        """
        return self.pairs

    def find_pairs(self, pattern: str) -> DarknetImageLabelPairs:
        """根据模式查找匹配的图像和标注对。

        Args:
            pattern (str): 用于匹配文件名的模式字符串。

        Returns:
            DarknetImageLabelPairs: 匹配的图像和标注对列表。
        """
        pairs = []
        for image_file, label in self.pairs:
            if pattern in image_file.name:
                pairs.append((image_file, label))
        return pairs

    def save(self) -> None:
        """保存本格式的数据集"""
        darknet_save_pairs(self.pairs)


def darknet_load_pairs(folder: Path) -> DarknetImageLabelPairs:
    """加载 darknet 格式的数据集"""
    image_dir = Path(folder, "images")
    label_dir = Path(folder, "labels")

    label_files = files_in(label_dir, DARKNET_EXT)

    pairs = []
    for label_file in label_files:
        label = DarknetImageLabel.load(label_file).unwrap()
        image_file = (image_dir / label_file.name).with_suffix(".jpg")
        assert image_file.exists(), f"图像文件不存在: {image_file}"

        pairs.append((image_file, label))
    return pairs


def darknet_label_path_of(image_file: Path) -> Path:
    """从图片文件获取对应的标注文件路径"""
    return (image_file.parent.parent / "labels" / image_file.name).with_suffix(
        DARKNET_EXT
    )


img2label = darknet_label_path_of
"""图像路径 -> darknet 标注路径(images/x.jpg -> labels/x.txt)"""


def darknet_save_pairs(pairs: DarknetImageLabelPairs) -> None:
    """保存 darknet 格式的数据集"""
    if not pairs:
        logger.warning("No pairs to save.")
        return

    for image_file, label in pairs:
        p = darknet_label_path_of(image_file)
        label.save(p)


class DarknetSet(A2dLabelSet):
    """Darknet标注格式数据集"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return DarknetDir.valid_dir(folder)

    def __init__(self, folder: Path, meta_id: int = 0) -> None:
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
            fp.write(f"{c} {f[0]:f} {f[1]:f} {f[2]:f} {f[3]:f}\n")


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

    logger.info(f"\n开始生成样本({len(labels)})：")
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
