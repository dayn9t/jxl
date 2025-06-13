from pathlib import Path
from typing import *

from jcx.sys.fs import files_in
from jcx.text.txt_json import load_txt, save_txt
from jvi.geo.rectangle import Rect
from loguru import logger
from pydantic import BaseModel
from rustshed import *

from jxl.label.a2d.dd import A2dObjectLabel, A2dImageLabel

DARKNET_EXT = ".txt"  # 标注文件扩展名


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

    objects: List[DarknetObjectLabel]
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
            path (Path): 保存文件的路径。

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


DarknetImageLabelPair: TypeAlias = Tuple[Path, DarknetImageLabel]
"""Darknet图像与标注信息对"""

DarknetImageLabelPairs: TypeAlias = List[DarknetImageLabelPair]
"""Darknet图像与标注信息对集"""


class DarknetDir:
    """Darknet标注格式数据集目录"""

    @classmethod
    def valid_dir(cls, folder: Path, _meta_id: int = 0) -> bool:
        """检验路径是否是本格式的数据集"""
        return Path(folder / "images").is_dir() and Path(folder / "labels").is_dir()

    def __init__(self, folder: Path):
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
    # print(f"darknet_load_labels: {len(label_files)}")

    pairs = []
    for label_file in label_files:
        label = DarknetImageLabel.load(label_file).unwrap()
        image_file = (image_dir / label_file.name).with_suffix(".jpg")
        assert image_file.exists(), f"图像文件不存在: {image_file}"

        pairs.append((image_file, label))
    return pairs


def darknet_label_path_of(image_file: Path) -> Path:
    """从图片文件获取对应的标注文件路径"""
    label_file = (image_file.parent.parent / "labels" / image_file.name).with_suffix(
        DARKNET_EXT
    )
    return label_file


def darknet_save_pairs(pairs: DarknetImageLabelPairs) -> None:
    """保存 darknet 格式的数据集"""
    if not pairs:
        logger.warning("No pairs to save.")
        return

    for image_file, label in pairs:
        p = darknet_label_path_of(image_file)
        label.save(p)
