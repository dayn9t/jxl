from pathlib import Path
from typing import List, Self

from pydantic import BaseModel
from loguru import logger
from jvi.geo.rectangle import Rect


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

    def __repr__(self):
        """返回对象的字符串表示形式"""
        return f"DarknetObjectLabel(class_id={self.class_id}, x_center={self.x_center}, y_center={self.y_center}, width={self.width}, height={self.height})"

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


class DarknetLabel(BaseModel):
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
            DarknetLabel: 解析后的对象实例。
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
    def load(cls, path: Path) -> Self:
        """
        从指定路径加载Darknet标注文件。

        Args:
            path (StrPath): 标注文件的路径。

        Returns:
            DarknetLabel: 包含标注对象的实例。
        """

        load_txt()

        objects = []
        with open(path, "r") as file:
            for line in file:
                object = DarknetObjectLabel.from_str(line)
                objects.append(object)
        return DarknetLabel(objects=objects)
