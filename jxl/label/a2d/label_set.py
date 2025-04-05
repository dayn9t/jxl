from abc import abstractmethod, ABC
from enum import IntEnum
from pathlib import Path
from typing import Optional, Self

from jxl.label.a2d.dd import A2dImageLabelPairs

HOP = "hop"
DARKNET = "darknet"
IMAGENET = "imagenet"
KITTI = "kitti"


class LabelFormat(IntEnum):
    """标注格式"""

    HOP = 1
    """通用对象属性标注格式"""
    IMAGENET = 2
    """ImageNet分类器标注格式"""
    DARKNET = 3
    """DarkNet标注格式"""
    KITTI = 4
    """KITTI标注格式"""
    COCO = 5
    """COCO标注格式"""
    GOOGLE = 6
    """Google标注格式"""

    @classmethod
    def parse(cls, name: str) -> Optional[Self]:
        """解析字符串成枚举, 解析失败则为Null"""
        return cls._member_map_.get(name.upper())


class A2dLabelSet(ABC):
    """2D分析标注集合"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return False

    def __init__(self, label_format: LabelFormat, folder: Path, meta_id: int):
        self._format = label_format
        self._folder = folder
        self._meta_id = meta_id

    def __str__(self) -> str:
        return f"LabelFormat(format={self._format},meta_id={self._meta_id})"

    @abstractmethod
    def __len__(self) -> int:
        """获取集合中样本总数"""
        pass

    def format(self) -> LabelFormat:
        """获取标注格式"""
        return self._format

    @abstractmethod
    def find_pairs(self, pattern: str) -> A2dImageLabelPairs:
        """查找满足条件的标签/图像对"""
        pass

    @abstractmethod
    def save(self, root: Path) -> None:
        """保存本格式的数据集"""
        pass
