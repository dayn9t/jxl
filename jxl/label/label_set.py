from abc import abstractmethod, ABC
from enum import IntEnum
from pathlib import Path
from typing import Optional

from jml.label.info import ImageLabelInfos, ImageLabelPairs

HOP = 'hop'
DARKNET = 'darknet'
IMAGENET = 'imagenet'
KITTI = 'kitti'


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
    def parse(cls, name: str) -> Optional['LabelFormat']:
        """解析字符串成枚举, 解析失败则为Null"""
        return cls._member_map_.get(name.upper())


class LabelSet(ABC):
    """标注集合"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return False

    def __init__(self, format_id: LabelFormat, folder: Path, meta_id: int, pattern: str):
        self.format_id = format_id
        self.folder = folder
        self.meta_id = meta_id
        self.pattern = pattern

    def __str__(self) -> str:
        return f'LabelFormat(format={self.format_id},meta_id={self.meta_id})'

    @abstractmethod
    def __len__(self) -> int:
        """获取集合中样本总数"""
        pass

    @abstractmethod
    def load_pairs(self) -> ImageLabelPairs:
        """加载本格式的数据集"""
        pass

    @abstractmethod
    def save(self, root: Path) -> None:
        """报错本格式的数据集"""
        pass
