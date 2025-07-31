from abc import abstractmethod, ABC
from enum import IntEnum
from pathlib import Path
from typing import Optional, Self, Tuple, TypeAlias, List

from jcx.sys.fs import files_in
from jcx.text.txt_json import load_json

from jxl.det.a2d import A2dResult
from jxl.label.a2d.dd import A2dImageLabelPairs
from jxl.label.a2d.label_set import LabelFormat

A2D_EXT = ".json"

A2dResultLabelPair: TypeAlias = Tuple[Path, A2dResult]
"""Darknet图像与标注信息对"""

A2dResultLabelPairs: TypeAlias = List[A2dResultLabelPair]
"""Darknet图像与标注信息对集"""


def a2d_load_pairs(folder: Path) -> A2dResultLabelPairs:
    """加载 darknet 格式的数据集"""
    image_dir = Path(folder, "image")
    label_dir = Path(folder, "a2d")

    label_files = files_in(label_dir, A2D_EXT)
    # print(f"darknet_load_labels: {len(label_files)}")

    pairs = []
    for label_file in label_files:
        label = load_json(label_file, A2dResult).unwrap()
        image_file = (image_dir / label_file.name).with_suffix(".jpg")
        assert image_file.exists(), f"图像文件不存在: {image_file}"

        pairs.append((image_file, label))
    return pairs


class A2dLabelSet(ABC):
    """2D分析标注集合"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return Path(folder / "image").is_dir() and Path(folder / "a2d").is_dir()

    def __init__(self, folder: Path, meta_id: int):
        self._folder = folder
        self._meta_id = meta_id
        self.pairs = a2d_load_pairs(folder)

    def __str__(self) -> str:
        return f"LabelFormat(format={self.format()},meta_id={self._meta_id})"

    def __len__(self) -> int:
        """获取集合中样本总数"""
        return len(self.pairs)

    def format(self) -> LabelFormat:
        """获取标注格式"""
        return LabelFormat.A2D

    def find_pairs(self, pattern: str) -> A2dImageLabelPairs:
        """查找满足条件的标签/图像对"""
        pairs = []
        for image_file, label in self.pairs:
            if pattern in image_file.name:
                pairs.append((image_file, label))
        return pairs

    def save(self, root: Path) -> None:
        """保存本格式的数据集"""
        pass
