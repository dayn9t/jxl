"""A2D 自动检测标注数据集(a2d_m{meta_id}/, img_label/sam_label 产物)."""

from pathlib import Path

from jxl.label.a2d.dd import A2dImageLabelPairs
from jxl.label.a2d.label_set import A2dLabelSet, LabelFormat
from jxl.label.io import load_image_label_pairs

A2D_FIX = "a2d"
"""A2D 标注目录前缀(MetaDataset format_name)"""


class A2dSampleSet(A2dLabelSet):
    """A2D 自动检测标注格式.

    目录结构::

        {folder}/image/{stem}.jpg
        {folder}/a2d_m{meta_id}/{stem}.json   (A2dImageLabel)

    由 img_label.py / sam_label.py 自动标注产出, 经 jxl_sample --format a2d 导出 YOLO.
    """

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集."""
        return Path(folder, f"{A2D_FIX}_m{meta_id}").is_dir()

    def __len__(self) -> int:
        """样本数量."""
        return len(self.find_pairs())

    def format(self) -> LabelFormat:
        """获取标注格式."""
        return LabelFormat.A2D

    def find_pairs(self, _pattern: str = "") -> A2dImageLabelPairs:
        """加载本格式的数据集."""
        return load_image_label_pairs(self._folder, self._meta_id, A2D_FIX)

    def save(self, _root: Path) -> None:
        """只读数据集, 保存为空操作(自动标注产物不经此处回写)."""
