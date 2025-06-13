from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, TypeAlias

from jcx.sys.fs import with_parent, StrPath, files_in
from jcx.text.txt_json import load_json, save_json
from jvi.gui.record_viewer import FileRecord
from jxl.label.ias import ias_label_path_of
from jxl.label.a2d.dd import A2dImageLabelPairs, A2dImageLabel, IMG_EXT
from jxl.label.io import label_path_of
from jxl.label.a2d.label_set import A2dLabelSet, LabelFormat, HOP
from jxl.label.meta import meta_fix
from rustshed import Option

HOP_EXT = ".json"  # 标注文件扩展名
HOP_FIX = "hop"  # HOP名称前缀/后缀


def image_label_path(img_file: Path, meta_id: int, ext: str) -> Path:
    """获取图像对应的标注文件路径"""
    file = Path(img_file).with_suffix(ext)
    return with_parent(file, f"{HOP_FIX}_{meta_fix(meta_id)}")


def hop_label_path_of(img_file: StrPath, meta_id: int) -> Path:
    """获取图像对应的HOP标注文件路径"""
    return label_path_of(img_file, HOP_FIX, meta_id, HOP_EXT)


def hop_load_label(img_file: StrPath, meta_id: int) -> Option[A2dImageLabel]:
    """加载标签"""
    label_file = hop_label_path_of(img_file, meta_id)
    return load_json(label_file, A2dImageLabel).ok()


def hop_save_label(label: A2dImageLabel, img_file: StrPath, meta_id: int) -> Path:
    """保存标签"""
    label_file = hop_label_path_of(img_file, meta_id)
    save_json(label, label_file)
    return label_file


def hop_del_label(img_file: StrPath, meta_id: int) -> None:
    label_file = hop_label_path_of(img_file, meta_id)
    label_file.unlink(True)
    print("删除标注:", label_file)


def hop_load_labels(folder: StrPath, meta_id: int) -> A2dImageLabelPairs:
    hs = HopSet(Path(folder), meta_id)
    return hs.find_pairs()


class LabelFilter(IntEnum):
    ALL = 1  # 所有图片样本
    EXPORT = 2  # 可导入标注样本
    LABELED = 3  # 已标注的样本

    def has_label(self, image_file: Path, meta_id: int) -> bool:
        if self == LabelFilter.ALL:
            return True
        if self == LabelFilter.EXPORT:
            label = ias_label_path_of(image_file, meta_id)
        elif self == LabelFilter.LABELED:
            label = hop_label_path_of(image_file, meta_id)
        else:
            return False
        return label.is_file()


def get_label(image_file: Path, meta_id: int) -> A2dImageLabel:
    """获取图像文件的标注信息"""
    cur_label = hop_load_label(image_file, meta_id)
    if cur_label.is_null():
        cur_label = import_label(image_file, meta_id)
    return cur_label.unwrap_or(A2dImageLabel(user_agent="jxl_label"))


@dataclass
class LabelRecord(FileRecord):
    """标注记录"""

    label: A2dImageLabel = field(default_factory=A2dImageLabel)


LabelRecords: TypeAlias = list[LabelRecord]
"""文件记录列表"""


def load_label_records(
    folder: StrPath,
    meta_id: int,
    label_filter: LabelFilter,
    pattern: Optional[str] = None,
    conf_thr: float = 1.0,
) -> LabelRecords:
    """加载目录下的图片信息记录"""
    pattern = (pattern or "*") + IMG_EXT
    print("pattern:", pattern)
    files = sorted(Path(folder, "image").glob(pattern))
    rs = []
    for f in files:
        if label_filter.has_label(f, meta_id):
            label = get_label(f, meta_id)
            if label.min_conf() <= conf_thr:
                rs.append(LabelRecord(f, label=label))
    return rs


def import_label(img_file: StrPath, meta_id: int) -> Option[A2dImageLabel]:
    """读取从IAS导入的标签"""
    # label_file = label_path(img_file, meta_id, LBL_EXT)
    msg_file = ias_label_path_of(img_file, meta_id)
    return load_json(msg_file, A2dImageLabel).ok()


class HopSet(A2dLabelSet):
    """皓维对象与属性标注格式(Howell Object & Properties)"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return Path(folder, f"{HOP}_m{meta_id}").is_dir()

    def __init__(self, folder: Path, meta_id: int):
        super().__init__(folder, meta_id)

    def __len__(self) -> int:
        assert self
        return 0

    def format(self) -> LabelFormat:
        """获取标注格式"""
        return LabelFormat.HOP

    def find_pairs(self, pattern: str = "") -> A2dImageLabelPairs:
        """加载本格式的数据集"""
        image_dir = Path(self._folder, "image")
        label_dir = Path(self._folder, f"hop_m{self._meta_id}")
        label_files = files_in(label_dir, HOP_EXT)

        pairs = []
        for label_file in label_files:
            label = load_json(label_file, A2dImageLabel).unwrap()
            if label:
                image_file = image_dir / (label_file.stem + IMG_EXT)
                pairs.append((image_file, label))
            else:
                print("WARN: load label fail @", label_file)
        return pairs

    def save(self, _root: Path) -> None:
        assert self
