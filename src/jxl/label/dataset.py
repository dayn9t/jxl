"""数据集布局、标注集抽象与工厂 —— jxl label 格式层（2026-09-21 迁出、2026-11-05 回迁）。

合并来源（2026-09-21 迁出时按内聚归一，回迁保持归一后形态）：
`label/a2d/label_set.py`（LabelFormat/A2dLabelSet）、`label/meta_dataset.py`
（MetaDataset）、`label/io.py`（布局读写）、`label/a2d/sample_set.py`
（A2dSampleSet）、`label/factory.py`（open_label_set）。

数据集布局约定（MetaDataset 写、load_image_label_pairs 读）::

    {ds}/image/{stem}.jpg
    {ds}/{fmt}_m{id}/{stem}.json

DarknetSet 在 `jxl.label.formats.darknet`，导入 `jxl.label.formats` 时经
register_label_set() 反向注册进 _class_map（避免 dataset → formats 循环 import）。

类型域声明：本模块全部为【标注域】数据（A2dImageLabel：多边形 + 属性表），
勿与检测域类型（rect 矩形、无属性表）混淆。
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import Final, Self

from jcx.sys.fs import (
    StrPath,
    copy_file,
    files_in,
    make_parents,
    move_file,
    remake_dir,
    with_parent,
)
from jcx.text.txt_json import load_json, save_json
from jvi.image.image_nda import ImageNda
from loguru import logger
from pydantic import BaseModel
from rustshed import Err, Null, Ok, Option, Result, Some

from jxl.label.a2d import IMG_EXT, A2dImageLabel, A2dImageLabelPairs, A2dImageLabels
from jxl.label.meta_jxl import meta_fix


class LabelFormat(IntEnum):
    """标注格式"""

    HOP = 1
    """通用对象属性标注格式"""
    A2D = 2
    """ImageNet分类器标注格式"""
    DARKNET = 3
    """DarkNet标注格式"""
    IMAGENET = 4
    """ImageNet分类器标注格式"""
    KITTI = 5
    """KITTI标注格式"""
    COCO = 6
    """COCO标注格式"""
    GOOGLE = 7
    """Google标注格式"""

    @classmethod
    def parse(cls, name: str) -> Self | None:
        """解析字符串成枚举, 解析失败则为Null"""
        r = cls._member_map_.get(name.upper())
        return r if isinstance(r, cls) else None


HOP = "hop"
DARKNET = "darknet"
IMAGENET = "imagenet"
KITTI = "kitti"


class A2dLabelSet(ABC):
    """2D分析标注集合"""

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return False

    def __init__(self, folder: Path, meta_id: int) -> None:
        self._folder = folder
        self._meta_id = meta_id

    def __str__(self) -> str:
        return f"LabelFormat(format={self.format()},meta_id={self._meta_id})"

    @abstractmethod
    def __len__(self) -> int:
        """获取集合中样本总数"""

    @abstractmethod
    def format(self) -> LabelFormat:
        """获取标注格式"""

    @abstractmethod
    def find_pairs(self, pattern: str) -> A2dImageLabelPairs:
        """查找满足条件的标签/图像对"""

    @abstractmethod
    def save(self, root: Path) -> None:
        """保存本格式的数据集"""


class MetaDataset:
    """带有Meta的样本集文件夹管理器"""

    @classmethod
    def meta_dir_name(cls, format_name: str, meta_id: int) -> str:
        """获取元数据文件夹名称"""
        return f"{format_name}_m{meta_id}"

    def __init__(self, folder: Path, format_name: str, meta_id: int) -> None:
        """创建A2D样本集文件夹管理器"""
        self._folder: Path = folder
        self._meta_id: int = meta_id
        self._format_name: str = format_name
        self._image_dir = self._folder / "image"
        self._label_dir = self._folder / self.meta_dir_name(format_name, meta_id)

    def valid(self) -> bool:
        """检验路径是否是本格式的数据集"""
        return self._image_dir.is_dir() and self._label_dir.is_dir()

    def import_sample(
        self, image_file: Path, label_file: Path, move: bool = False
    ) -> None:
        """导入样本"""
        dst_image = self._image_dir / image_file.name
        dst_label = self._label_dir / image_file.with_suffix(".json").name
        make_parents(dst_image)
        make_parents(dst_label)

        if move:
            move_file(image_file, dst_image).expect(f"移动样本图片失败: {image_file}")
            move_file(label_file, dst_label).expect(f"移动样本标注失败: {label_file}")
        else:
            copy_file(image_file, dst_image).expect(f"复制样本图片失败: {image_file}")
            copy_file(label_file, dst_label).expect(f"复制样本标注失败: {label_file}")

    def add_sample(self, name: str, image: ImageNda, label: BaseModel) -> None:
        """添加样本"""
        dst_image = self._image_dir / f"{name}.jpg"
        dst_label = self._label_dir / f"{name}.json"
        make_parents(dst_image)
        make_parents(dst_label)

        image.save(dst_image)
        save_json(label, dst_label).expect(f"保存样本标注失败: {dst_label}")


LABEL_EXT: Final[str] = ".json"
"""标注文件扩展名"""


def label_tail(meta_id: int, ext: str) -> str:
    """获取标签文件尾部"""
    return f"_{meta_fix(meta_id)}{ext}"


def label_path_of(img_file: StrPath, format_name: str, meta_id: int, ext: str) -> Path:
    """获取图像对应的标注文件路径"""
    file = Path(img_file).with_suffix(ext)
    return with_parent(file, f"{format_name}_{meta_fix(meta_id)}")


def load_image_label_pairs(
    folder: StrPath, meta_id: int, format_name: str
) -> A2dImageLabelPairs:
    """加载目录下指定格式的图像-标注对.

    读取 ``MetaDataset`` 写入的结构::

        {folder}/image/{stem}.jpg
        {folder}/{format_name}_m{meta_id}/{stem}.json

    Args:
        folder: 数据集根目录
        meta_id: 元数据 ID
        format_name: 标注格式名(目录前缀), 如 ``hop`` / ``a2d``

    Returns:
        图像与标注对集

    Raises:
        FileNotFoundError: 标注目录不存在时
    """
    root = Path(folder)
    image_dir = root / "image"
    label_dir = root / f"{format_name}_{meta_fix(meta_id)}"

    if not label_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {label_dir}")

    pairs: A2dImageLabelPairs = []
    for label_file in files_in(label_dir, LABEL_EXT):
        label = load_json(label_file, A2dImageLabel).expect(f"加载标注失败: {label_file}")
        image_file = image_dir / (label_file.stem + IMG_EXT)
        pairs.append((image_file, label))
    return pairs


def load_label_dir(folder: StrPath, meta_id: int) -> A2dImageLabels:
    """加载目录下的图片标注记录"""
    folder = Path(folder)
    rs = []
    tail = label_tail(meta_id, ".todo")
    logger.info(f"tail: {tail}")

    files = sorted(folder.rglob("*" + tail))
    for lbl_file in files:
        label = load_json(lbl_file, A2dImageLabel).expect(f"加载标注失败: {lbl_file}")
        rs.append(label)
    return rs


def dump_label_prop(
    label_pairs: A2dImageLabelPairs,
    dst: Path,
    category_id: int,
    prop_id: int,
    keep_dst_dir: bool,
    prefix: str,
) -> int:
    """保存标注多项属性, 生成分类样本"""

    if not keep_dst_dir:
        remake_dir(dst)

    total = 0
    for file, label in label_pairs:
        image = ImageNda.load(file)
        n = 0
        for o in label.objects:
            if o.prob_class.value == category_id:
                cat = o.prop(prop_id).value
                if cat < 0:
                    continue
                n += 1
                path = dst / str(cat) / f"{prefix}{file.stem}_{n:04}{IMG_EXT}"
                obj_img = image.roi(o.rect())
                obj_img.save(path)
        total += n
    return total


def dump_label_prop_demo(label_pairs: A2dImageLabelPairs, dst_dir: Path) -> int:
    """保存标注多项属性, 生成分类样本"""

    remake_dir(dst_dir)

    total = 0
    for file, label in label_pairs:
        image = ImageNda.load(file)
        n = 0
        for o in label.objects:
            if o.prob_class.value >= 0:
                n += 1
                path = dst_dir / file.stem / f"{o.id:02}{IMG_EXT}"
                make_parents(path)
                obj_img = image.roi(o.rect())
                obj_img.save(path)
        total += n
    return total


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


_class_map: Final[dict[LabelFormat, type[A2dLabelSet]]] = {
    LabelFormat.A2D: A2dSampleSet,
    # LabelFormat.DARKNET: DarknetSet 由 jxl.label.formats 包导入时反向注册（避免循环 import）。
}


def register_label_set(fmt: LabelFormat, cls: type[A2dLabelSet]) -> None:
    """注册格式实现 —— 供 formats 层装配（dataset 不反向 import formats，避免循环）."""
    _class_map[fmt] = cls


def open_label_set(
    folder: Path, label_format: LabelFormat, meta_id: int
) -> Result[A2dLabelSet, str]:
    """打开标注数据集合"""

    fmt_cls = guess_format_cls(folder, label_format, meta_id)

    if fmt_cls is None:
        return Err(f'Format "{label_format}" not found')

    if not fmt_cls.valid_set(folder, meta_id=meta_id):
        return Err(f'Invalid folder for format "{label_format}"')

    return Ok(fmt_cls(folder, meta_id))


def guess_format_cls(
    folder: Path, label_format: LabelFormat, meta_id: int
) -> type[A2dLabelSet] | None:
    """猜测数据集格式"""
    if label_format:
        return _class_map.get(label_format)
    for cls in _class_map.values():
        if cls.valid_set(folder, meta_id):
            return cls
    return None


HOP_EXT: Final[str] = ".json"
"""HOP 标注文件扩展名。"""


def hop_label_path_of(img_file: StrPath, meta_id: int) -> Path:
    """获取图像对应的 HOP 标注文件路径。"""
    return label_path_of(img_file, HOP, meta_id, HOP_EXT)


def hop_load_label(img_file: StrPath, meta_id: int) -> Option[A2dImageLabel]:
    """加载图像对应的 HOP 标注。

    不存在返回 Null；损坏文件显式报错并点名文件——「无标注」与「标注损坏」
    不得混同（后者吞成空样本 = 静默数据丢失，与 hop_save_label 的显式
    传播裁决同源）。
    """
    label_file = hop_label_path_of(img_file, meta_id)
    if not label_file.is_file():
        return Null
    return Some(load_json(label_file, A2dImageLabel).expect(f"加载 HOP 标注失败: {label_file}"))


def hop_save_label(label: A2dImageLabel, img_file: StrPath, meta_id: int) -> Path:
    """保存 HOP 标注，返回标注文件路径；IO 失败显式抛出（不吞 save_json 结果）。"""
    label_file = hop_label_path_of(img_file, meta_id)
    save_json(label, label_file).expect(f"保存 HOP 标注失败: {label_file}")
    return label_file


def hop_load_labels(folder: StrPath, meta_id: int) -> A2dImageLabelPairs:
    """加载 HOP 格式（`hop_m{meta_id}`）的图像-标注对集。"""
    return load_image_label_pairs(folder, meta_id, HOP)


class LabelFilter(IntEnum):
    """标注记录过滤器（EXPORT 成员随 ias 归档删除，2026-09-21 迁移裁决）。

    成员值保持 jxl 原值（ALL=1/LABELED=3）——迁移保真，不重排。
    """

    ALL = 1
    """所有图片样本。"""
    LABELED = 3
    """已标注的样本。"""

    def has_label(self, image_file: Path, meta_id: int) -> bool:
        """判断图片在本过滤器下是否计入。"""
        if self == LabelFilter.ALL:
            return True
        return hop_label_path_of(image_file, meta_id).is_file()
