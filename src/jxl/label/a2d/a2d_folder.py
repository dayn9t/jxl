from pathlib import Path
from typing import Final

from fontTools.varLib.interpolatable import ensure_parent_dir
from jcx.sys.fs import make_parents, copy_file, move_file
from jcx.text.txt_json import save_json

from jxl.label.a2d.dd import A2dImageLabel

A2D: Final[str] = "a2d"


class A2dLabelFolder:
    """A2D数据集文件夹基类"""

    @classmethod
    def meta_dir_name(cls, meta_id: int) -> str:
        """获取元数据文件夹名称"""
        return f"{A2D}_m{meta_id}"

    @classmethod
    def valid_set(cls, folder: Path, meta_id: int) -> bool:
        """检验路径是否是本格式的数据集"""
        return Path(folder, cls.meta_dir_name(meta_id)).is_dir()

    def __init__(self, folder: Path, meta_id: int):
        self._folder = folder
        self._meta_id = meta_id
        self._image_dir = self._folder / "image"
        self._label_dir = self._folder / self.meta_dir_name(meta_id)

    def add(self, image_file: Path, label: A2dImageLabel, move_image: bool = False) -> None:
        """添加图片与标签文件"""
        dst_image = self._image_dir / image_file.name
        dst_label = self._label_dir / image_file.with_suffix(".json").name
        make_parents(dst_image)
        make_parents(dst_label)

        if move_image:
            move_file(image_file, dst_image).unwrap()
        else:
            copy_file(image_file, dst_image).unwrap()
        save_json(label, dst_label).unwrap()
