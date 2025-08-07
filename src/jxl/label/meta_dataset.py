from pathlib import Path

from jcx.sys.fs import make_parents, copy_file, move_file
from jcx.text.txt_json import save_json
from jvi.image.image_nda import ImageNda
from pydantic import BaseModel


class MetaDataset:
    """带有Meta的样本集文件夹管理器"""

    @classmethod
    def meta_dir_name(cls, format_name: str, meta_id: int) -> str:
        """获取元数据文件夹名称"""
        return f"{format_name}_m{meta_id}"

    def __init__(self, folder: Path, format_name: str, meta_id: int):
        """创建A2D样本集文件夹管理器"""
        self._folder: Path = folder
        self._meta_id = meta_id
        self._format_name = format_name
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
            move_file(image_file, dst_image).unwrap()
            move_file(label_file, dst_label).unwrap()
        else:
            copy_file(image_file, dst_image).unwrap()
            copy_file(label_file, dst_label).unwrap()

    def add_sample(self, name: str, image: ImageNda, label: BaseModel) -> None:
        """添加样本"""
        dst_image = self._image_dir / f"{name}.jpg"
        dst_label = self._label_dir / f"{name}.json"
        make_parents(dst_image)
        make_parents(dst_label)

        image.save(dst_image)
        save_json(label, dst_label).unwrap()
