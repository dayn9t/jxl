from pathlib import Path
from typing import Callable

from jcx.sys.fs import files_in, dirs_in
from jvi.image.image_nda import ImageNda
from pandas import DataFrame, concat
from pydantic import BaseModel


def num_columns(count: int, prefix: str = "c") -> list[str]:
    """获取指定前缀的序号列名称"""
    return [f"{prefix}{i}" for i in range(count)]


def mat_to_df(mat: list[list[float]], col_prefix: str = "c") -> DataFrame:
    """矩阵转换为 DataFrame"""
    assert len(mat) > 0
    columns = num_columns(len(mat[0]), col_prefix)
    return DataFrame(mat, columns=columns)


class ClassCount(BaseModel):
    """分类计数"""

    cls: int
    """类别索引"""
    count: int
    """该类别数量"""


class Extractor(BaseModel):
    """分类样本特征提取"""

    fun: Callable[[ImageNda], list[float]]
    """特征提取函数"""
    vec_size: int
    """调整向来向量长度"""
    col_prefix: str = "c"
    """列名前缀"""
    label_name: str = "label"
    """标签列名称"""
    image_ext: str = ".jpg"
    """图片文件扩展名"""

    def extract_dir(self, image_dir: Path) -> list[list[float]]:
        """将目录图片特征提取到2D数组"""
        files = files_in(image_dir, self.image_ext)
        assert len(files) > 0
        mat = []
        for file in files:
            im = ImageNda.load(file)
            v = self.fun(im)
            assert len(v) == self.vec_size
            mat.append(v)
        return mat

    def columns(self) -> list[str]:
        """获取全部列名称"""
        return num_columns(self.vec_size, self.col_prefix) + [self.label_name]

    def extract_classes(self, sample_dir: Path, data_file: Path) -> list[ClassCount]:
        """提取多个类别的图片特征到csv文件"""

        stat = []
        samples = DataFrame()
        dirs = dirs_in(sample_dir)
        assert len(dirs) > 0, f"{sample_dir} 没找到分类样本目录"
        for dir_ in dirs:
            mat = self.extract_dir(dir_)
            df = DataFrame(mat)
            cls = int(dir_.name)  # 目录名作为类别索引
            df[self.label_name] = cls
            stat.append(ClassCount(cls=cls, count=len(mat)))
            samples = concat([samples, df], ignore_index=True)

        samples.columns = self.columns()
        samples.to_csv(data_file, index=False)
        return stat
