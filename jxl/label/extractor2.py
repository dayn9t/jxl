from typing import Any

from jcx.sys.fs import last_parts
from jml.label.extractor import *
from loguru import logger
from pandas import DataFrame, concat
from pydantic import BaseModel


class Extractor2(BaseModel):
    """匹配图样本特征提取"""
    fun: Callable[[ImageNda, ImageNda], list[float]]
    """特征提取函数"""
    vec_size: int
    """调整向来向量长度"""
    col_prefix: str = 'c'
    """列名前缀"""
    image_ext: str = '.jpg'
    """图片文件扩展名"""

    def extract_dir(self, i: int, image_dir: Path) -> list[list[Any]]:
        """将目录图片特征提取到2D数组"""
        files = files_in(image_dir, self.image_ext)
        logger.info(f'#{i} 提取: {image_dir}({len(files)})')
        if len(files) < 2:
            logger.warning(f'目录中至少需要两张图片: {image_dir}')
            return []
        mat = []
        file_a = files[0]
        im_a = ImageNda.load(file_a)
        for file_b in files[1:]:
            im_b = ImageNda.load(file_b)
            v = self.fun(im_a, im_b)
            assert len(v) == self.vec_size

            mat.append([last_parts(file_a, 2), last_parts(file_b, 2)] + v)
            file_a = file_b
            im_a = im_b

        return mat

    def columns(self) -> list[str]:
        """获取全部列名称"""
        return ['a', 'b'] + num_columns(self.vec_size, self.col_prefix)

    def extract_cameras(self, cameras_dir: Path, data_file: Path) -> int:
        """提取多个类别的图片特征到csv文件"""

        samples = DataFrame()
        dirs = dirs_in(cameras_dir)
        assert len(dirs) > 0, f'{cameras_dir} 没找到摄像机目录'
        for i, dir_ in enumerate(dirs):
            mat = self.extract_dir(i, dir_)
            if not mat:
                continue
            df = DataFrame(mat)

            samples = concat([samples, df], ignore_index=True)

        samples.columns = self.columns()
        samples.to_csv(data_file, index=False)
        return len(samples)
