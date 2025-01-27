#!/opt/ias/env/bin/python
from pathlib import Path

import fire
from jiv.geo.size2d import size_parse
from jml.iqa.diag_extractor import MatchVec
from jml.label.extractor2 import Extractor2


def extract_all(cameras_dir: str, cols: int = 5, rows: int = 3, block_size: str = '256x240',
                image_ext: str = '.jpg') -> None:
    """从摄像机目录文件各个子目录中相邻图像文件的提取匹配特征"""
    path = Path(cameras_dir)
    assert path.is_dir(), f'指定数据目录不存在:{cameras_dir}'
    data_file = path.with_suffix('.csv')

    size = size_parse(block_size)
    assert size, f'无效的尺寸: {block_size}'

    match_vec = MatchVec(cols, rows, size)

    e = Extractor2(fun=match_vec, vec_size=cols * rows, image_ext=image_ext)
    n = e.extract_cameras(path, data_file)
    print(f'生成样本文件: {data_file}({n})')


if __name__ == '__main__':
    fire.Fire(extract_all)
