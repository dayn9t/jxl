#!/opt/ias/env/bin/python
from pathlib import Path

import fire
from jxl.iqa.diag_extractor import sharpness, chroma
from jxl.label.extractor import Extractor

fun_map = {
    'clearness': sharpness,
    'chroma': chroma,
}


def extract_all(data_dir: str, prop_name: str, vec_size: int = 256, label: str = 'label', ext: str = '.jpg') -> None:
    path = Path(data_dir)
    assert path.is_dir(), f'指定数据目录不存在:{data_dir}'
    fun = fun_map.get(prop_name)
    assert fun, f'不支持的属性: {prop_name}'

    names = ['train', 'val', 'test']
    e = Extractor(fun=fun, vec_size=vec_size, label_name=label, image_ext=ext)

    for name in names:
        dir_ = path / name
        file = path / (name + '.csv')
        m = e.extract_classes(dir_, file)
        s = 0
        print(f'生成样本文件: {file}')
        for cc in m:
            s += cc.count
            print(f'  class{cc.cls}\t{cc.count: 6}')
        print(f'  总数\t{s: 6}')


if __name__ == '__main__':
    fire.Fire(extract_all)
