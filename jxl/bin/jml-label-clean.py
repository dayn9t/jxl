#!/opt/ias/env/bin/python

import argparse
import shutil
from pathlib import Path

from jcx.sys.fs import files_in
from jml.label.hop import HOP_EXT
from jml.label.info import IMG_EXT


def main() -> None:
    parser = argparse.ArgumentParser(description='清理标注数据中的无用数据')
    parser.add_argument('folder', type=Path, help='样本目录')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    folder: Path = opt.folder
    for d in folder.glob('ias_m*'):
        if d.is_dir():
            print(f'- 删除消息目录: {d.relative_to(folder)}')
            shutil.rmtree(d)

    labels = set()
    for d in folder.glob('hop_m*'):
        if d.is_dir():
            s = set([f.stem for f in files_in(d, HOP_EXT)])
            print(f'- 加载标注目录: {d.relative_to(folder)}({len(s)})')
            labels.update(s)

    img_dir = folder / 'image'
    total = 0
    n = 0
    for f in img_dir.glob('*' + IMG_EXT):
        if f.is_file():
            total += 1
            if f.stem not in labels:
                n += 1
                f.unlink()
    print(f'- 删除图像文件: image({n}/{total})')


if __name__ == '__main__':
    # catch_show_err(main, verbose=True)
    main()
