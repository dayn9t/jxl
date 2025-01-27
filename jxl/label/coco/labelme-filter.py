#!/opt/ias/env/bin/python

import argparse
from datetime import datetime
from pathlib import Path

from dateutil.parser import parse

from jcx.fs import find


def main():
    parser = argparse.ArgumentParser(description='Labelme标注文件过滤 & 并链接到特定目录')
    parser.add_argument('src_dir', metavar='SRC_DIR', type=str, help='Labelme标注目录')
    parser.add_argument('dst_dir', metavar='DST_DIR', type=str, help='链接目录')
    parser.add_argument('-d', '--date', type=str, help='筛选指定日期的样本和标注')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')

    opt = parser.parse_args()
    opt_date = parse(opt.date).date()

    src_dir = Path(opt.src_dir).absolute()
    dst_dir = Path(opt.dst_dir).absolute()
    files = find(src_dir, '.json')

    for src in files:
        date = datetime.fromtimestamp(src.stat().st_mtime).date()
        # print(src, date)
        if date != opt_date:
            continue
        relative = src.relative_to(src_dir)
        print('relative:', relative)
        dst = dst_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
        dst.with_suffix('.jpg').symlink_to(src.with_suffix('.jpg'))


if __name__ == '__main__':
    main()
