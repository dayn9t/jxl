#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jiv.geo.rectangle import Rect
from jxl.coco.coco import DataCoco


def main():
    parser = argparse.ArgumentParser(description='COCO转Darknet格式')
    parser.add_argument('coco_file', metavar='COCO_FILE', type=Path, help='COCO格式文件')
    parser.add_argument('darknet_dir', metavar='DARKNET_DIR', type=Path, help='Darknet样本标注目录')
    parser.add_argument('-c', '--cats', nargs='+', type=str, default=None, help='选中的类别，默认全部')
    parser.add_argument('-r', '--rect', nargs='+', type=int, default=None, help='感兴趣区域RECT，默认全部')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')

    opt = parser.parse_args()

    coco_file = opt.coco_file.with_suffix('.json').absolute()
    print(coco_file)

    darknet_dir = opt.darknet_dir.absolute()
    print(darknet_dir)

    coco_data = DataCoco(coco_file)

    if opt.rect and len(opt.rect) != 4:
        print("ERROR: Invalid rect")
        exit(-1)
    rect = Rect(*opt.rect) if opt.rect else None
    if rect:
        print("ROI:", rect)
    coco_data.dump_darknet(darknet_dir, opt.cats, rect, opt.verbose)


if __name__ == '__main__':
    main()
