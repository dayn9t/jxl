#!/opt/ias/env/bin/python

import argparse
import os

import labelme2coco
from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser(description='Labelme转COCO格式')
    parser.add_argument('labelme_dir', metavar='LABELME_DIR', type=str, help='Labelme样本标注目录')
    parser.add_argument('coco_file', metavar='COCO_FILE', type=str, help='COCO格式文件')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')

    opt = parser.parse_args()

    print(opt.labelme_dir, opt.coco_file)

    if os.path.splitext(opt.coco_file)[-1] == '':
        opt.coco_file += '.json'

    opt.labelme_dir = os.path.abspath(opt.labelme_dir)
    opt.coco_file = os.path.abspath(opt.coco_file)

    print(opt.labelme_dir, opt.coco_file)

    labelme2coco.convert(opt.labelme_dir, opt.coco_file)

    coco = COCO(opt.coco_file)

    cats = coco.loadCats(coco.getCatIds())
    names = sorted([cat['name'] for cat in cats])

    print('\nCOCO categories: ')
    for i, name in enumerate(names):
        print('  %02d. %s' % (i + 1, name))

    imgs = coco.loadImgs()
    print('\nCOCO loadImgs: %s' % imgs)


if __name__ == '__main__':
    main()
