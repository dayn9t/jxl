#!/opt/ias/env/bin/python

import argparse
import os

from jxl.label.coco import DataCoco


def main():
    parser = argparse.ArgumentParser(description="COCO转Darknet格式")
    parser.add_argument("coco_file", metavar="COCO_FILE", type=str, help="COCO格式文件")
    parser.add_argument(
        "darknet_dir", metavar="DARKNET_DIR", type=str, help="Darknet样本标注目录"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")

    opt = parser.parse_args()
    if os.path.splitext(opt.coco_file)[-1] == "":
        opt.coco_file += ".json"

    opt.coco_file = os.path.abspath(opt.coco_file)
    opt.darknet_dir = os.path.abspath(opt.darknet_dir)

    coco_data = DataCoco(opt.coco_file)

    opt.cats = None
    coco_data.dump_darknet(opt.darknet_dir, opt.cats, opt.verbose)


if __name__ == "__main__":
    main()
