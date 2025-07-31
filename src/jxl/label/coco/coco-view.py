#!/opt/ias/env/bin/python

import os
from abc import ABC

from jxl.coco.coco import DataCoco, show_label

import argparse

from jvi.drawing.color import rectangle, bgr_color_tab
from jvi.geo import Point, Size
from jvi.gui.image_viewer import ImageViewer
from jxl.det.y5.io import img2label, load_labels
from numpy import ndarray


class LabelViewer(ImageViewer, ABC):
    def __init__(self, size: Size):
        super().__init__("LabelViewer", size)
        self.labels = []

    def on_change_background(self, image, file):
        print("#%d" % self._index, file)

        label_file = img2label(file)
        print("label_file:", label_file)
        self.labels = load_labels(label_file)
        for label in self.labels:
            print("label:", label)

    def on_draw(self, canvas: ndarray, _pos: Point):
        for label in self.labels:
            color = bgr_color_tab[label.cat]
            rectangle(canvas, label.rect, color, 2)

    def load(self, coco_file):
        coco_data = DataCoco(coco_file)

        for id, label in sorted(coco_data.labels.items()):
            if id >= opt.start_index:
                if not show_label(label):
                    break


def main():
    parser = argparse.ArgumentParser(description="COCO标注查看")
    parser.add_argument("coco_file", metavar="COCO_FILE", type=str, help="COCO格式文件")
    parser.add_argument(
        "-s",
        "--start_index",
        default=0,
        metavar="START_INDEX",
        type=int,
        help="样本起始索引",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")

    opt = parser.parse_args()
    if os.path.splitext(opt.coco_file)[-1] == "":
        opt.coco_file += ".json"

    opt.coco_file = os.path.abspath(opt.coco_file)

    coco_data = DataCoco(opt.coco_file)

    for id, label in sorted(coco_data.labels.items()):
        if id >= opt.start_index:
            if not show_label(label):
                break


if __name__ == "__main__":
    main()
