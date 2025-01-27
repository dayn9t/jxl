#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jiv.gui.record_viewer import RecordViewer
from jxl.label.factory import open_label_set
from jxl.label.label_set import LabelFormat
from jxl.label.meta import find_meta
from jxl.label.viewer import LabelRecord


def main() -> int:
    parser = argparse.ArgumentParser(description='标注样本查看器')
    parser.add_argument('folder', type=Path, help='图片目录')
    parser.add_argument('-m', '--meta_id', type=int, default=0, help='元数据ID')
    parser.add_argument('-p', '--pattern', type=str, help='文件要匹配的模式，用于过滤数据')
    parser.add_argument('-f', '--format', type=str, default='',
                        help='标注格式: hop,imagenet,darknet,kitti,coco,google')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    meta = find_meta(opt.meta_id, opt.folder).unwrap()
    print('加载目录:', opt.folder)

    label_format = None
    if opt.format:
        label_format = LabelFormat.parse(opt.format)
        if label_format is None:
            print('Invalid format:', opt.format)
            return -1

    assert label_format
    label_set = open_label_set(opt.folder, label_format, opt.meta_id, opt.pattern).unwrap()
    print(f'标注集: {label_set}')

    label_pairs = label_set.load_pairs()
    print('样本总数:', len(label_pairs))

    rs = [LabelRecord(meta, image, label) for image, label in label_pairs]

    win = RecordViewer('files://' + str(opt.folder), meta.view_size)
    win.set_records(rs)

    win.run()
    return 0


if __name__ == '__main__':
    # catch_show_err(main)
    main()
