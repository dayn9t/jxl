#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jml.label.hop import hop_load_labels
from jml.label.io import dump_label_prop_demo


def main() -> None:
    parser = argparse.ArgumentParser(description='样本生成程序')
    parser.add_argument('src_dir', metavar='SRC_DIR', type=Path, help='来源标注目录')
    parser.add_argument('dst_dir', metavar='DST_DIR', type=Path, help='目的样本目录')
    parser.add_argument('meta_id', metavar='META_ID', type=int, help='元数据ID')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    assert opt.src_dir.is_dir(), f'数据来源目录不存在: {opt.src_dir}'

    print(f'加载目录: {opt.src_dir}')
    labels = hop_load_labels(opt.src_dir, opt.meta_id)
    assert len(labels) > 0

    dst_dir = opt.dst_dir
    total = dump_label_prop_demo(labels, dst_dir)

    print(f"\n样本({total})生成完毕!")


if __name__ == '__main__':
    # catch_show_err(main, True)
    main()
