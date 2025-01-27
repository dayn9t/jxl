#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jiv.geo.size2d import size_parse
from jxl.label.darknet import darknet_dump_labels
from jxl.label.hop import hop_load_labels
from jxl.label.meta import find_meta


def resize_labels(labels: ImageLabelPairs, folder: Path, meta: LabelMeta) -> int:
    """保存darknet样本标注信息, TODO: meta 可能有更多的用处"""

    labels_dir = remake_subdir(folder, 'labels')
    images_dir = remake_subdir(folder, 'images')

    print('\n开始生成样本(%d)：' % len(labels))
    total = 0
    for image, label in labels:
        src: ImageNda = ImageNda.load(image)
        jpg = images_dir / f'{image.stem}.jpg'
        txt = labels_dir / f'{image.stem}.txt'
        save_objects(label.objects, txt)

        dst = get_roi_image(src, label.roi().unwrap(), Color.parse(meta.sample.background))
        dst.save(jpg)
        total += len(label.objects)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description='改变样本尺寸程序')
    parser.add_argument('src_dir', metavar='SRC_DIR', type=Path, help='来源样本标注目录')
    parser.add_argument('dst_dir', metavar='DST_DIR', type=Path, help='目的样本标注目录')
    parser.add_argument('meta_id', metavar='META_ID', type=int, help='元数据ID')
    parser.add_argument('-s', '--size', type=str, default='640x640', help='目标样本图像尺寸')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    assert opt.src_dir.is_dir(), f'数据来源目录不存在: {opt.src_dir}'
    dst_size = size_parse(opt.size)
    assert dst_size, f'目标样本图像尺寸无效: {opt.size}'

    meta = find_meta(opt.meta_id, opt.src_dir).unwrap()

    print(f'加载目录: {opt.src_dir}')

    labels = hop_load_labels(opt.src_dir, opt.meta_id)

    assert len(labels) > 0

    dst_dir = opt.dst_dir

    # TODO: 坐标变换
    total = darknet_dump_labels(labels, dst_dir, meta)
    print(f"\n样本({total})生成完毕!")


if __name__ == '__main__':
    # catch_show_err(main, True)
    main()
