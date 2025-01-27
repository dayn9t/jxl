#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jiv.geo.size2d import size_parse
from jxl.label.blend import ObjectBlender

epilog = '''
Examples:

    cd_inpaint.py src_dir dst_dir object_dir
    
'''


def main() -> None:
    parser = argparse.ArgumentParser(description='变化检测(CD)样本绘制工具', epilog=epilog)
    parser.add_argument('src_dir', type=Path, help='来源图片目录')
    parser.add_argument('dst_dir', type=Path, help='目的图片目录')
    parser.add_argument('object_dir', type=Path, help='绘制图片目录')
    parser.add_argument('-c', '--size', type=str, default='1024x1024', help='目标图片尺寸')
    parser.add_argument('-m', '--multiple', type=int, default=5, help='目标样本倍数')
    parser.add_argument('-o', '--object_num', type=int, default=4, help='图片内绘制目标数量')
    parser.add_argument('-e', '--ext', type=str, default='.png', help='图片扩展名')

    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    size = size_parse(opt.size)
    assert size is not None

    blender = ObjectBlender(multiple=opt.multiple, ob_count=opt.object_num, size=size, verbose=opt.verbose)

    n = blender.load_objects(opt.object_dir, opt.ext)
    print('load objects:', n)
    n = blender.make_samples(opt.src_dir, opt.dst_dir)
    print(f'Done({n})!')


if __name__ == '__main__':
    main()
