#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

import cv2  # type: ignore
from ias.io import pack_source
from jcx.util.err import mand
from jiv.geo.point2d import Point
from jiv.geo.rectangle import Rect
from jiv.geo.size2d import Size, SIZE_FHD, size_parse
from jiv.image.io import load_image_pairs_in

epilog = '''
Examples:

    cd-cut src_dir dst_dir
    
'''

SIZE = SIZE_FHD


def cut_n(src_dir: Path, dst_dir: Path, block_size: Size, n: int, ext: str) -> None:
    image_pairs = load_image_pairs_in(src_dir, ext)
    print(f'加载图片: {len(image_pairs)}')
    assert image_pairs, '没有找到任何图片'

    # 等距离从图像水平选择n块区域, 各块可重叠
    x0 = 0
    y0 = SIZE.height - block_size.height  # 试图躲避时间OSD
    dx = (SIZE.width - block_size.width) // (n - 1)
    rects = [Rect.from_ps(Point(x0 + dx * i, y0), block_size) for i in range(n)]
    # print(rects)

    # 裁切并保存
    prefix = pack_source(src_dir)
    for file, im in image_pairs:
        assert im.size() == SIZE_FHD
        for i in range(n):
            dst_file = dst_dir / f'{prefix}_{i + 1}' / (file.stem + '.png')
            print(f'保存: {dst_file}')
            im.roi(rects[i]).save(dst_file)


def main() -> None:
    parser = argparse.ArgumentParser(description='用于变化检测的图片切分工具', epilog=epilog)
    parser.add_argument('src_dir', type=Path, help='来源图片')
    parser.add_argument('dst_dir', type=Path, help='目的图片')
    parser.add_argument('-s', '--block_size', type=str, default='1024x1024', help='从图片裁切块的尺寸')
    parser.add_argument('-n', '--block_num', type=int, default=3, help='从图片裁切的块数')
    parser.add_argument('-e', '--exp', type=str, default='.jpg', help='图片文件扩展名')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    block_size = mand(size_parse(opt.block_size))

    cut_n(opt.src_dir, opt.dst_dir, block_size, opt.block_num, opt.exp)


if __name__ == '__main__':
    main()
