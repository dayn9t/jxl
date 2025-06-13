#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jcx.util.err import catch_show_err
from jxl.cls.dataset import dataset_split


# 参考：https://zhuanlan.zhihu.com/p/48976706　分层采样的方式分割数据集


def main() -> None:
    parser = argparse.ArgumentParser(description="样本分割程序")
    parser.add_argument(
        "src", type=Path, help="来源样本目录，支持darknet/imagenet数据集"
    )
    parser.add_argument("dst", type=Path, help="目的目录，包括：train，val，test")
    parser.add_argument("-e", "--exp", type=str, default=".jpg", help="图片文件扩展名")
    parser.add_argument(
        "-r", "--radio", nargs=3, type=int, default=[8, 1, 1], help="三个集合的比例"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    opt = parser.parse_args()

    dataset_split(opt.src, opt.dst, opt.radio, opt.exp)


if __name__ == "__main__":
    catch_show_err(main)
