#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jxl.model.util import show_model


def main():
    parser = argparse.ArgumentParser(description='深度学习模型工具')
    parser.add_argument('model', type=Path, help='模型文件路径')

    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    show_model(opt.model, opt)


if __name__ == '__main__':
    main()
