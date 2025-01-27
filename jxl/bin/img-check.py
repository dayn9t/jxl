#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jxl.cls.dataset_checker import DatasetChecker


def main() -> None:
    parser = argparse.ArgumentParser(description='图像分类器测试')
    parser.add_argument('model', type=Path, help='模型文件路径')
    parser.add_argument('dataset', type=Path, help='待测试数据集')
    parser.add_argument('--img-size', type=int, default=224, help='输入图像尺寸，默认224')
    parser.add_argument('-n', '--num_classes', type=int, help='类别数')
    parser.add_argument('-N', '--non_normalized', action='store_true', help='不归一化数据')
    parser.add_argument('-r', '--review', action='store_true', help='复核错误 & 不可信数据')
    parser.add_argument('-c', '--max_conf', type=float, default=0.95, help='复核时筛选的最大置信度')
    parser.add_argument('-i', '--class_id', type=int, help='只处理指定类别')
    parser.add_argument('-t', '--top_num', type=int, default=20, help='复核时每个类别最多显示的样本条目')
    parser.add_argument('-e', '--ext', type=str, default='.jpg', help='样本图片扩展名')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()
    print('class_id:', opt.class_id)

    tester = DatasetChecker(opt.model, opt, opt.max_conf, opt.top_num, opt.ext)

    tester.check(opt.dataset, opt.class_id)


if __name__ == '__main__':
    main()
