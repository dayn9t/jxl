#!/opt/ias/env/bin/python

import argparse
import glob
import os
import sys

import torch
from jcx.ui.key import Key
from jvi.image.image_nda import ImageNda
from jvi.image.trace import trace_image
from jxl.cls.classifier import ClassifierOpt
from jxl.cls.classifier_tch import ClassifierTch
from jxl.io.draw import draw_class_item


def main():
    parser = argparse.ArgumentParser(description="图像分类器")
    parser.add_argument("model_path", type=str, help="模型文件路径")
    parser.add_argument("source", type=str, help="数据源，文件/目录")
    parser.add_argument("--img-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("-n", "--num_classes", type=int, help="类别数")
    parser.add_argument(
        "-N", "--non_normalized", action="store_true", help="放弃数据归一化"
    )
    parser.add_argument(
        "-f",
        "--data_format",
        type=int,
        default=0,
        help="数据格式：0-完整模型，1-参数包",
    )
    parser.add_argument("-m", "--max-prob", type=float, default=1, help="概率上限")
    parser.add_argument(
        "--category", type=int, default=-1, help="类别，取值：[0,N]，默认不限制"
    )
    parser.add_argument("-s", "--save_model_file", type=str, help="保存完整模型到文件")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    opt = parser.parse_args()

    if os.path.isdir(opt.source):
        s = os.path.join(opt.source, "**/*.jpg")
        files = sorted(glob.glob(s, recursive=True))
    elif os.path.isfile(opt.source):
        files = [opt.source]
    else:
        print("数据源不存在")
        sys.exit(0)

    with torch.no_grad():  # 不计算导数

        cls_opt = ClassifierOpt(
            (opt.img_size, opt.img_size),
            opt.num_classes,
            not opt.non_normalized,
            opt.data_format,
        )
        print("classifier opt:", cls_opt)
        classifier = ClassifierTch(opt.model_path, cls_opt)

        if opt.verbose:
            classifier.show_detail()

        for i, f in enumerate(files):
            image = ImageNda.try_load(f)
            if image.is_err():
                print(i, f + "\tERROR")
                continue
            image = image.unwrap()
            ret = classifier(image)
            # print('ret:', ret)
            if len(ret) > 0:
                print("#%d" % i, ret.top_index(), f)
                if ret.top_confidence() > opt.max_prob:
                    continue
                draw_class_item(image, ret.top())
            else:
                print(i, f, "Invalid res")

            key = trace_image(image)
            if key == Key.ESC.value:
                break
        if opt.save_model_file:
            classifier.save(opt.save_model_file)


if __name__ == "__main__":
    main()
