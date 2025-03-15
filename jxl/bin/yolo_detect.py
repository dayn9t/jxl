#!/opt/ias/env/bin/python

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore
from jcx.sys.fs import find, stem_append
from jcx.ui.key import Key
from jvi.drawing.color import LIME
from jvi.drawing.shape import put_text
from jvi.geo.point2d import Point
from jvi.geo.size2d import size_parse
from jvi.image.image_nda import ImageNda
from jvi.image.proc import resize

from jxl.det.d2d import D2dOpt, draw_d2d_objects
from jxl.det.yolo.d2d_yolo import D2dYolo


# params:/home/jiang/ws/trash/cans/model_dir/can.pt /var/ias/snapshot/shtm/n1/work/2040600111/2021-04-18


def main():
    parser = argparse.ArgumentParser(description="Yolo5检测器")
    parser.add_argument("model", type=Path, help="模型文件路径")
    parser.add_argument("src_dir", type=Path, help="图像来源目录")
    parser.add_argument(
        "-c", "--conf_thr", type=float, default=0.5, help="置信度阈值"
    )
    parser.add_argument(
        "-i", "--iou_thr", type=float, default=0.7, help="非极大值抑制重叠率阈值"
    )
    parser.add_argument("-p", "--max_prob", type=float, default=1.99, help="概率上限")
    parser.add_argument("-w", "--wait", type=float, default=0, help="等待的秒数")
    parser.add_argument("-s", "--img_size", type=int, default=640, help="输入图像尺寸")
    parser.add_argument(
        "-o", "--output_size", type=str, default="HD", help="输出图像尺寸"
    )
    parser.add_argument("-d", "--dst_dir", type=Path, default=None, help="图像目标目录")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    opt = parser.parse_args()

    out_size = size_parse(opt.output_size)
    print(f"\nOPT: conf_thr={opt.conf_thr} iou_thr={opt.iou_thr}\n")
    print(f"\nOutput: size={out_size}\n")

    files = find(opt.src_dir, ".jpg")
    if not files:
        print("没有搜索到指定的文件")
        sys.exit(0)

    det_opt = D2dOpt(input_shape=(opt.img_size, opt.img_size), conf_thr=opt.conf_thr, iou_thr=opt.iou_thr)
    detector = D2dYolo(opt.model, det_opt)
    wait = int(opt.wait * 1000)
    image_out = ImageNda(out_size)

    # index = 0
    for file in files:
        image_in = ImageNda.load(str(file))  # BGR
        resize(image_in, image_out)
        print("  ", file)

        res = detector.detect(image_in)

        if len(res.objects) > 0:
            # if det.min_prob() > opt.max_prob:
            #    continue
            draw_d2d_objects(image_out, res.objects)
        else:
            print("Invalid det")
        text = f"Model: {opt.model.stem}"
        put_text(image_out, text, Point(x=32, y=32), LIME, thickness=2, scale=1)

        if isinstance(opt.dst_dir, Path):
            dst_file = stem_append(opt.dst_dir / file.name, opt.model.stem)
            image_out.save(str(dst_file))
        if wait >= 0:
            cv2.imshow(opt.src_dir.name, image_out.data())
            if cv2.waitKey(wait) == Key.ESC.value:
                break
    if wait >= 0:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
