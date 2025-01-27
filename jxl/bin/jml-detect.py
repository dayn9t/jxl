#!/opt/ias/env/bin/python

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore
from jcx.sys.fs import find, stem_append
from jcx.ui.key import Key
from jiv.drawing.color import LIME
from jiv.drawing.shape import put_text
from jiv.geo.point2d import Point
from jiv.geo.size2d import size_parse
from jiv.image.image_nda import ImageNda
from jiv.image.proc import resize
from jxl.det.idetector import DetOpt, draw_objects
from jxl.det.yolo.detector_y8 import DetectorY8


# params:/home/jiang/ws/trash/cans/model_dir/can.pt /var/ias/snapshot/shtm/n1/work/2040600111/2021-04-18

def main():
    parser = argparse.ArgumentParser(description='Yolo5检测器')
    parser.add_argument('model', type=Path, help='模型文件路径')
    parser.add_argument('src_dir', type=Path, help='图像来源目录')
    parser.add_argument('-c', '--conf_thres', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('-i', '--iou_thres', type=float, default=0.7, help='非极大值抑制重叠率阈值')
    parser.add_argument('-p', '--max_prob', type=float, default=1.99, help='概率上限')
    parser.add_argument('-w', '--wait', type=float, default=0, help='等待的秒数')
    parser.add_argument('-s', '--img_size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('-o', '--output_size', type=str, default='HD', help='输出图像尺寸')
    parser.add_argument('-d', '--dst_dir', type=Path, default=None, help='图像目标目录')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    out_size = size_parse(opt.output_size)
    print(f'\nOPT: conf_thres={opt.conf_thres} iou_thres={opt.iou_thres}\n')
    print(f'\nOutput: size={out_size}\n')

    files = find(opt.src_dir, '.jpg')
    if not files:
        print("没有搜索到指定的文件")
        sys.exit(0)

    det_opt = DetOpt((opt.img_size, opt.img_size), opt.conf_thres, opt.iou_thres)
    detector = DetectorY8(opt.model, det_opt)
    wait = int(opt.wait * 1000)
    image_out = ImageNda(out_size)

    # index = 0
    for file in files:
        image_in = ImageNda.load(str(file))  # BGR
        resize(image_in, image_out)
        print('  ', file)

        det = detector(image_in)

        if len(det) > 0:
            if det.min_prob() > opt.max_prob:
                continue
            '''print('objects:')
            for o in det.objects():
                print(o)'''
            draw_objects(image_out, det.objects())
        else:
            print('Invalid det')
        text = f'Model: {opt.model.stem}'
        put_text(image_out, text, Point(32, 32), LIME, thickness=2, scale=1)

        if isinstance(opt.dst_dir, Path):
            dst_file = stem_append(opt.dst_dir / file.name, opt.model.stem)
            image_out.save(str(dst_file))
        if wait >= 0:
            cv2.imshow(opt.src_dir.name, image_out.data())
            if cv2.waitKey(wait) == Key.ESC.value:
                break
    if wait >= 0:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
