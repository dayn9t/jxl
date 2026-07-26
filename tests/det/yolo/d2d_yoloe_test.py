from pathlib import Path

from jvi.image.image_nda import ImageNda
from jvi.image.trace import close_all_windows, trace_image

from jxl.det.d2d import D2dOpt, draw_d2d_objects
from jxl.det.yolo.d2d_yoloe import D2dYoloE


def a_test() -> None:
    file = "/home/jiang/cc/py/jxl/assets/person/p2.jpg"
    conf = 0.4
    iou = 0.5
    opt = D2dOpt(input_shape=(640, 640), conf_thr=conf, iou_thr=iou)

    model_name = "yoloe-11l-seg.pt"
    model_file = Path("/home/jiang/cc/py/jxl/models/yoloe", model_name)
    names = ["person"]

    model = D2dYoloE(model_file, opt, names)

    image: ImageNda = ImageNda.load(file)

    ret = model.detect(image)

    draw_d2d_objects(image, ret.objects)

    trace_image(image, "result")

    close_all_windows()


if __name__ == "__main__":
    a_test()
