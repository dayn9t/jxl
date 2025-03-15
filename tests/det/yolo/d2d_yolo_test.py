import cv2  # type: ignore
from jcx.sys.fs import files_in

from jxl.det.d2d import D2dOpt
from jxl.det.yolo.d2d_yolo import D2dYolo
from jxl.det.yolo_old.idetector_old import draw_objects
from jxl.det.yolo_old.detector_y8 import *


def a_test() -> None:
    conf = 0.3
    iou = 0.5
    model_file = Path("/home/jiang/ws/s4/sign/model_dir/2025-03-05_sign.pt")

    files = files_in("/var/howell/s4/ias/snapshot/d1/n1/12/0/2025-03-07", ".jpg")

    opt = D2dOpt(input_shape=(640, 640), conf_thr=conf, iou_thr=iou)
    model = D2dYolo(model_file, opt)

    # results = model(image.data()[:, :, ::-1])
    # print('results', results)
    for i, f in enumerate(files):
        image: ImageNda = ImageNda.load(f)
        print(f"#{i} {f}")
        ret = model(image)
        # print('YOLO result:', len(ret))

        # draw_objects(image, ret.objects())
        # res_plotted = rs[0].plot()
        cv2.imshow("result", image.data())
        cv2.waitKey(0)

    cv2.destroyAllWindows()
