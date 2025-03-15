import cv2  # type: ignore
from jcx.sys.fs import files_in
from jxl.det.yolo_old.idetector_old import draw_objects
from jxl.det.yolo_old.detector_y8 import *


def a_test() -> None:
    conf = 0.1
    iou = 0.7  # 出现重叠框, 则需要调高
    # model_file = Path('/opt/ias/env/lib/pyias/model/cabin.pt')
    model_file = Path(
        "/home/jiang/ws/trash/cabin/model_dir/cabin.pt"
    )  # TODO: 平均 12ms
    # model_file = Path('/home/jiang/ws/trash/cabin/model_dir/cabin-y8m.engine') # TODO: 平均 8ms, 优势明显, 还有提升空间

    files = files_in("/home/jiang/ws/trash/dates/2023-03-15/image", ".jpg")

    opt = DetOpt((640, 640), conf, iou, verbose=True)
    model = DetectorY8(model_file, opt)

    # results = model(image.data()[:, :, ::-1])
    # print('results', results)
    for i, f in enumerate(files):
        image: ImageNda = ImageNda.load(f)
        print(f"#{i} {f}")
        ret = model(image)
        # print('YOLO result:', len(ret))

        draw_objects(image, ret.objects())
        # res_plotted = rs[0].plot()
        cv2.imshow("result", image.data())
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def export_1() -> None:
    model_file = Path("/home/jiang/ws/trash/cabin/model_dir/cabin-y8m.pt")
    # model_file = '/home/jiang/ws/trash/cabin/yolov8n.pt'

    model = YOLO(model_file)
    # model.export(format='onnx')
    model.export(format="engine", device=0)


def export_2() -> None:
    model_file = "/home/jiang/ws/trash/cabin/yolov8n.onnx"

    # FIXME: 需要降级 torch*, onnx, 大概opencv依赖旧版本
    net = cv2.dnn.readNetFromONNX(str(model_file))
    print(net)


def engine1() -> None:
    model_file = "/home/jiang/ws/trash/cabin/model_dir/cabin-y8m.engine"
    model = YOLO(model_file)


if __name__ == "__main__":
    a_test()
    # export_1()
