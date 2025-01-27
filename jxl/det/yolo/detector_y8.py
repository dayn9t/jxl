from pathlib import Path

from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size
from jvi.image.image_nda import ImageNda
from jxl.det.idetector import DetRes, DetOpt, IDetector, DetObjects, DetObject
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results


def b2o(box: Boxes, image_size: Size) -> DetObject:
    """box => object"""
    arr = box.data.tolist()[0]  # box.shape: (1, 6)

    r = Rect.from_ltrb_list(arr).normalize(image_size)
    conf = arr[4]
    cls = int(arr[5])
    # print('r', r, 'cls', cls, 'conf', conf)
    return DetObject.new(cls, conf, r)


class YoloResU8(DetRes):
    """检测器结果"""

    def __init__(self, rs: Results, image_size: Size) -> None:
        self._objects = [b2o(b, image_size) for b in rs.boxes]

    def draw(self, img: ImageNda, colors, names=None) -> None:
        """画出检测器结果"""
        pass

    def __len__(self) -> int:
        return len(self._objects)

    def objects(self) -> DetObjects:
        """获取检测结果中的条目"""
        return self._objects


class DetectorY8(IDetector):
    """目标检测器"""

    model_class = "detector_y8"

    def __init__(self, model_path: Path, opt: DetOpt, device_name: str = ""):
        super().__init__(model_path, opt, device_name)

        self._model = YOLO(model_path, task="detect")

    def __call__(self, image: ImageNda) -> YoloResU8:
        """检测目标"""
        # data = image.data()[:, :, ::-1]  # BGR => RGB
        data = image.data()
        rs = self._model(
            data,
            conf=self._opt.conf_thr,
            iou=self._opt.iou_thr,
            verbose=self._opt.verbose,
        )
        assert isinstance(rs, list)
        assert len(rs) == 1
        assert isinstance(rs[0], Results)
        # print('YOLO result:', type(rs[0]))

        return YoloResU8(rs[0], image.size())
