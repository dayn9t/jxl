from pathlib import Path

from jvi.image.image_nda import ImageNda
from ultralytics import YOLO
from ultralytics.engine.results import Results

from jxl.det.d2d import Detector2D, D2dOpt, D2dResults, D2dResult
from jxl.det.yolo.adapter import boxes_to_d2d


class D2dYolo(Detector2D):
    """目标检测器"""

    model_class = "D2dYolo"

    def __init__(
        self,
        model_path: Path,
        opt: D2dOpt,
        device_name: str = "",
        verbose: bool = False,
    ):
        super().__init__(model_path, opt, device_name, verbose)

        self._model = YOLO(model_path, task="detect")

    def detect(self, image: ImageNda) -> D2dResult:
        """检测目标"""
        # data = image.data()[:, :, ::-1]  # BGR => RGB
        data = image.data()

        if self._opt.track:
            rs = self._model.track(
                data,
                persist=True,
                # conf=self._opt.conf_thr,
                # iou=self._opt.iou_thr,
                verbose=self._verbose,
            )
        else:
            rs = self._model(
                data,
                conf=self._opt.conf_thr,
                iou=self._opt.iou_thr,
                verbose=self._verbose,
            )
        assert isinstance(rs, list)
        assert len(rs) == 1
        assert isinstance(rs[0], Results)
        # print('YOLO result:', type(rs[0]))
        objects = boxes_to_d2d(rs[0].boxes)
        return D2dResult(objects=objects)
