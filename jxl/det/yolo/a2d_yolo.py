from pathlib import Path

from jvi.image.image_nda import ImageNda
from ultralytics import YOLO

from jxl.det.a2d import Analyzer2D, A2dOpt, A2dResult, from_d2d
from jxl.det.d2d import D2dResult
from jxl.det.yolo.d2d_yolo import D2dYolo


class A2dYolo(Analyzer2D):
    """2D目标分析器"""

    model_class = "A2dYolo"

    def __init__(
        self, model_dir: Path, opt: A2dOpt, device_name: str = "", verbose: bool = False
    ):
        """创建分类器, 为 cls.new 提供模板"""
        super().__init__(model_dir, opt, device_name, verbose)

        self._d2d_model = D2dYolo(
            model_dir / opt.d2d_name, opt.d2d, device_name, verbose
        )
        self._prop_models = {}
        for _id, name in opt.props.items():
            model = YOLO(model_dir / name, task="classify")
            self._prop_models[_id] = model

    def detect(self, image: ImageNda) -> A2dResult:
        """检测"""
        d2d: D2dResult = self._d2d_model.detect(image)

        a2d = from_d2d(d2d)

        for _id, model in self._prop_models.items():
            for obj in a2d.objects:
                roi = image.roi(obj.rect)
                rs = model(roi.data(), verbose=False)
                obj.props[_id] = rs[0].probs.data.cpu().tolist()

        return a2d
