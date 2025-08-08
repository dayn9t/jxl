from pathlib import Path

from jvi.image.image_nda import ImageNda
from ultralytics import YOLOE

from jxl.det.d2d import Detector2D, D2dOpt, D2dResult
from jxl.yolo.results import results_list_to_d2d_result

# https://docs.ultralytics.com/models/yoloe/


class D2dYoloE(Detector2D):
    """目标检测器"""

    model_class = "D2dYoloE"

    def __init__(
        self,
        model_path: Path,
        opt: D2dOpt,
        names: list[str],
        device_name: str = "",
        verbose: bool = False,
    ):
        super().__init__(model_path, opt, device_name, verbose)

        self._model = YOLOE(model_path)
        self._model.set_classes(names, self._model.get_text_pe(names))

    def detect(self, image: ImageNda, persist: bool = True) -> D2dResult:
        """检测图像中的目标

        对输入图像进行目标检测。根据配置选择是否启用目标跟踪功能。

        Args:
            image: ImageNda - 输入的图像数据，格式为NumPy数组
            persist: bool - 当启用目标跟踪时，是否保持跟踪状态，默认为True。
                            这允许在视频序列中保持目标ID的连续性

        Returns:
            D2dResult - 包含检测到的目标列表的结果对象，每个目标包含边界框、类别和置信度信息
        """
        # data = image.data()[:, :, ::-1]  # BGR => RGB
        data = image.data()

        if self._opt.track:
            results_list = self._model.track(
                data,
                persist=persist,
                verbose=self._verbose,
            )
        else:
            results_list = self._model.predict(
                data,
                conf=self._opt.conf_thr,
                iou=self._opt.iou_thr,
                verbose=self._verbose,
            )
        return results_list_to_d2d_result(results_list)
