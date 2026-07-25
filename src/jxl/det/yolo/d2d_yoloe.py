"""YOLO-E目标检测器实现.

该模块基于Ultralytics YOLOE模型实现了2D目标检测器接口(Detector2D).
YOLO-E是YOLO系列的高效版本, 提供了优秀的检测性能和速度平衡.

参考文档: https://docs.ultralytics.com/models/yoloe/

典型用法:
1. 创建检测器选项 D2dOpt
2. 实例化D2dYoloE检测器类, 传入模型文件路径、选项和类别名称
3. 调用detect方法进行目标检测
4. 处理返回的D2dResult结果
"""

from pathlib import Path

from jvi.image.image_nda import ImageNda
from ultralytics import YOLOE

from jxl.det.d2d import D2dOpt, D2dResult, Detector2D
from jxl.yolo.results import results_list_to_d2d_result


class D2dYoloE(Detector2D):
    """基于YOLO-E的目标检测器.

    该类实现了Detector2D抽象接口, 使用Ultralytics的YOLOE模型进行目标检测.
    支持自定义类别名称和目标跟踪功能.
    """

    model_class = "D2dYoloE"
    """模型类型标识"""

    def __init__(
        self,
        model_path: Path,
        opt: D2dOpt,
        names: list[str],
        device_name: str = "",
        verbose: bool = False,
    ) -> None:
        """初始化YOLO-E目标检测器.

        Args:
            model_path: YOLO-E模型文件路径
            opt: 检测器选项, 包含置信度阈值、IOU阈值等参数
            names: 类别名称列表, 与模型输出的类别索引对应
            device_name: 运行设备名称, 如'cpu'或'cuda:0', 默认为空字符串, 表示自动选择
            verbose: 是否启用详细日志输出, 默认为False

        """
        super().__init__(model_path, opt, device_name, verbose)

        self._model = YOLOE(model_path)
        self._model.set_classes(names, self._model.get_text_pe(names))

    def detect(self, image: ImageNda, persist: bool = True) -> D2dResult:
        """检测图像中的目标.

        对输入图像进行目标检测.根据配置选择是否启用目标跟踪功能.

        Args:
            image: 输入的图像数据, 格式为NumPy数组
            persist: 当启用目标跟踪时, 是否保持跟踪状态, 默认为True.
                    这允许在视频序列中保持目标ID的连续性

        Returns:
            D2dResult: 包含检测到的目标列表的结果对象, 每个目标包含边界框、类别和置信度信息

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
