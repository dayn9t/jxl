"""2D目标检测模块.

该模块定义了2D目标检测相关的基础类和接口, 包括检测器选项、检测结果、以及检测器的抽象基类.
所有的2D目标检测器实现都应该继承自本模块定义的Detector2D抽象基类, 并实现其定义的接口.

典型用法:
1. 创建检测器选项 D2dOpt
2. 实例化具体检测器类
3. 调用检测器的detect方法进行目标检测
4. 处理返回的D2dResult结果

"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from jvi.drawing.color import COLORS7
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from pydantic import BaseModel

from jxl.io.draw import draw_boxf
from jxl.model.types import ModelInfo


class D2dOpt(BaseModel):
    """2D目标检测器选项.

    配置2D目标检测器的各项参数, 如输入图像尺寸、置信度阈值等.
    """

    input_shape: tuple = (640, 640)
    """输入形状, 默认为(640, 640)"""
    conf_thr: float = 0.5
    """置信度阈值, 小于此阈值的检测结果将被过滤, 默认为0.5"""
    iou_thr: float = 0.7
    """非极大值抑制中的重叠率阈值, 用于过滤重叠框, 默认为0.7"""
    track: bool = False
    """是否开启目标跟踪功能, 默认为False"""


class D2dObject(BaseModel):
    """检测到的2D目标.

    表示单个检测到的目标对象, 包含ID、类别、置信度和位置信息.
    """

    id: int
    """目标ID, 用于唯一标识一个检测到的目标"""
    cls: int
    """目标类别索引, 表示目标所属的类别"""
    conf: float
    """目标置信度, 表示检测结果的可信程度, 范围[0,1]"""
    rect: Rect
    """目标区域, 使用归一化坐标表示"""

    def conf_int(self) -> int:
        """获取置信度的整数值.

        将[0,1]范围的浮点置信度转换为0-100的整数值, 便于显示.

        Returns:
            int: 0-100范围内的整数置信度值

        """
        return int(self.conf * 100)


type D2dObjects = list[D2dObject]
"""检测到的2D目标集合, 表示为D2dObject对象的列表"""


class D2dResult(BaseModel):
    """2D目标检测器结果.

    封装单张图像的检测结果, 包含检测到的所有目标对象.
    """

    objects: D2dObjects
    """检测到的目标对象集合, 类型为D2dObjects(D2dObject列表)"""

    def min_conf(self) -> float:
        """获取最低置信度.

        计算所有检测目标中的最低置信度值.如果没有检测到目标, 则返回1.0.

        Returns:
            float: 最低置信度值, 如果没有目标则返回1.0

        """
        if not self.objects:
            return 1.0  # 没有目标时返回最高置信度
        return min(ob.conf for ob in self.objects)

    def empty(self)-> bool:
        """判定结果是否为空.

        检查是否存在检测到的目标对象.

        Returns:
            bool: 如果没有检测到任何目标则返回True, 否则返回False

        """
        return len(self.objects) == 0


type D2dResults = list[D2dResult]
"""检测到的2D目标结果集合, 表示为D2dResult对象的列表, 通常用于存储视频序列的检测结果"""


class Detector2D(ABC):
    """2D目标检测器抽象基类.

    定义了2D目标检测器的通用接口, 所有具体的2D目标检测器实现都应该继承此类.
    """

    model_class = "Detector2D"
    """模型类型标识, 用于标识这是一个2D检测器"""

    @classmethod
    def new(cls: type[Self], info: ModelInfo, model_root: Path) -> Self:
        """创建检测器实例.

        根据模型信息创建检测器实例的工厂方法.

        Args:
            info: 模型信息, 包含模型文件、设备、选项等
            model_root: 模型根目录, 用于与模型文件名组合构成完整路径

        Returns:
            Self: 返回创建的检测器实例

        """
        file = model_root / info.file
        return cls(file, info.opt, info.device)

    @abstractmethod
    def __init__(
        self, model_path: Path, opt: D2dOpt, device_name: str, verbose: bool = False
    ) -> None:
        """初始化检测器.

        这是一个抽象方法, 需要由子类实现.提供了基础的属性初始化逻辑.

        Args:
            model_path: 模型文件路径
            opt: 检测器选项
            device_name: 运行设备名称, 如'cpu'或'cuda:0'
            verbose: 是否启用详细日志输出

        """
        self._model_path = model_path
        self._opt = opt
        self._device_name = device_name
        self._verbose = verbose

    @abstractmethod
    def detect(self, image: ImageNda) -> D2dResult:
        """执行目标检测.

        在给定的图像上执行目标检测, 这是一个抽象方法, 需要由子类实现.

        Args:
            image: 输入图像

        Returns:
            D2dResult: 检测结果, 包含检测到的所有目标对象

        """


def draw_d2d_objects(
    canvas: ImageNda, objects: D2dObjects, thickness: int = 2, no_label: bool = False
) -> None:
    """绘制检测对象到图像上.

    将检测到的目标对象(边界框、ID和置信度)可视化到图像上.

    Args:
        canvas: 要绘制的图像画布
        objects: 要绘制的目标对象集合
        thickness: 边框线条粗细, 默认为2像素
        no_label: 是否不显示标签信息, 默认显示

    """
    for ob in objects:
        color = COLORS7[ob.cls]
        label = "" if no_label else f"{ob.id}({ob.conf_int()})"
        draw_boxf(canvas, ob.rect, color, label, thickness)
