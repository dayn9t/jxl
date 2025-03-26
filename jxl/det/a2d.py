from abc import abstractmethod, ABC
from pathlib import Path
from typing import Self, List, Dict
from typing import TypeAlias, Type

from fontTools.qu2cu.qu2cu import Point
from jvi.drawing.color import COLORS7
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from pydantic import BaseModel
from ultralytics import YOLO
from jxl.det.d2d import D2dOpt, D2dResult, D2dObject
from jxl.det.yolo.d2d_yolo import D2dYolo
from jxl.io.draw import draw_boxf
from jxl.model.types import ModelInfo


class A2dOpt(BaseModel):
    """2D目标分析器选项"""

    d2d: D2dOpt
    d2d_name: str

    props: Dict[int, str]


class A2dObject(D2dObject):
    """检测到的2D目标"""

    props: Dict[int, List[float]]
    """属性值概率分布集合"""

    def conf_int(self) -> int:
        """获取置信度的整数值"""
        return int(self.conf * 100)


A2dObjects: TypeAlias = List[A2dObject]
"""检测到的2D目标集合"""


class A2dResult(BaseModel):
    """2D目标检测器结果"""

    roi: List[Point]
    """检测区域"""
    objects: A2dObjects
    """目标"""


def from_d2d(d2d_result: D2dResult) -> A2dResult:
    """从2D检测结果创建"""
    objects = []
    for obj in d2d_result.objects:
        objects.append(
            A2dObject(
                id=obj.id,
                cls=obj.cls,
                conf=obj.conf,
                rect=obj.rect,
                props={},
            )
        )
    return A2dResult(roi=[], objects=objects)


A2dResults: TypeAlias = List[A2dResult]


class Analyzer2D(ABC):
    """2D目标分析器"""

    model_class = "Analyzer2D"

    @abstractmethod
    def __init__(
        self, model_dir: Path, opt: A2dOpt, device_name: str, verbose: bool = False
    ):
        """创建分类器, 为 cls.new 提供模板"""
        self._model_dir = model_dir
        self._opt = opt
        self._device_name = device_name
        self._verbose = verbose

    # @abstractmethod
    def detect(self, image: ImageNda) -> A2dResult:
        """检测"""
        pass
