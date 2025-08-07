from abc import abstractmethod, ABC
from pathlib import Path
from typing import Self, List
from typing import TypeAlias, Type

from jvi.drawing.color import COLORS7
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from pydantic import BaseModel

from jxl.io.draw import draw_boxf
from jxl.model.types import ModelInfo


class D2dOpt(BaseModel):
    """2D目标检测器选项"""

    input_shape: tuple = (640, 640)
    """输入形状"""
    conf_thr: float = 0.5
    """置信度阈值"""
    iou_thr: float = 0.7
    """非极大值抑制中的重叠率阈值"""
    track: bool = False
    """是否跟踪"""
    # class Config:
    #    allow_mutation = False


class D2dObject(BaseModel):
    """检测到的2D目标"""

    id: int
    """目标ID"""
    cls: int
    """目标类别索引"""
    conf: float
    """目标概率"""
    rect: Rect
    """目标区域，归一化"""

    def conf_int(self) -> int:
        """获取置信度的整数值"""
        return int(self.conf * 100)


D2dObjects: TypeAlias = List[D2dObject]
"""检测到的2D目标集合"""


class D2dResult(BaseModel):
    """2D目标检测器结果"""

    objects: D2dObjects
    """目标集合"""

    def min_conf(self) -> float:
        """获取最低置信度"""
        if not self.objects:
            return 1.0  # 没有目标时返回最高置信度
        return min(ob.conf for ob in self.objects)

    def empty(self):
        """判定结果是否为空"""
        return len(self.objects) == 0


D2dResults: TypeAlias = List[D2dResult]
"""检测到的2D目标集合"""


class Detector2D(ABC):
    """2D目标检测器"""

    model_class = "Detector2D"

    @classmethod
    def new(cls: Type[Self], info: ModelInfo, model_root: Path) -> Self:
        """创建检测器-根据信息"""
        file = model_root / info.file
        d = cls(file, info.opt, info.device)
        return d

    @abstractmethod
    def __init__(
        self, model_path: Path, opt: D2dOpt, device_name: str, verbose: bool = False
    ):
        """创建分类器, 为 cls.new 提供模板"""
        self._model_path = model_path
        self._opt = opt
        self._device_name = device_name
        self._verbose = verbose

    @abstractmethod
    def detect(self, image: ImageNda) -> D2dResult:
        """检测"""
        pass


def draw_d2d_objects(
    canvas: ImageNda, objects: D2dObjects, thickness: int = 2, no_label: bool = False
) -> None:
    """绘制检测条目"""
    for ob in objects:
        color = COLORS7[ob.cls]
        label = "" if no_label else f"{ob.id}({ob.conf_int()})"
        draw_boxf(canvas, ob.rect, color, label, thickness)
