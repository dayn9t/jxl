from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Dict
from typing import TypeAlias

from jvi.drawing.color import COLORS7
from jvi.image.image_nda import ImageNda
from pydantic import BaseModel

from jxl.det.d2d import D2dOpt, D2dResult, D2dObject
from jvi.geo.point2d import Points

from jxl.io.draw import draw_boxf
from jxl.label.a2d.dd import A2dImageLabel, A2dObjectLabel
from jxl.label.prop import ProbValue


class A2dOpt(BaseModel):
    """2D目标分析器选项"""

    d2d: D2dOpt
    """2D目标检测器选项"""
    d2d_name: str
    """2D目标检测器名称"""
    props: Dict[int, str]
    """属性名称集合, key: 属性索引, value: 属性名称"""


class A2dObject(D2dObject):
    """检测到的2D目标"""

    props: Dict[int, List[float]]
    """属性值概率分布集合"""

    def conf_int(self) -> int:
        """获取置信度的整数值"""
        return int(self.conf * 100)

    def to_label(self) -> A2dObjectLabel:
        """将检测到的2D目标转换为标注格式

        将当前的A2dObject对象转换为标注格式A2dObjectLabel，
        包括类别、置信度、边界框和属性信息。

        Returns:
            A2dObjectLabel: 对应的目标标注数据
        """
        # 转换polygon点集，使用矩形的四个角点
        polygon = self.rect.vertexes()

        props = {k: v for k, v in self.props.items()}

        # 转换属性，保持原有结构
        return A2dObjectLabel(
            id=self.id,
            prob_class=ProbValue(self.cls, self.conf),
            polygon=polygon,
            properties=props,
        )


A2dObjects: TypeAlias = List[A2dObject]
"""检测到的2D目标集合"""


class A2dResult(BaseModel):
    """2D目标检测器结果"""

    roi: Points
    """检测区域"""
    objects: A2dObjects
    """目标"""

    def to_label(self) -> A2dImageLabel:
        """
        将 A2dResult 转换为 A2dImageLabel。

        Returns:
            A2dImageLabel: 转换后的A2dImageLabel对象。
        """
        objects = [ob.to_label() for ob in self.objects]
        return A2dImageLabel(user_agent="a2d_result", roi=self.roi, objects=objects)


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

    @abstractmethod
    def detect(self, image: ImageNda, persist: bool = True) -> A2dResult:
        """检测"""
        pass


def draw_a2d_objects(
    canvas: ImageNda, objects: A2dObjects, thickness: int = 2, no_label: bool = False
) -> None:
    """绘制检测条目"""
    for ob in objects:
        color = COLORS7[ob.cls]
        label = "" if no_label else f"{ob.id}({ob.conf_int()})"
        draw_boxf(canvas, ob.rect, color, label, thickness)
