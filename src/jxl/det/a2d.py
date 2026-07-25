from abc import ABC, abstractmethod
from pathlib import Path

from jvi.drawing.color import COLORS7
from jvi.geo.point2d import Points
from jvi.image.image_nda import ImageNda
from pydantic import BaseModel

from jxl.det.d2d import D2dObject, D2dOpt, D2dResult
from jxl.io.draw import draw_boxf
from jxl.label.a2d.dd import A2dImageLabel, A2dObjectLabel
from jxl.label.prop import ProbValue


class A2dOpt(BaseModel):
    """2D目标分析器选项"""

    d2d: D2dOpt
    """2D目标检测器选项"""
    d2d_name: str
    """2D目标检测器名称"""
    props: dict[int, str]
    """属性名称集合, key: 属性索引, value: 属性名称"""


def prop_to_label(
    prop_id: int, probs: list[float], prop_names: list[str]
) -> tuple[str, ProbValue]:
    """将属性概率分布转换为标签和值"""

    max_conf = max(probs)
    max_index = probs.index(max_conf)
    return prop_names[prop_id], ProbValue(max_index, max_conf)


def props_to_label(
    props: dict[int, list[float]], prop_names: list[str]
) -> dict[str, ProbValue]:
    """将属性概率分布集合转换为标签和值集合"""
    label_props = {}
    for prop_id, probs in props.items():
        prop_name, prob_value = prop_to_label(prop_id, probs, prop_names)
        label_props[prop_name] = prob_value
    return label_props


class A2dObject(D2dObject):
    """检测到的2D目标"""

    props: dict[int, list[float]]
    """属性值概率分布集合"""

    def conf_int(self) -> int:
        """获取置信度的整数值"""
        return int(self.conf * 100)

    def to_label(self, names: list[str]) -> A2dObjectLabel:
        """将检测到的2D目标转换为标注格式

        将当前的A2dObject对象转换为标注格式A2dObjectLabel，
        包括类别、置信度、边界框和属性信息。

        Returns:
            A2dObjectLabel: 对应的目标标注数据
        """

        # 转换属性，保持原有结构
        # TODO(dayn9t): props_to_label 返回 Dict[str, ProbValue]，但 A2dObjectLabel.properties
        # 期望 Dict[int, ProbValue]（按属性 ID）。det→label 属性映射存在名称/ID 设计缺口，
        # 后续重构 props_to_label 使其按 ID 输出，此处暂保留以维持现有行为。
        label_props = props_to_label(self.props, names)
        return A2dObjectLabel(
            id=self.id,
            prob_class=ProbValue(self.cls, self.conf),
            polygon=self.rect.vertexes(),
            properties=label_props,
        )


type A2dObjects = list[A2dObject]
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
        # TODO(dayn9t): A2dObject.to_label 需要 names 参数，但 A2dResult.to_label 无从获取
        # 类别名表。后续重构 A2dResult 携带 names，此处暂传空表维持现有调用约定。
        objects = [ob.to_label([]) for ob in self.objects]
        return A2dImageLabel(user_agent="a2d_result", roi=self.roi, objects=objects)

    def min_conf(self) -> float:
        """获取最低置信度"""
        if not self.objects:
            return 1.0  # 没有目标时返回最高置信度
        return min(ob.conf for ob in self.objects)

    def empty(self) -> bool:
        """判定结果是否为空"""
        return len(self.objects) == 0


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


type A2dResults = list[A2dResult]


class Analyzer2D(ABC):
    """2D目标分析器"""

    model_class = "Analyzer2D"

    @abstractmethod
    def __init__(
        self, model_dir: Path, opt: A2dOpt, device_name: str, verbose: bool = False
    ) -> None:
        """创建分类器, 为 cls.new 提供模板"""
        self._model_dir = model_dir
        self._opt = opt
        self._device_name = device_name
        self._verbose = verbose

    @abstractmethod
    def detect(self, image: ImageNda, persist: bool = True) -> A2dResult:
        """检测"""


def draw_a2d_objects(
    canvas: ImageNda, objects: A2dObjects, thickness: int = 2, no_label: bool = False
) -> None:
    """绘制检测条目"""
    for ob in objects:
        color = COLORS7[ob.cls]
        label = "" if no_label else f"{ob.id}({ob.conf_int()})"
        draw_boxf(canvas, ob.rect, color, label, thickness)
