from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypeAlias, TypeVar, Type

from jcx.util.lict import Lict
from jvi.drawing.color import COLORS7, Colors
from jvi.geo.point2d import Points, Point, array_normalize
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size
from jvi.image.image_nda import ImageNda
from jxl.label.prop import ProbValue, ProbProperties, ProbPropertyMap
from jxl.io.draw import draw_box
from jxl.model.types import ModelInfo


@dataclass(frozen=True)
class DetOpt:
    """检测器选项"""

    input_shape: tuple
    """输入形状"""
    conf_thr: float
    """置信度阈值"""
    iou_thr: float
    """非极大值抑制中的重叠率阈值"""
    augment: bool = False
    agnostic: bool = False
    verbose: bool = False
    """显示细节信息"""


@dataclass
class DetObject:
    """检测到的目标"""

    id: int
    """对象ID"""
    prob_class: ProbValue
    """类别 & 概率"""
    life: int = 1
    """生存周期"""
    polygon: Points = field(default_factory=list)
    """包含目标的多边形区域"""
    properties: ProbProperties = field(default_factory=list)
    """属性集合"""

    @staticmethod
    def new(
        class_index: int,
        confidence: float,
        rect: Optional[Rect] = None,
        polygon: Optional[Points] = None,
        id_: int = 0,
    ) -> "DetObject":
        """创建基本的检测目标"""
        if polygon is None:
            assert rect
            polygon = rect.vertexes()
        return DetObject(id_, ProbValue(class_index, confidence), polygon=polygon)

    def class_index(self) -> int:
        """类别索引"""
        return self.prob_class.value

    def rect(self) -> Rect:
        """获取外包矩形"""
        return Rect.bounding(self.polygon)

    def center(self) -> Point:
        """获取外包矩形中心"""
        return self.rect().center()

    def prop(self, name: str) -> Optional[ProbValue]:
        """获取属性名"""
        return Lict(self.properties).get(name)

    def set_prop(self, name: str, prob_class: ProbValue) -> None:
        """追加一个属性"""
        Lict(self.properties)[name] = prob_class

    def property_map(self) -> ProbPropertyMap:
        """ "获取属性字典"""
        return Lict(self.properties).to_dict()


DetObjects: TypeAlias = list[DetObject]
"""检测到的目标集合"""


def coord_trans_n(objects: DetObjects, dst_cs: Size, offset: Point = Point()) -> None:
    """检测到的目标集合坐标变换 & 归一化"""
    for ob in objects:
        ob.polygon = array_normalize(ob.polygon, dst_cs, offset)


def draw_objects(
    bgr: ImageNda, objects: DetObjects, thickness: int = 2, no_label: bool = False
) -> None:
    """绘制检测条目"""
    for o in objects:
        color = COLORS7[o.prob_class.value]
        label = ""
        if not no_label:
            label = str(int(o.prob_class.conf * 100))
            for name, pv in Lict(o.properties).items():
                if pv.value or pv.conf < 0.8:
                    fmt = " %s=%d" if isinstance(pv.value, int) else " %s=%.2f"
                    label += fmt % (name, pv.value)
                    # print(p.confidence)
                    if pv.conf < 0.8:
                        label += "(%d)" % int(pv.conf * 100)
        draw_box(bgr, o.rect(), color, label, thickness)


Strings: TypeAlias = list[str]


class DetRes(ABC):
    """检测器返回结果"""

    @abstractmethod
    def draw(
        self, image: DetOpt, colors: Colors, names: Optional[Strings] = None
    ) -> None:
        """绘制检测结果"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """检测器结果包含对象数量"""
        pass

    @abstractmethod
    def objects(self) -> DetObjects:
        """获取检测结果中的条目"""
        pass

    def min_prob(self) -> float:
        """检测器结果中的最小置信度, 用于筛选低置信度样本"""
        confs = [o.prob_class.conf for o in self.objects()]
        return min(confs)


Self = TypeVar("Self", bound="IDetector")


class IDetector(ABC):
    """目标检测器"""

    model_class = "detector"

    @classmethod
    def new(cls: Type[Self], info: ModelInfo, model_root: Path) -> Self:
        """创建检测器-根据信息"""
        file = model_root / info.file
        d = cls(file, info.opt, info.device)
        return d

    @abstractmethod
    def __init__(self, model_path: Path, opt: DetOpt, device_name: str):
        """创建分类器, 为 cls.new 提供模板"""
        self._model_path = model_path
        self._opt = opt
        self._device_name = device_name

    @abstractmethod
    def __call__(self, image: ImageNda) -> DetRes:
        """检测"""
        pass
