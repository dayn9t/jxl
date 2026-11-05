"""A2d 标注格式类型（jxl 现役）—— jxl 原生类型（2026-09-21 迁出、2026-11-05 回迁）。

A2d 是 vlabel Label 2.0（``jxl.label.label``）的前身格式，两格式并存不合并
（prob_class 嵌套 vs category+confidence 扁平、单 roi vs rois[]、version float vs str）。
2026-09-21 迁出时剥离项（保持剥离）：检测域（D2d）转换构造器（检测→标注职责归
``jxl.det.d2d_converters`` 自由函数）、GUI 绘制方法 draw_on（绘制职责归 jxl.io/jvi.gui）。
类型域声明：本模块全部为【标注域】类型（多边形 + 属性表），勿与检测域类型
（rect 矩形、无属性表）混淆。
Points 类型保留 jvi.geo 原引用（A2d 数据兼容：现役磁盘标注即用它）。
"""

from copy import deepcopy
from pathlib import Path
from typing import Final, Protocol, Self

from jcx.time.dt import now_iso_str
from jvi.geo.point2d import Point, Points
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size
from jvi.geo.trans import points_ncs_trans_in_win
from jvi.number import align_down
from loguru import logger
from pydantic import BaseModel, Field
from rustshed import Null, Option, Some

from jxl.label.meta_jxl import LabelMeta
from jxl.label.prop import ProbPropertyMap, ProbValue

ID_ERROR: Final[int] = -3
"""该值错误，需要改正"""
ID_EXCLUDE: Final[int] = -2
"""该值将被排除，不予考虑"""
ID_PENDING: Final[int] = -1
"""该值待定，需要设置"""

CAT_PENDING: Final[int] = -1
"""ROI对象默认类别"""

IMG_EXT: Final[str] = ".jpg"
"""图片文件扩展名"""
MSG_EXT: Final[str] = ".json"
"""传感器消息扩展名"""

ROI_FULL: Final[Points] = Rect.one().vertexes()
"""感兴趣区域-全图"""


class A2dObjectLabel(BaseModel):
    """2D分析目标标注信息"""

    id: int
    """目标ID"""
    prob_class: ProbValue
    """类别"""
    polygon: Points
    """包含目标的多边形区域"""
    properties: ProbPropertyMap = Field(default_factory=dict)
    """属性集合"""

    @classmethod
    def new(
        cls,
        id_: int,
        category: int,
        confidence: float,
        polygon: Points,
        properties: Option[ProbPropertyMap] = Null,
    ) -> Self:
        """创建标签信息"""
        assert polygon is not None
        return cls(
            id=id_,
            prob_class=ProbValue(category, confidence),
            polygon=polygon,
            properties=properties.unwrap_or({}),
        )

    def rect(self) -> Rect:
        """获取外包矩形, 外包矩形必须存在"""
        return Rect.bounding(self.polygon)

    def set_rect(self, r: Rect) -> None:
        """获取外包矩形, 外包矩形必须存在"""
        self.polygon = r.vertexes()

    def center(self) -> Point:
        """获取外包矩形的中心"""
        return self.rect().center()

    def polygon_to_rect(self) -> Rect:
        """ROI多边形区域化为矩形，并返回矩形"""
        self.polygon = self.rect().vertexes()
        return self.rect()

    def is_objective(self) -> bool:
        """判定是否为客观存在的目标"""
        return self.prob_class.value >= 0

    def prop(self, prop_id: int) -> ProbValue:
        """获取属性"""
        return self.properties.get(prop_id, ProbValue(0, 0))

    def set_prop(self, prop_id: int, value: int, conf: float = 2.0) -> None:
        """设置属性值"""
        self.properties[prop_id] = ProbValue(value, conf)

    def remove_prop(self, prop_id: int) -> None:
        """删除属性值"""
        if prop_id in self.properties:
            self.properties.pop(prop_id)

    def min_conf(self) -> float:
        """最小置信度"""
        if self.prob_class.value < 0:
            return 1.0
        p = min(
            self.properties.values(),
            default=self.prob_class,
            key=lambda x: x.conf,
        )
        return min(self.prob_class.conf, p.conf)

    def move(self, offset: Point) -> None:
        self.polygon = [p + offset for p in self.polygon]

    def clone(self) -> Self:
        return deepcopy(self)


A2dObjectLabels = list[A2dObjectLabel]
"""2D分析目标标注信息集"""


class A2dImageLabel(BaseModel):
    """2D分析图片标注信息"""

    version: float = 2.0
    """版本号，用于新旧格式转换"""
    user_agent: str = ""
    """用户代理信息"""
    date: str = now_iso_str()
    """标注时间"""
    last_modified: str = now_iso_str()
    """最后修改时间"""
    host: str = ""
    """主机"""
    sensor: int = 0
    """对象集合"""
    roi: Points = Field(default_factory=lambda: deepcopy(ROI_FULL))
    """感兴趣区域"""
    """数据来源传感器"""
    objects: A2dObjectLabels = Field(default_factory=list)

    @classmethod
    def new(cls, user_agent: str, objects: A2dObjectLabels) -> Self:
        """构造图片标注"""
        return cls(user_agent=user_agent, objects=objects)

    def roi_rect(self) -> Rect:
        """获取 ROI 外包矩形"""
        return Rect.bounding(self.roi)

    def objects_rect(self) -> Option[Rect]:
        """获取所有客观目标的外包矩形"""
        rect: Option[Rect] = Null
        for o in self.objects:
            if o.is_objective():
                match rect:
                    case Some(r):
                        rect = Some(r.unite(o.rect()))
                    case _:
                        rect = Some(o.rect())
        return rect

    def next_id(self) -> int:
        """获取下一个ID"""
        id_ = max([o.id for o in self.objects], default=0) + 1
        return max(id_, 1)

    def new_object(self, p: Point) -> A2dObjectLabel:
        """添加一个目标"""
        id_ = self.next_id()
        o = A2dObjectLabel.new(id_, 1, 2.0, [p])
        self.objects.append(o)
        return o

    def extend_objects(self, objs: A2dObjectLabels) -> None:
        """追加目标集"""
        id_ = self.next_id()
        for o in objs:
            o1 = deepcopy(o)
            o1.id = id_
            self.objects.append(o1)
            id_ += 1

    def clean(self, meta: LabelMeta) -> None:
        """清理无效数据"""
        objs = []
        for ob in self.objects:
            cat = meta.cat_meta(ob.prob_class.value)
            r = cat.check(ob)
            if r.is_err():
                logger.info(f"Meta验证失败，原因：{r}, 舍弃: {ob} ")
            elif ob.center().outside(self.roi):
                logger.info(f"中心超出ROI范围：舍弃: {ob} ")
            else:
                objs.append(ob)
        self.objects = objs

    def min_conf(self) -> float:
        """计算最小置信度, TODO: 分属性计算"""
        confs = map(lambda o: o.min_conf(), self.objects)
        return min(confs, default=1.0)

    def crop_by_roi(self, im_size: Size, extend_side: int = 4) -> tuple[Rect, Self]:
        """根据ROI裁切标注样本, 以期提高目标的分辨率"""
        rect = self.roi_rect()
        assert rect.is_normalized()
        rect = rect.absolutize(im_size)
        rect = rect.dilate(extend_side)
        rect = rect.intersect(Rect.from_size(im_size))
        rect.width = align_down(int(rect.width), 4)
        rect.height = align_down(int(rect.height), 4)

        rect = rect.normalize(im_size)

        label = self.clone()
        for ob in label.objects:
            ob.polygon = points_ncs_trans_in_win(ob.polygon, rect)

        return rect, label

    def clone(self) -> Self:
        return deepcopy(self)


A2dImageLabels = list[A2dImageLabel]
"""图像标注信息集"""

A2dImageLabelPair = tuple[Path, A2dImageLabel]
"""图像与标注信息对"""

A2dImageLabelPairs = list[A2dImageLabelPair]
"""图像与标注信息对集"""


class ToA2dObjectLabel(Protocol):
    """可转换为2D目标标注信息协议"""

    def to_label(self) -> A2dObjectLabel:
        """转换为标注信息"""


class ToA2dImageLabel(Protocol):
    """可转换为2D图片标注信息协议"""

    def to_label(self) -> A2dImageLabel:
        """转换为标注信息"""
