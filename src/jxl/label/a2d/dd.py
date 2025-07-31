from copy import deepcopy
from pathlib import Path
from typing import *

from jcx.m.number import align_down
from jcx.time.dt import now_iso_str
from jvi.drawing.color import Color, Colors
from jvi.drawing.shape import polylines
from jvi.geo.point2d import Points, Point
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size
from jvi.geo.trans import points_ncs_trans_in_win
from jvi.image.image_nda import ImageNda
from jvi.image.util import make_roi_surround_color

from jxl.det.d2d import D2dResult, D2dObject
from jxl.label.prop import ProbValue, ProbPropertyMap
from jxl.io.draw import draw_boxf
from jxl.label.meta import LabelMeta
from loguru import logger
from rustshed import Option, Some, Null
from pydantic import BaseModel, Field

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

    @classmethod
    def from_d2d(cls, ob: D2dObject) -> Self:
        """从D2dObject转换为A2dObjectLabel"""
        return cls.new(
            ob.id,
            ob.cls,
            ob.conf,
            ob.rect.vertexes(),
            properties={},
        )

    '''
    @classmethod
    def new_roi(cls, polygon: Points) -> Self:
        """创建感兴趣区域"""
        return cls.new(ID_ROI, CAT_ROI, 0, polygon)
'''

    def draw_on(self, bgr: ImageNda, colors: Colors, line_thickness: int = 1) -> None:
        """绘制标注信息在图上"""
        # color = colors[self.cat]
        # draw_boxf(bgr, self.rect, color, None, line_thickness)
        pass

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

    def prop(self, name: str) -> ProbValue:
        """获取属性"""
        return self.properties.get(name, ProbValue(0, 0))

    def set_prop(self, name: str, value: int, conf: float = 2.0) -> None:
        """设置属性值"""
        self.properties[name] = ProbValue(value, conf)

    def remove_prop(self, name: str) -> None:
        """删除属性值"""
        if name in self.properties:
            self.properties.pop("act")

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


A2dObjectLabels = List[A2dObjectLabel]
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
    def from_d2d(cls, d2d: D2dResult) -> "A2dImageLabel":
        """从D2dResult转换为A2dImageLabel"""
        label = A2dImageLabel()

        label.objects = [A2dObjectLabel.from_d2d(ob) for ob in d2d.objects]
        for i, ob in enumerate(label.objects):
            ob.id = i + 1
        return label

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

    def draw_on(
        self,
        canvas: ImageNda,
        cfg: LabelMeta,
        visible_props: List[str],
        show_conf: bool = True,
        cat_filter: int = -1,
    ) -> None:
        """绘制标注信息在图上"""
        # TODO: show_conf应该由cfg.label.title_style控制
        make_roi_surround_color(canvas, self.roi)

        for ob in self.objects:
            if cat_filter >= 0 and cat_filter != ob.prob_class.value:
                continue

            cat_cfg = cfg.cat_meta(ob.prob_class.value)
            assert cat_cfg
            color = Color.parse(cat_cfg.color)
            assert color
            label = cat_cfg.name + (ob.prob_class.conf_str() if show_conf else "")
            for prop_name, v in ob.properties.items():
                if "all" not in visible_props and prop_name not in visible_props:
                    continue
                p = cfg.prop_value_sign(ob.prob_class.value, prop_name, v.value)
                match p:
                    case Some(v_name):
                        if len(v_name) > 0:
                            label += " " + v_name + (v.conf_str() if show_conf else "")
                    case _:
                        print("WARN: 无效属性", ob.prob_class.value, prop_name, v.value)
                    # fmt = ' %s=%d' if type(v) == int else ' %s=%.2f'
                    # label += fmt % (k, v)
            if cfg.label.title_style == 0:
                label = ""
            polylines(canvas, ob.polygon, color, 1)
            draw_boxf(canvas, ob.rect(), color, label, cfg.label.thickness)

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

    def crop_by_roi(self, im_size: Size, extend_side: int = 4) -> Tuple[Rect, Self]:
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


A2dImageLabels = List[A2dImageLabel]
"""图像标注信息集"""

A2dImageLabelPair = Tuple[Path, A2dImageLabel]
"""图像与标注信息对"""

A2dImageLabelPairs = List[A2dImageLabelPair]
"""图像与标注信息对集"""


class ToA2dObjectLabel(Protocol):
    """可转换为2D目标标注信息协议"""

    def to_label(self) -> A2dObjectLabel:
        """转换为标注信息"""
        pass


class ToA2dImageLabel(Protocol):
    """可转换为2D图片标注信息协议"""

    def to_label(self) -> A2dImageLabel:
        """转换为标注信息"""
        pass
