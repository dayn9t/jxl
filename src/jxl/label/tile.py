from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from jcx.sys.fs import StrPath
from jvi.drawing.color import GRAY, LIME, Color
from jvi.drawing.shape import rectangle
from jvi.geo.point2d import Point
from jvi.geo.rectangle import Rect, Rects
from jvi.geo.size2d import Size
from jvi.image.image_nda import ImageNda
from jvi.image.proc import resize
from loguru import logger
from rustshed import Null, Option, Some

from jxl.io.draw import draw_boxi
from jxl.label.a2d.dd import A2dImageLabel, A2dObjectLabel
from jxl.label.hop import (
    LabelFilter,
    hop_load_label,
    hop_save_label,
    load_label_records,
)
from jxl.label.meta import PropMeta
from jxl.label.prop import CONF_EXCLUDE, PROP_EXCLUDE, ProbValue


@dataclass
class TileObject:
    """目标瓦片图"""

    path: Path
    """图像路径"""
    obj: A2dObjectLabel
    """目标标签"""
    root: A2dImageLabel
    """目标所属的图片标注信息"""
    meta_id: int
    """传感器类型"""
    dst_rect: Rect = field(default_factory=Rect)
    """目标所在绘图区域"""
    image: Option[ImageNda] = Null
    """目标图像"""

    def value_of(self, prop_id: int) -> int:
        """获取对象属性值"""
        return self.prop_of(prop_id).value

    def conf_of(self, prop_id: int) -> float:
        """获取对象属性值"""
        return self.prop_of(prop_id).conf

    def prop_of(self, prop_id: int) -> ProbValue:
        """获取对象属性值/置信度对"""
        return self.obj.prop(prop_id)

    def draw_on(self, canvas: ImageNda, cfg: PropMeta) -> None:
        """绘制瓦片"""
        if self.image.is_null():
            img = ImageNda.load(self.path)
            r = self.obj.rect()
            if cfg.border_extend:
                r = cfg.border_extend.extend(r)
            r = r.intersect(Rect.one())
            r = r.absolutize(img.size())

            self.image = Some(img.roi(r).clone())

        dst = canvas.roi(self.dst_rect)
        resize(self.image.unwrap(), dst)

    def draw_label(self, prop_meta: PropMeta, canvas: ImageNda, active: bool) -> None:
        """绘制标注信息"""
        p = self.obj.prop(prop_meta.id)
        value = prop_meta.value_meta(p.value)
        # rectangle(canvas, self.dst_rect, color, 4)
        label = f"{value.name}({int(100 * p.conf)}%)"
        draw_boxi(canvas, self.dst_rect, Color.parse(value.color), label, 2)

        if active:
            rectangle(canvas, self.dst_rect.dilate(5), LIME, 2)

    def set_prop(self, prop_id: int, value: int, conf: float = 2.0) -> None:
        """设置属性值"""
        self.obj.set_prop(prop_id, value, conf)
        self.root.user_agent = "jxl_prop"
        f = hop_save_label(self.root, self.path, self.meta_id)
        logger.info(f"设置属性, 保存: {f}")

    def exclude_prop_if(self, prop_id: int, conf_thr: float) -> None:
        """将属性设置为排除, 当该属性置信度超过阈值"""
        p = self.obj.prop(prop_id)
        if conf_thr < p.conf <= 1:
            self.obj.set_prop(prop_id, PROP_EXCLUDE, CONF_EXCLUDE)
            hop_save_label(self.root, self.path, self.meta_id)


TileObjects = list[TileObject]


@dataclass(frozen=True, slots=True)
class TileRecord:
    """平铺文件记录"""

    size: Size
    """图片大小"""
    objects: TileObjects
    """瓦片对象"""

    cfg: PropMeta

    @classmethod
    def new(cls, size: Size, rects: Rects, objects: TileObjects, cfg: PropMeta) -> Self:
        """创建对象"""
        for i, o in enumerate(objects):
            o.dst_rect = rects[i]
        return cls(size, objects, cfg)

    def get_image(self) -> ImageNda:
        """加载图片"""
        image = ImageNda(self.size, color=GRAY)
        for o in self.objects:
            o.draw_on(image, self.cfg)
        return image

    def image_file(self) -> Path:
        """获取图片路径"""
        return self.objects[0].path

    def draw_on(self, canvas: ImageNda, _pos: Point) -> None:
        """把记录绘制在画板上"""


def load_tiles(
    src_dir: StrPath,
    meta_id: int,
    category: int,
    prop_id: int,
    exclude_conf: float,
    min_prop: int,
    sort_by_conf: int,
) -> TileObjects:
    """加载瓦片对象"""
    rs = load_label_records(src_dir, meta_id, LabelFilter.LABELED)
    assert len(rs) > 0
    logger.info(f"加载图片: {len(rs)}")

    tiles: TileObjects = []
    for r in rs:
        label = hop_load_label(r.path, meta_id).unwrap()
        assert label
        for ob in label.objects:
            if ob.prob_class.value == category:
                t = TileObject(r.path, ob, label, meta_id)
                t.exclude_prop_if(prop_id, exclude_conf)
                v = t.value_of(prop_id)
                if v >= min_prop:
                    tiles.append(t)

    if sort_by_conf == 1:
        tiles.sort(key=lambda o1: o1.conf_of(prop_id))
    elif sort_by_conf == 2:
        tiles.sort(key=lambda o1: o1.conf_of(prop_id), reverse=True)
    else:
        tiles.sort(key=lambda o1: o1.value_of(prop_id))
    return tiles
