from dataclasses import dataclass
from pathlib import Path
from typing import List

from jcx.sys.fs import StrPath
from jvi.drawing.color import Color, LIME, GRAY
from jvi.drawing.shape import rectangle
from jvi.geo.point2d import Point
from jvi.geo.rectangle import Rect, Rects
from jvi.geo.size2d import Size
from jvi.image.image_nda import ImageNda
from jvi.image.proc import resize
from jxl.io.draw import draw_boxi
from jxl.label.hop import hop_save_label, load_label_records, LabelFilter, hop_load_label
from jxl.label.info import ObjectLabelInfo, ImageLabelInfo, ProbValue
from jxl.label.meta import PropMeta
from rustshed import Option, Null, Some


@dataclass
class TileObject:
    """目标瓦片图"""

    path: Path
    """图像路径"""
    obj: ObjectLabelInfo
    """目标标签"""
    root: ImageLabelInfo
    """目标所属的图片标注信息"""
    meta_id: int
    """传感器类型"""
    dst_rect: Rect = Rect()
    """目标所在绘图区域"""
    image: Option[ImageNda] = Null
    """目标图像"""

    def value_of(self, name: str) -> float:
        """获取对象属性值"""
        v = ProbValue(0, 0)
        return self.obj.properties.get(name, v).value

    def draw_on(self, canvas: ImageNda) -> None:
        """绘制瓦片"""
        if self.image.is_null():
            img = ImageNda.load(self.path)
            self.image = Some(img.roi(self.obj.rect()).clone())

        dst = canvas.roi(self.dst_rect)
        resize(self.image.unwrap(), dst)

    def draw_label(self, prop: str, canvas: ImageNda, active: bool, prop_meta: PropMeta) -> None:
        """绘制标注信息"""
        p = self.obj.prop(prop)
        value = prop_meta.value_meta(p.value)
        # rectangle(canvas, self.dst_rect, color, 4)
        label = "%s(%d%%)" % (value.name, int(100 * p.confidence))
        draw_boxi(canvas, self.dst_rect, Color.parse(value.color), label, 2)

        if active:
            rectangle(canvas, self.dst_rect.dilate(5), LIME, 2)

    def set_prop(self, name: str, value: int, conf: float = 2.0) -> None:
        """设置属性值"""
        self.obj.set_prop(name, value, conf)
        f = hop_save_label(self.root, self.path, self.meta_id)
        print('设置属性, 保存:', f)

    def exclude_prop_if(self, prop: str, conf_thr: float) -> None:
        """将属性设置为排除, 当该属性置信度超过阈值"""
        p = self.obj.properties.get(prop)
        if p and conf_thr < p.confidence <= 1:
            self.obj.properties[prop] = ProbValue.exclude()
            hop_save_label(self.root, self.path, self.meta_id)


TileObjects = List[TileObject]


@dataclass
class TileRecord:
    """平铺文件记录"""

    size: Size
    """图片大小"""
    objects: TileObjects
    """瓦片对象"""

    @classmethod
    def new(cls, size: Size, rects: Rects, objects: TileObjects) -> 'TileRecord':
        """创建对象"""
        i = 0
        for o in objects:
            o.dst_rect = rects[i]
            i += 1
        return TileRecord(size, objects)

    def load_image(self) -> ImageNda:
        """加载图片"""
        image = ImageNda(self.size, color=GRAY)
        for o in self.objects:
            o.draw_on(image)
        return image

    def image_file(self) -> Path:
        """获取图片路径"""
        return self.objects[0].path

    def draw_on(self, canvas: ImageNda, _pos: Point) -> None:
        """把记录绘制在画板上"""
        pass


def load_tiles(src_dir: StrPath, meta_id: int, category: int, prop: str, exclude_conf: float) -> TileObjects:
    """加载瓦片对象"""
    rs = load_label_records(src_dir, meta_id, LabelFilter.LABELED)
    assert len(rs) > 0
    print('加载图片:', len(rs))

    tiles = []
    for r in rs:
        label = hop_load_label(r.path, meta_id).unwrap()
        assert label
        for o in label.objects:
            if o.prob_class.value == category:
                t = TileObject(r.path, o, label, meta_id)
                t.exclude_prop_if(prop, exclude_conf)
                tiles.append(t)

    tiles.sort(key=lambda o1: o1.value_of(prop))
    return tiles
