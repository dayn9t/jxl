#!/home/jiang/py/jxl/.venv/bin/python

import argparse
from pathlib import Path

from jcx.ui.key import Key
from jvi.geo.point2d import Point
from jvi.geo.rectangle import Rect
from jvi.gui.record_viewer import RecordViewer
from jvi.image.image_nda import ImageNda
from jxl.label.meta import LabelMeta, find_meta
from jxl.label.tile import TileRecord, TileObject, load_tiles, TileObjects


class PropLabeler(RecordViewer):

    def __init__(self, meta: LabelMeta, cat_name: str, prop_name: str):
        super().__init__("labeler-%s-%s" % (cat_name, prop_name), meta.view_size)
        self.meta = meta
        self.cat_name = cat_name
        self.prop_name = prop_name
        self.prop_meta = meta.prop_meta(prop_name, cat_name=cat_name).unwrap()
        self.tile_index = 0  # 当前瓦片(标注对象)索引
        self.help_msgs = [
            "方向按键表：",
            "  [a] 左移/前一个",
            "  [d] 右移/后一个",
            "  [w] 上移/前十个",
            "  [s] 下移/后十个",
            "系统按键表：",
            "  [ESC] 退出程序",
            "  [F1] 显示当前信息",
            "对象类别按键表：",
        ]
        self.help_msgs.extend(meta.cat_key_strs())

    def on_change_image(self, index: int) -> None:
        """处理图片切换事件"""

        f = Path(self.cur_image_file())  # TODO: 不反应所有文件
        print("#%d" % index, f)
        self.tile_index = 0

    def on_key(self, key: int) -> int:
        """按键响应"""
        if key < 0:
            return 0
        if self._try_set_value(key):
            return 0
        if key == Key.BLANK:
            pass
            # self._save_label()
        else:
            return super().on_key(key)
        return 0

    def cur_object(self) -> TileObject:
        """当前选中对象"""
        return self.record().objects[self.tile_index]

    def _try_set_value(self, key: int) -> bool:
        """尝试设置目标属性值"""
        r = self.prop_meta.value_cfg_of_key(key)
        if r.is_null():
            return False
        value_cfg = r.unwrap()
        obj = self.cur_object()
        obj.set_prop(self.prop_name, value_cfg.id, value_cfg.conf)
        print(f"类别属性: {self.prop_name}={value_cfg}")
        self.tile_index += 1
        self.tile_index %= len(self.record().objects)
        return True

    def on_draw(self, canvas: ImageNda, _pos: Point) -> None:
        rec: TileRecord = self.record()
        for i, o in enumerate(rec.objects):
            o.draw_label(self.prop_name, canvas, i == self.tile_index, self.prop_meta)

    def on_left_button_down(self, cursor: Point, flags: int) -> None:
        """顶点/区域选择，节点区域左键保存选中区域内样本"""
        rec: TileRecord = self.record()
        for i, o in enumerate(rec.objects):
            if o.dst_rect.contains(cursor):
                self.tile_index = i
                return

    def load(self, rs: TileObjects) -> None:
        """加载数据"""
        view_size = self.meta.view_size
        tiles = Rect.from_size(view_size).to_tiles(
            size=self.meta.object_size, need_round=True
        )
        rects = [r.erode(8) for r in tiles]
        n = len(rects)  # 单页可以显示的区域
        rs1 = [
            TileRecord.new(view_size, rects, rs[i : i + n])
            for i in range(0, len(rs), n)
        ]
        self.set_records(rs1)


# args: /var/ias/snapshot/shtm/n1/work 2021-04-11 can amount


def main() -> None:
    parser = argparse.ArgumentParser(description="目标属性标注")
    parser.add_argument("folder", type=Path, help="图片目录")
    parser.add_argument("meta_id", type=int, help="元数据ID")
    parser.add_argument("category", type=str, help="目标类别")
    parser.add_argument("property", type=str, help="目标属性")
    parser.add_argument(
        "-e", "--exclude_conf", type=float, default=1.0, help="排除置信度大于该值的对象"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    opt = parser.parse_args()

    meta = find_meta(opt.meta_id, opt.folder).unwrap()

    cat = meta.cat_meta(name=opt.category).id
    rs = load_tiles(opt.folder, opt.meta_id, cat, opt.property, opt.exclude_conf)
    assert len(rs) > 0
    print("加载对象:", len(rs))

    labeler = PropLabeler(meta, opt.category, opt.property)
    labeler.load(rs)
    labeler.run()


if __name__ == "__main__":
    # catch_show_err(main)
    main()
