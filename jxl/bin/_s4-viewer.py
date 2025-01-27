#!/opt/ias/env/bin/python

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Final

from jcx.sys.fs import files_in
from jcx.time.dt import now_iso_str
from jcx.ui.key import Key, Flag
from jvi.drawing.color import WHITE, YELLOW_GREEN, LIME
from jvi.drawing.shape import polylines, cross, put_text
from jvi.geo.point2d import Point, Points, closest_point
from jvi.geo.polygon import Polygon
from jvi.gui.record_viewer import RecordViewer
from jvi.image.image_nda import ImageNda
from jxl.label.hop import (
    hop_load_label,
    import_label,
    hop_save_label,
    hop_del_label,
    load_label_records,
    LabelFilter,
)
from jxl.label.info import ImageLabelInfo, ObjectLabelInfo, ObjectLabelInfos
from jxl.label.meta import LabelMeta, find_meta
from rustshed import Option, Null, Some

NEAR_R2: Final[float] = 0.05**2 / 4  # TODO:
"""点的邻域半径平方"""


class Labeler(RecordViewer):

    def __init__(
        self, meta: LabelMeta, meta_id: int, snapshot_dir: Path, verbose: bool
    ):
        super().__init__("labeler", meta.view_size)
        self.meta_id = meta_id
        self.snapshot_dir = snapshot_dir
        self.verbose = verbose

        self.label_meta = meta
        self.cur_label: ImageLabelInfo = ImageLabelInfo.only_roi(
            "jxl_label"
        )  # 当前图片标注信息
        self.cur_object: Option[ObjectLabelInfo] = Null  # 当前标注对象
        self.cur_vertex: Option[Point] = Null  # 当前标注定点
        self.cur_category = 0
        self.show_all_category: bool = True
        self.locked_roi: Option[Points] = Null  # 锁定的ROI
        self.pre_objects: Option[ObjectLabelInfos] = Null  # 前一对象集
        self.saved = True  # 是否已经保存修改
        self.labeled = False  # 是否已经标注
        self.help_msgs = [
            "鼠标操作：",
            "  [L] 存在邻近当前顶点则删除，否则添加对象/顶点，",
            "  [Shift+L] 选择对象，选中/取消当前顶点",
            "  [Ctrl+L] 删除邻近顶点的两个邻近顶点",
            "方向按键表：",
            "  [a] 左移/前一个",
            "  [d] 右移/后一个",
            "  [w] 上移/前十个",
            "  [s] 下移/后十个",
            "系统按键表：",
            "  [ESC] 退出程序",
            "  [F1] 显示当前信息",
            "  [SPACE] 保存标注信息",
            "  [BACKSPACE] 删除标注信息",
            "  [x] 放弃保存标注信息",
            "  [j] 抓图",
            "ROI按键表：",
            "  [e] 删除ROI",
            "  [r] 目标区域变为矩形",
            "  [f] ROI锁定/解锁",
            "  [z] 目标区域充满ROI",
            "  [c] 复制以前一对象集",
            "对象类别按键表：",
        ]
        self.help_msgs.extend(meta.cat_key_strs())

        self.help_msgs.extend(
            [
                "鼠标功能表：",
                "  [单击] 因情况不同：",
                "  - 选中最近顶点",
                "  - 选中顶点所属区域",
                "  - 取消选中顶点/区域",
                "  - 删除顶点",
            ]
        )

    def on_change_image(self, index: int) -> None:
        """处理图片切换事件"""

        f = Path(self.cur_image_file())

        cur_label = hop_load_label(f, self.meta_id)
        self.labeled = True
        if cur_label.is_null():
            cur_label = import_label(f, self.meta_id)
            self.labeled = False
        self.cur_label = cur_label.unwrap_or(ImageLabelInfo.only_roi("jxl_label"))
        print("  #%d" % index, f, len(self.cur_label.objects) - 1)

        ocr_root = Path("/home/jiang/ws/trash/dates/demo/2024-10-21/samples")
        ocr_dir = ocr_root / f.stem
        files = files_in(ocr_dir, ".txt")
        for file in files:
            # 读取文件内容并打印
            with open(file, "r") as fp:
                print("  -", fp.read())

        if self.locked_roi.is_some():
            roi = self.cur_label.roi().unwrap()
            roi.clear()
            roi.extend(deepcopy(self.locked_roi.unwrap()))
            print("locked_roi:", self.locked_roi)

        self.cur_vertex = Null
        self.cur_object = Null
        self.saved = True

    def on_key(self, key: int) -> int:
        """按键响应"""
        if self._try_move(key):
            return 0
        if self._try_set_category(key):
            return 0
        if key == ord("`"):
            self._switch_category()
        if key == Key.BLANK:
            self._save_label()
        elif key == Key.BACKSPACE:
            self._del_label()
        elif key == ord("x"):
            self.cur_vertex = Null
            self.cur_object = Null
            self.saved = True
        elif key == ord("e"):
            self._del_object()
        elif key == ord("r"):
            self._polygon_to_rect()
        elif key == ord("f"):
            self._lock_roi()
        elif key == ord("c"):
            self._copy_objects()
        elif key == ord("z"):
            self._polygon_to_roi()
        elif key == ord("j"):
            self._take_snapshot()
        else:
            return super().on_key(key)
        return 0

    def _lock_roi(self) -> None:
        if self.locked_roi.is_some():
            self.locked_roi = Null
            print("ROI锁定解除")
        else:
            self.locked_roi = Some(deepcopy(self.cur_label.roi().unwrap()))
            print("ROI锁定，后继图片都使用ROI")

    def _copy_objects(self) -> None:
        match self.pre_objects:
            case Some(objs):
                self.cur_label.extend_objects(objs)
                print("复制了前一目标集(ROI会被丢弃)：", len(objs))
            case _:
                print("前一目标集不存在")

    def _del_object(self) -> None:
        """当前目标移除"""
        match self.cur_object:
            case Some(cur):
                self.cur_label.objects.remove(cur)
                self.cur_vertex = Null
                self.cur_object = Null
                self.saved = False
                print("当前目标移除")
            case _:
                print("当前目标不存在")

    def _polygon_to_rect(self) -> None:
        """目标区域变为矩形"""
        match self.cur_object:
            case Some(cur):
                r = cur.polygon_to_rect()
                self.cur_vertex = Null
                self.saved = False
                print("目标区域变为矩形：", r.absolutize(self.size()))
            case _:
                print("当前目标不存在")

    def _polygon_to_roi(self) -> None:
        """当前目标区域填满整个ROI"""
        match self.cur_object:
            case Some(cur):
                r = self.cur_label.roi().unwrap()
                cur.polygon = deepcopy(r)
                self.cur_vertex = Null
                self.saved = False
                print("目标区域充满ROI")
            case _:
                print("当前目标不存在")

    def _save_label(self) -> None:
        lab = self.cur_label
        lab.last_modified = now_iso_str()
        lab.user_agent = "jxl_label"
        lab.clean(self.label_meta)
        print("save_label", self.cur_image_file(), self.meta_id)
        hop_save_label(lab, self.cur_image_file(), self.meta_id)
        self.cur_vertex = Null
        self.cur_object = Null
        self.saved = True
        self.labeled = True
        self.pre_objects = Some(lab.objects)

    def _del_label(self) -> None:
        hop_del_label(self.cur_image_file(), self.meta_id)

    def _move_cur(self, dx: int, dy: int) -> None:
        """移动当前节点/ROI"""
        match self.cur_object:
            case Some(ob):
                d = Point(dx, dy).normalize(self.size())
                # print('对象移动:', d)
                ob.move(d)
                self.saved = False
            case _:
                pass

    def _try_move(self, key: int) -> bool:
        """尝试移动"""
        moved = True
        if key == ord("a"):
            self._move_cur(-1, 0)
        elif key == ord("d"):
            self._move_cur(1, 0)
        elif key == ord("w"):
            self._move_cur(0, -1)
        elif key == ord("s"):
            self._move_cur(0, 1)
        else:
            moved = False
        return moved and not self.saved

    def _try_set_category(self, key: int) -> bool:
        """尝试设置目标类别"""
        r = self.label_meta.key_to_cat(key)
        if r.is_null():
            return False
        category = r.unwrap()
        match self.cur_object:
            case Some(ob):
                ob.prob_class.value = category
                print("类别修改: %d %d" % (ob.id, category))
                self.saved = False
            case _:
                print("当前目标不存在")
        self.cur_category = category
        return True

    def _switch_category(self) -> None:
        self.show_all_category = not self.show_all_category
        if self.show_all_category:
            print("显示所有类别")
        else:
            print(f"显示类别: {self.cur_category}")

    def _take_snapshot(self) -> None:
        """抓图"""
        self.snapshot_dir.mkdir(exist_ok=True)
        src = self.cur_image_file()
        # dst = self.snapshot_dir / now_file('.jpg')
        # dst.symlink_to(src)
        dst = self.snapshot_dir / Path(src).name
        self.canvas().save(dst)
        print("抓图:", dst)

    def on_draw(self, canvas: ImageNda, _pos: Point) -> None:
        if self.show_all_category:
            cat_filter = -1
        else:
            cat_filter = self.cur_category
        self.cur_label.draw_on(canvas, self.label_meta, cat_filter=cat_filter)
        text = ""
        if self.labeled:
            text += "Labeled   "
        if self.locked_roi.is_some():
            text += "[f]UnlockROI   "
        else:
            text += "[f]LockROI   "
        if not self.saved:
            text += "[ ]Save [x]Cancel   "
        match self.cur_object:
            case Some(ob):
                polylines(canvas, ob.polygon, WHITE)
                if self.cur_vertex.is_some():
                    cross(canvas, self.cur_vertex.unwrap(), 7, YELLOW_GREEN, 3)
                text += "[e]RemoveObject [r]RectifyPolygon   "
        # put_text(canvas, text, Point(8, 32), LIME, 2)

    def _closest_object(self, cursor: Point) -> Option[ObjectLabelInfo]:
        """获取光标邻域内最接近的对象"""
        d2_min = NEAR_R2
        obj: Option[ObjectLabelInfo] = Null
        for o in self.cur_label.objects:
            d2, p = closest_point(cursor, o.polygon)
            if d2 < d2_min:
                d2_min = d2
                obj = Some(o)
        return obj

    def _closest_vertex(self, cursor: Point) -> Option[Point]:
        """获取光标邻域内最接近的顶点"""
        match self.cur_object:
            case Some(o):
                d2, p = closest_point(cursor, o.polygon)
                if d2 < NEAR_R2:
                    return Some(p)
            case _:
                pass
        return Null

    def on_left_button_down(self, cursor: Point, flags: int) -> None:
        """顶点/区域选择，节点区域左键保存选中区域内样本"""
        # print('cursor:', cursor, flags)
        cursor = cursor.normalize(self.size())
        if flags & Flag.SHIFT:  # 选择一个对象
            obj = self._closest_object(cursor)
            self.cur_object = obj if obj != self.cur_object else Null
            print("切换对象选择")
        elif self.cur_object.is_some():
            vertex = self._closest_vertex(cursor)
            match vertex:
                case Some(v):
                    if self.cur_vertex == vertex:
                        self._remove_vertex(v)
                        print("删除当前顶点")
                    else:
                        self.cur_vertex = vertex
                        print("选择顶点:", v.absolutize(self.size()))
                case _:
                    self._insert_vertex(cursor)
                    print("插入顶点")
        else:
            self._new_object(cursor)  # 创建一个新对象

    def _new_object(self, cursor: Point) -> None:
        """创建新目标"""
        o = self.cur_label.new_object(cursor)
        o.prob_class.value = self.cur_category
        self.cur_object = Some(o)
        self.cur_vertex = Some(cursor)
        self.saved = False
        print("新建目标：", o.id, cursor.absolutize(self.size()))

    def _remove_vertex(self, vertex: Point) -> None:
        """删除节点"""
        polygon = self.cur_object.unwrap().polygon
        if len(polygon) < 2:
            print("不能删除最后一个顶点")
        else:
            i = polygon.index(vertex)
            polygon.remove(vertex)
            self.cur_vertex = Some(polygon[i % len(polygon)])
            self.saved = False
            print("删除顶点:", vertex.absolutize(self.size()))

    def _insert_vertex(self, cursor: Point) -> None:
        """插入节点"""
        polygon = self.cur_object.unwrap().polygon
        Polygon(polygon).insert_best(cursor)
        self.cur_vertex = Some(cursor)
        self.saved = False
        print("插入顶点:", cursor.absolutize(self.size()))


# params: /var/ias/snapshot/shtm/n1/work 2021-04-11


def main() -> None:
    parser = argparse.ArgumentParser(description="目标检测标注")
    parser.add_argument("folder", type=Path, help="样本目录")
    parser.add_argument("meta_id", type=int, help="元数据ID")
    parser.add_argument(
        "-l",
        "--label_filter",
        type=int,
        default=2,
        help="样本标注过滤：1-所有样本，2-可导入标注样本，3-已标注样本",
    )
    parser.add_argument(
        "-c",
        "--conf_thr",
        type=float,
        default=1.0,
        help="置信度阈值, 标注错误置信度为-2",
    )
    parser.add_argument(
        "-p", "--pattern", type=str, help="文件要匹配的模式，用于过滤数据"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    opt = parser.parse_args()

    meta = find_meta(opt.meta_id, opt.folder).unwrap()

    rs = load_label_records(
        opt.folder,
        opt.meta_id,
        LabelFilter(opt.label_filter),
        opt.pattern,
        opt.conf_thr,
    )
    assert len(rs) > 0, "不存在标注记录"
    print("加载样本总数:", len(rs))

    snapshot_dir = Path(opt.folder, "snapshot")
    labeler = Labeler(meta, opt.meta_id, snapshot_dir, opt.verbose)
    labeler.set_records(rs)
    labeler.run()


if __name__ == "__main__":
    # catch_show_err(main, verbose=True)
    main()
