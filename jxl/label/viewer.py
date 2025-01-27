from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2  # type: ignore
from jcx.time.dt import iso_to_local
from jiv.drawing.color import LIME
from jiv.geo.point2d import Point
from jiv.gui.record_viewer import PImageEntry
from jiv.image.image_nda import ImageNda
from jxl.label.info import ImageLabelInfo
from jxl.label.meta import LabelMeta


@dataclass
class LabelRecord(PImageEntry):
    """文件记录"""

    meta: LabelMeta
    """元数据"""
    image: Path
    """图像路径"""
    info: ImageLabelInfo
    """图像标注信息"""

    def load_image(self) -> ImageNda:
        """加载图片"""
        image: ImageNda = ImageNda.load(self.image)
        return image

    def image_file(self) -> Path:
        """获取图片路径"""
        return self.image

    def draw_on(self, canvas: ImageNda, _pos: Point) -> None:
        """把记录绘制在画板上"""
        self.info.draw_on(canvas, self.meta)
        time = iso_to_local(self.info.last_modified)
        label = f'modified={time}  image={self.image}'
        color = LIME
        cv2.putText(canvas.data(), label, (8, 16), 0, 0.5, color.bgr(), thickness=1, lineType=cv2.LINE_AA)


LabelRecords: TypeAlias = list[LabelRecord]
"""标注记录列表"""

"""
def show_main():
    from dateutil.parser import parse
    path = '/var/ias/snapshot/shtm/n1/work'
    meta = find_meta(31, path).unwrap()

    folder = Path("/home/jiang/ws/mine/sorting/darknet")
    print('darknet_dir:', folder)

    labels = darknet_load_labels(folder)
    assert len(labels) > 0

    rs = [LabelRecord(meta, label) for label in labels]

    win = RecordViewer('files://' + str(folder), meta.view_size)
    win.set_records(rs)

    print('image number:', win.image_count())
    win.run()


if __name__ == '__main__':
    show_main()
"""
