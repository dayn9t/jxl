from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
from jcx.time.dt import iso_to_local
from jvi.drawing.color import LIME
from jvi.geo.point2d import Point
from jvi.gui.record_viewer import PImageEntry
from jvi.image.image_nda import ImageNda
from jxl.label.a2d.dd import A2dImageLabel
from jxl.label.meta import LabelMeta


@dataclass
class LabelRecord(PImageEntry):
    """文件记录"""

    meta: LabelMeta
    """元数据"""
    image: Path
    """图像路径"""
    info: A2dImageLabel
    """图像标注信息"""

    def get_image(self) -> ImageNda:
        """加载图片"""
        image: ImageNda = ImageNda.load(self.image)
        return image

    def image_file(self) -> Path:
        """获取图片路径"""
        return self.image

    def draw_on(self, canvas: ImageNda, _pos: Point) -> None:
        """把记录绘制在画板上"""
        self.info.draw_on(canvas, self.meta, ["all"])
        time = iso_to_local(self.info.last_modified)
        label = f"modified={time}  image={self.image}"
        color = LIME
        cv2.putText(
            canvas.data(),
            label,
            (8, 16),
            0,
            0.5,
            color.bgr(),
            thickness=1,
            lineType=cv2.LINE_AA,
        )


LabelRecords: TypeAlias = list[LabelRecord]
"""标注记录列表"""
