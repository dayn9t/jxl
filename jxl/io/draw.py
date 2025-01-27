import cv2  # type: ignore

from jiv.drawing.color import COLORS7, Color
from jiv.geo.rectangle import Rect
from jiv.image.image_nda import ImageNda
from jxl.common import ProbValue


def draw_boxi(image: ImageNda, rect: Rect, color: Color, label: str = '', thickness: int = 0) -> None:
    """绘制带标签的矩形框(整数坐标)"""
    # TODO: 移除cv2调用
    bgr = image.data()
    tl = thickness or round(0.002 * (bgr.shape[0] + bgr.shape[1]) / 2) + 1
    c1, c2 = (rect.x, rect.y), (rect.x + rect.width, rect.y + rect.height)
    cv2.rectangle(bgr, c1, c2, color.bgr(), thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
        cv2.rectangle(bgr, c1, c2, color.bgr(), -1, cv2.LINE_AA)  # filled
        color = color.inverse()
        # cv2.putText(bgr, label, (c1[0], c1[1] - 2), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(bgr, label, (c1[0], c1[1] - 2 + t_size[1] + 3), 0, tl / 3, color.bgr(), thickness=tf,
                    lineType=cv2.LINE_AA)


def draw_boxf(bgr: ImageNda, rect: Rect, color: Color, label: str = '', thickness: int = 0) -> None:
    """绘制带标签的矩形框(归一化坐标)"""
    rect = rect.absolutize(bgr.size())
    draw_boxi(bgr, rect, color, label, thickness)


def draw_box(bgr: ImageNda, rect: Rect, color: Color, label: str = '', thickness: int = 0) -> None:
    """绘制带标签的矩形框"""
    if rect.is_normalized():
        draw_boxf(bgr, rect, color, label, thickness)
    else:
        draw_boxi(bgr, rect, color, label, thickness)


def draw_class_item(bgr: ImageNda, item: ProbValue, thickness: int = 3) -> None:
    """绘制分类条目"""
    rect = Rect(0.1, 0.1, 0.8, 0.8)
    color = COLORS7[item.value]
    label = str(int(item.confidence * 100))
    draw_boxf(bgr, rect, color, label, thickness)
