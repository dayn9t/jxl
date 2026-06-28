from jvi.drawing.color import RED
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import SIZE_VGA
from jvi.image.image_nda import ImageNda
from jvi.image.trace import trace_image

from jxl.io.draw import draw_box, draw_class_item
from jxl.label.prop import ProbValue


def show_draw_box() -> None:
    im = ImageNda(SIZE_VGA)

    r = Rect.new(0.25, 0.25, 0.5, 0.5)

    draw_box(im, r, RED, "this a label")
    draw_class_item(im, ProbValue(2, 0.5))

    trace_image(im)


if __name__ == "__main__":
    show_draw_box()
