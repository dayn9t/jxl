from jiv.drawing.color import RED
from jiv.geo.size2d import SIZE_VGA
from jiv.image.trace import trace_image
from jml.io.draw import *


def show_draw_box():
    im = ImageNda(SIZE_VGA)

    r = Rect(0.25, 0.25, 0.5, 0.5)

    draw_box(im, r, RED, 'this a label')
    draw_class_item(im, ProbValue(2, 0.5))

    trace_image(im)


if __name__ == '__main__':
    show_draw_box()
