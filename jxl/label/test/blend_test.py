from random import choices, shuffle

from jiv.drawing.color import random_color
from jiv.drawing.shape import rectangle
from jiv.image.image_nda import ImageNda
from jiv.image.trace import trace_image
from jxl.label.blend import *


def show_random_object_pos():
    size = Size(512, 512)
    im = ImageNda(size)
    for _i in range(10):
        r = random_object_pos(size, Size(100, 100))
        rectangle(im, r, random_color())

    trace_image(im)


def demo_blender():
    src_dir = '/home/jiang/ws/scene/square/1'
    dst_dir = '/home/jiang/ws/scene/dst1'
    ob_dir = '/home/jiang/ws/scene/objects/'

    blender = ObjectBlender(2)

    n = blender.load_objects(ob_dir)
    print('load objects:', n)
    blender.make_samples(src_dir, dst_dir)


if __name__ == "__main__":
    demo_blender()
    # show_random_object_pos()
