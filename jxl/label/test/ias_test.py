from jml.label.ias import *


def image_path_of_label_test():
    p = Path('/2021-04-22/12-17-08.551_s31.lbl')
    p1 = Path('/2021-04-22/12-17-08.551.jpg')
    p2 = ias_image_path_of_label(p, '.jpg')
    assert p1 == p2
    p3 = ias_image_path_of_label('a', '.jpg')
    assert p3 is None
