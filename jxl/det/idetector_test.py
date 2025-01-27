from jxl.det.idetector import *


def test_detected_object():
    ps = Rect.one().vertexes()
    o1 = DetObject.new(0, 0, Rect.one())
    o2 = DetObject.new(0, 0, polygon=ps)

    assert o1.rect() == o2.rect()
