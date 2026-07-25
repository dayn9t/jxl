from jvi.geo.rectangle import Rect

from jxl.det.d2d import D2dObject


def test_d2d_object() -> None:
    """D2dObject 可实例化并序列化。"""
    ob1 = D2dObject(id=1, cls=0, conf=1, rect=Rect.one())
    print(ob1)
