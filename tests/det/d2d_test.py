from jxl.det.d2d import D2dObject, D2dObjectTrack
from jvi.geo.rectangle import Rect


def test_d2d_object():

    # Test if the class can be instantiated
    ob1 = D2dObject(cls=0, conf=1, rect=Rect.one())

    print(ob1)

    ob2 = D2dObjectTrack(**ob1.model_dump(), id=1)

    print(ob2)
    # Test if the class has the expected attributes
    assert hasattr(ob2, "id1")
