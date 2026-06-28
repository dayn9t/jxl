import pytest
from jxl.det.d2d import D2dObject
from jvi.geo.rectangle import Rect


# FIXME(pre-existing, jvi-migration): D2dObjectTrack 类已不存在(jxl/jvi 全仓 grep 为空),
# 且断言 hasattr(ob2, "id1") 引用已废弃字段。待 PR-3 跟随 jvi pydantic 迁移修复或删除。
@pytest.mark.skip(reason="pre-existing: D2dObjectTrack removed in jvi migration")
def test_d2d_object():
    # Test if the class can be instantiated
    ob1 = D2dObject(id=0, cls=0, conf=1, rect=Rect.one())

    print(ob1)  # noqa: T201

    ob2 = D2dObjectTrack(**ob1.model_dump(), id=1)  # noqa: F821

    print(ob2)  # noqa: T201
    # Test if the class has the expected attributes
    assert hasattr(ob2, "id1")
