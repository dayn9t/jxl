"""geometry 镜像测试 —— 断言 JSON 形状与 fixtures/vlabel_0001.json5 一致。"""

from jxl.label.geometry import Point, Polygon


def test_point_json_shape() -> None:
    p = Point(x=0.1, y=0.2)
    assert p.model_dump() == {"x": 0.1, "y": 0.2}


def test_polygon_is_point_array() -> None:
    # rs 侧 Polygon<f32> 序列化为 Point 数组（见 fixtures/vlabel_0001.json5 polygon 字段）
    poly = Polygon.from_points([Point(x=0.1, y=0.1), Point(x=0.2, y=0.1), Point(x=0.2, y=0.2)])
    dumped = poly.model_dump()
    assert dumped == [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.1}, {"x": 0.2, "y": 0.2}]


def test_polygon_empty_and_valid() -> None:
    assert Polygon.empty().is_valid() is False
    tri = Polygon.from_points([Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1)])
    assert tri.is_valid() is True
