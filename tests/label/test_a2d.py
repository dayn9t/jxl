"""a2d/prop 回迁测试 —— jxl 现役格式行为保持。"""

import inspect

from jvi.geo.point2d import Point
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size

from jxl.label.a2d import A2dImageLabel, A2dObjectLabel
from jxl.label.meta_jxl import (
    CatMeta,
    FilterCfg,
    LabelCfg,
    LabelMeta,
    PropVar,
    SampleCfg,
)
from jxl.label.prop import (
    CONF_ERROR,
    CONF_EXCLUDE,
    PROP_ERROR,
    PROP_EXCLUDE,
    PROP_PENDING,
    ProbValue,
)


def test_prob_value_frozen() -> None:
    pv = ProbValue(value=1, conf=0.9)
    assert pv.value == 1 and pv.conf == 0.9


def test_sentinels() -> None:
    assert (PROP_ERROR, PROP_EXCLUDE, PROP_PENDING) == (-1, -2, -3)
    assert (CONF_ERROR, CONF_EXCLUDE) == (-2.0, -1.0)


def test_prob_value_semantics() -> None:
    excluded = ProbValue.exclude()
    assert excluded.is_excluded() and not excluded.is_normal()

    pv = ProbValue(1, 0.456)
    assert pv.conf_str() == "(45)"
    assert ProbValue(1, 0.99).conf_str() == ""
    assert pv.round(2).conf == 0.46


def test_a2d_object_label_has_no_detection_deps() -> None:
    # 类型域纪律：A2dImageLabel/A2dObjectLabel 不得引用检测域类型
    import jxl.label.a2d as m

    src = inspect.getsource(m)
    assert "from jxl.det" not in src
    assert "from_d2d" not in src


def test_object_label_props() -> None:
    ob = A2dObjectLabel(id=1, prob_class=ProbValue(1, 0.9), polygon=[Point(x=1, y=1)])
    ob.set_prop(0, 2, 0.5)
    assert ob.prop(0) == ProbValue(2, 0.5)
    assert ob.prop(9) == ProbValue(0, 0)
    assert ob.min_conf() == 0.5
    ob.remove_prop(0)
    assert ob.min_conf() == 0.9
    assert ob.is_objective()


def test_object_label_move_and_rect() -> None:
    ob = A2dObjectLabel.new(1, 1, 1.0, [Point(x=0, y=0), Point(x=2, y=2)])
    ob.move(Point(x=1, y=1))
    assert ob.polygon == [Point(x=1, y=1), Point(x=3, y=3)]
    r = ob.polygon_to_rect()
    assert r == ob.rect()
    assert ob.center() == Point(x=2, y=2)


def test_image_label_objects() -> None:
    label = A2dImageLabel.new("test", [])
    assert label.version == 2.0
    o1 = label.new_object(Point(x=1, y=1))
    o2 = label.new_object(Point(x=2, y=2))
    assert (o1.id, o2.id) == (1, 2)
    assert label.next_id() == 3

    objs = [A2dObjectLabel.new(7, 1, 1.0, [Point(x=0, y=0)])]
    label.extend_objects(objs)
    assert label.objects[-1].id == 3  # 追加目标重编号

    # 全部客观目标的外包矩形：覆盖 (0,0)-(2,2)
    assert label.objects_rect().unwrap() == Rect.from_ltrb(Point(x=0, y=0), Point(x=2, y=2))


def make_meta_with_filter(aspect_range: list[float]) -> LabelMeta:
    return LabelMeta(
        id=1,
        name="demo",
        description="演示",
        view_size=Size(width=1920, height=1080),
        object_size=Size(width=96, height=96),
        sample=SampleCfg(
            size=Size(width=96, height=96),
            background="BLACK",
            categories=["car"],
            properties=[],
        ),
        label=LabelCfg(title_style=1, thickness=1),
        auto_save=True,
        categories=[
            CatMeta(
                id=1,
                name="car",
                description="轿车",
                keys="c",
                color="RED",
                properties=[PropVar(name="color", type="ColorProp")],
                filter=FilterCfg(aspect_ratio=aspect_range),
            ),
        ],
        properties=[],
    )


def test_image_label_clean_drops_bad_objects() -> None:
    label = A2dImageLabel.new(
        "test",
        [
            # 中心在单位 ROI 内，但纵横比 1.0 不在 [2,3] → meta 校验失败舍弃
            A2dObjectLabel.new(1, 1, 1.0, [Point(x=0.1, y=0.1), Point(x=0.2, y=0.2)]),
            # 纵横比 2.5 在 [2,3] 内通过校验，但中心 (5.5,4) 超出单位 ROI → 舍弃
            A2dObjectLabel.new(2, 1, 1.0, [Point(x=3, y=3), Point(x=8, y=5)]),
        ],
    )
    label.clean(make_meta_with_filter([2.0, 3.0]))
    assert label.objects == []
