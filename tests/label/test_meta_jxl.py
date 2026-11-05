"""meta_jxl 迁入测试 —— jxl 现役 meta 行为保持。

注：jxl 侧 meta_test.py 因磁盘 fixture 污染已 skip（pre-existing FIXME），
本测试改为程序化构造最小 meta，覆盖核心查找/按键/别名语义。
"""

import pytest
from jvi.geo.size2d import Size
from pydantic import ValidationError
from rustshed import is_null

from jxl.label.meta import BorderExtend as ContractBorderExtend
from jxl.label.meta_jxl import (
    BorderExtend,
    CatMeta,
    FilterCfg,
    LabelCfg,
    LabelMeta,
    PropMeta,
    PropVar,
    SampleCfg,
    ValueCfg,
    meta_fix,
)


def make_meta() -> LabelMeta:
    """构造最小 jxl 现役 meta"""
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
            properties=["ColorProp"],
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
            ),
        ],
        properties=[
            PropMeta(
                id=0,
                name="ColorProp",
                description="颜色",
                size=Size(width=32, height=32),
                values=[
                    ValueCfg(id=1, name="red", description="红", keys="r", color="RED", sign="R"),
                    ValueCfg(id=2, name="blue", description="蓝", keys="b", color="BLUE"),
                ],
            ),
        ],
    )


def test_meta_fix() -> None:
    assert meta_fix(2) == "m2"


def test_cat_meta_lookup() -> None:
    meta = make_meta()
    assert meta.cat_meta(id_=1).name == "car"
    assert meta.cat_meta(name="car").id == 1
    assert meta.cat_key_strs() == ["  [c] 轿车"]


def test_key_to_cat() -> None:
    meta = make_meta()
    assert meta.key_to_cat(ord("c")).unwrap() == 1
    assert is_null(meta.key_to_cat(0))


def test_prop_lookup_and_sign() -> None:
    meta = make_meta()
    p = meta.prop_meta_by_id(0, cat_id=1).unwrap()
    assert p.id == 0
    assert meta.prop_value_sign(1, 0, 1).unwrap() == "R"
    assert meta.prop_value_name(1, "color", 2) == "blue"


def test_filter_alias_parses_legacy_misspell() -> None:
    """历史误拼 aspect_radio 经 validation_alias 解析"""
    f = FilterCfg.model_validate({"aspect_radio": [0.5, 2.0]})
    assert f.aspect_ratio == [0.5, 2.0]


def test_border_extend_unified_with_contract() -> None:
    """meta_jxl 第二定义已删：BorderExtend 即 jxl.label.meta 契约版（同一类）。"""
    assert BorderExtend is ContractBorderExtend


def test_prop_border_extend_finite_contract() -> None:
    """有限值契约穿透 PropMeta：有限值接受、NaN 构造即拒。"""
    p = PropMeta(
        id=1,
        name="p",
        description="d",
        size=Size(width=32, height=32),
        border_extend=BorderExtend(left=0.1, right=-0.25, top=0.0, bottom=0.4),
    )
    assert p.border_extend is not None
    assert p.border_extend.left == 0.1

    with pytest.raises(ValidationError):
        BorderExtend(left=float("nan"), right=0.0, top=0.0, bottom=0.0)
