"""meta 镜像测试 —— 以 fixtures/meta.json5（ulabel 仓 assets 拷贝，契约变更时同步）为序列化形状权威。"""

from pathlib import Path

import json5
import pytest

from jxl.label.meta import LabelMeta


def test_parse_assets_meta() -> None:
    fixture = Path(__file__).parent / "fixtures" / "meta.json5"
    meta = LabelMeta.model_validate(json5.loads(fixture.read_text()))
    assert meta.name == "FDOD"
    person = next(c for c in meta.categories if c.id == 0)
    assert person.names.zh == "人员"
    assert person.names.en == "person"
    role = person.properties[0]
    assert role.names.en == "role"
    gender_type = next(pt for pt in meta.property_types if pt.names.en == "gender")
    male = next(v for v in gender_type.values if v.names.en == "male")
    assert male.sign == "M"
    specials = meta.property_special_values
    assert [s.id for s in specials] == [-3, -2, -1]


def test_bilingual_roundtrip() -> None:
    raw = (Path(__file__).parent / "fixtures" / "meta.json5").read_text()
    meta = LabelMeta.model_validate(json5.loads(raw))
    assert LabelMeta.model_validate(meta.model_dump()) == meta


def test_border_extend_rejects_non_finite() -> None:
    """有限值契约：nan/inf 构造即拒（pydantic FiniteFloat，覆盖 meta 加载/调用全入口）。"""
    from pydantic import ValidationError

    from jxl.label.meta import BorderExtend

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError, match="finite"):
            BorderExtend(left=bad, right=0.0, top=0.0, bottom=0.0)
        with pytest.raises(ValidationError, match="finite"):
            BorderExtend(left=0.0, right=bad, top=0.0, bottom=0.0)
        with pytest.raises(ValidationError, match="finite"):
            BorderExtend(left=0.0, right=0.0, top=bad, bottom=0.0)
        with pytest.raises(ValidationError, match="finite"):
            BorderExtend(left=0.0, right=0.0, top=0.0, bottom=bad)


def test_border_extend_finite_values_accepted() -> None:
    from jxl.label.meta import BorderExtend

    be = BorderExtend(left=-0.25, right=0.0, top=0.1, bottom=0.4)
    assert (be.left, be.right, be.top, be.bottom) == (-0.25, 0.0, 0.1, 0.4)
