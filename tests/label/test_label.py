"""label 镜像测试 —— 含 note 可选性、tz-aware 时间戳与 rs 序列化形状对齐。"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from jxl.label.geometry import Point, Polygon
from jxl.label.label import Label, Object, PropertyEntry

TZ = timezone(timedelta(hours=8))
"""受控时区（与 assets fixture 的 +08:00 一致）"""


def _obj(note: str | None = None) -> Object:
    return Object(
        id=1,
        category=0,
        confidence=1.0,
        polygon=Polygon.from_points([Point(x=0.1, y=0.1), Point(x=0.5, y=0.1), Point(x=0.5, y=0.5)]),
        note=note,
        properties=[PropertyEntry(id=0, value=1, confidence=0.9, note="侧脸误判" if note else None)],
    )


def test_note_none_dump_omits_key() -> None:
    dumped = _obj(None).model_dump(exclude_none=True)
    assert "note" not in dumped
    assert "note" not in dumped["properties"][0]


def test_note_some_roundtrip() -> None:
    obj = _obj("被门框遮挡")
    assert obj.model_dump(exclude_none=True)["note"] == "被门框遮挡"
    assert Object.model_validate(obj.model_dump(exclude_none=True)) == obj


def test_old_file_without_note_parses() -> None:
    # 旧文件无 note 字段（rs 侧 serde(default) 语义）—— py 侧默认 None
    raw = {
        "id": 1,
        "category": 0,
        "confidence": 1.0,
        "polygon": [{"x": 0.0, "y": 0.0}],
    }
    obj = Object.model_validate(raw)
    assert obj.note is None
    assert obj.properties == []


def _label(note: str | None = None) -> Label:
    return Label(
        version="2.0",
        user_agent="test",
        created_at=datetime(2024, 1, 1, tzinfo=TZ),
        last_modified=datetime(2024, 1, 1, tzinfo=TZ),
        note=note,
        objects=[_obj()],
    )


def test_label_note_none_dump_omits_key() -> None:
    dumped = _label(None).model_dump(exclude_none=True)
    assert "note" not in dumped


def test_label_note_some_roundtrip() -> None:
    label = _label("整图背光，多边形边界保守")
    assert label.model_dump(exclude_none=True)["note"] == "整图背光，多边形边界保守"
    assert Label.model_validate(label.model_dump(exclude_none=True)) == label


def test_label_old_file_without_note_parses() -> None:
    # 旧文件无 note 字段（rs 侧 serde(default) 语义）—— py 侧默认 None
    raw = {
        "version": "2.0",
        "user_agent": "test",
        "created_at": "2024-01-01T00:00:00+08:00",
        "last_modified": "2024-01-01T00:00:00+08:00",
        "objects": [],
    }
    label = Label.model_validate(raw)
    assert label.note is None


def test_label_naive_timestamp_rejected() -> None:
    # 构造即拒 naive：rs 真源 chrono DateTime<Local> 的 serde 只接受带
    # offset 的 RFC3339（POST verdicts 与 GUI 同一解析器，naive 必拒）——
    # 镜像侧接受域精确对齐，非法状态不可表示。
    raw = {
        "version": "2.0",
        "user_agent": "test",
        "created_at": "2024-01-01T00:00:00",
        "last_modified": "2024-01-01T00:00:00+08:00",
        "objects": [],
    }
    with pytest.raises(ValidationError, match="tz-aware"):
        Label.model_validate(raw)
    with pytest.raises(ValidationError, match="tz-aware"):
        Label(
            version="2.0",
            user_agent="test",
            created_at=datetime(2024, 1, 1),  # naive datetime 实例
            last_modified=datetime(2024, 1, 1, tzinfo=TZ),
            objects=[],
        )


def test_label_tz_aware_roundtrip_keeps_offset() -> None:
    label = _label()
    dumped = label.model_dump(mode="json", exclude_none=True)
    assert dumped["created_at"] == "2024-01-01T00:00:00+08:00"
    assert Label.model_validate(dumped) == label


def test_label_parse_assets_fixture() -> None:
    from pathlib import Path

    import json5

    fixture = Path(__file__).parent / "fixtures" / "vlabel_0001.json5"
    label = Label.model_validate(json5.loads(fixture.read_text()))
    assert label.version == "2.0"
    assert len(label.objects) == 4
    assert isinstance(label.created_at, datetime)
    assert label.note is None  # 现行 fixture 无文件级 note（可选字段）
