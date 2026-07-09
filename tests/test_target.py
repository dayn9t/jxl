from pathlib import Path

import pytest

from jxl.target import TargetProfile, load_target


def test_target_profile_optional_rfdetr() -> None:
    p = TargetProfile(
        name="head",
        yolo_text="head",
        rfdetr_cls_id=None,
        vlm_prompt="x",
        weights="/w.pt",
        output_cls_id=0,
    )
    assert p.rfdetr_cls_id is None


def test_load_target_builtin_person() -> None:
    p = load_target("person")  # targets/person.toml
    assert p.name == "person"
    assert p.yolo_text == "person"
    assert p.rfdetr_cls_id == 0


def test_load_target_explicit_path() -> None:
    p = load_target("x", profile_path=Path("targets/person.toml"))
    assert p.name == "person"


def test_load_target_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_target("no_such_target_xyz")
