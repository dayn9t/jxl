"""jxl.label.io 单元测试."""

from pathlib import Path

import pytest
from jcx.text.txt_json import save_json
from jvi.geo.rectangle import Rect

from jxl.det.d2d import D2dObject, D2dResult
from jxl.label.a2d.dd import A2dImageLabel, A2dObjectLabel
from jxl.label.io import load_image_label_pairs


def _make_label(category: int = 0) -> A2dImageLabel:
    """构造带单个目标的测试标注."""
    obj = A2dObjectLabel.new(
        id_=1,
        category=category,
        confidence=0.9,
        polygon=Rect(x=0.1, y=0.2, width=0.5, height=0.6).vertexes(),
    )
    return A2dImageLabel(user_agent="test", objects=[obj])


def _write_sample(root: Path, fmt: str, meta_id: int, name: str) -> None:
    """按 MetaDataset 约定写入一个样本(图像占位 + 标注 json)."""
    image_dir = root / "image"
    label_dir = root / f"{fmt}_m{meta_id}"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    save_json(_make_label(), label_dir / f"{name}.json").unwrap()
    (image_dir / f"{name}.jpg").write_bytes(b"")


def test_load_a2d_pairs(tmp_path: Path) -> None:
    """读取 a2d_m{meta_id}: img_label.py 自动标注产物."""
    _write_sample(tmp_path, "a2d", 0, "x")

    pairs = load_image_label_pairs(tmp_path, 0, "a2d")

    assert len(pairs) == 1
    image_file, label = pairs[0]
    assert image_file == tmp_path / "image" / "x.jpg"
    assert isinstance(label, A2dImageLabel)
    assert len(label.objects) == 1
    assert label.objects[0].prob_class.value == 0


def test_load_hop_pairs(tmp_path: Path) -> None:
    """读取 hop_m{meta_id}: 与 a2d 同构, 仅前缀不同."""
    _write_sample(tmp_path, "hop", 0, "y")

    pairs = load_image_label_pairs(tmp_path, 0, "hop")

    assert len(pairs) == 1
    assert pairs[0][0].name == "y.jpg"


def test_missing_label_dir_raises(tmp_path: Path) -> None:
    """标注目录不存在时立即报错(No Silent Degradation)."""
    _write_sample(tmp_path, "a2d", 0, "x")

    with pytest.raises(FileNotFoundError):
        load_image_label_pairs(tmp_path, 0, "hop")


def test_load_a2d_from_d2d_path(tmp_path: Path) -> None:
    """img_label.py 真实路径: D2dResult -> A2dImageLabel.from_d2d -> 存盘 -> 读回."""
    image_dir = tmp_path / "image"
    label_dir = tmp_path / "a2d_m0"
    image_dir.mkdir()
    label_dir.mkdir()

    d2d = D2dResult(
        objects=[
            D2dObject(
                id=1, cls=0, conf=0.9, rect=Rect(x=0.1, y=0.2, width=0.5, height=0.6)
            )
        ]
    )
    label = A2dImageLabel.from_d2d(d2d)
    save_json(label, label_dir / "x.json").unwrap()
    (image_dir / "x.jpg").write_bytes(b"")

    pairs = load_image_label_pairs(tmp_path, 0, "a2d")

    assert len(pairs) == 1
    got = pairs[0][1]
    assert isinstance(got, A2dImageLabel)
    assert len(got.objects) == 1
    assert got.objects[0].prob_class.value == 0
