"""jxl.label.dataset 回迁测试 —— jxl label/io + label_set 行为保持（2026-09-21 迁出 ulabel，2026-11-05 回迁）.

用例语义拷自 jxl tests/label/io_test.py + label_set_test.py；
test_load_a2d_from_d2d_path 未迁（依赖 jxl.det.d2d 与已剥离的 A2dImageLabel.from_d2d，
检测域转换留守 jxl 侧）。
"""

from pathlib import Path

import pytest
from jcx.text.txt_json import save_json
from jvi.geo.rectangle import Rect
from rustshed import is_err, is_ok

from jxl.label.a2d import A2dImageLabel, A2dObjectLabel
from jxl.label.dataset import (
    A2dSampleSet,
    LabelFormat,
    MetaDataset,
    load_image_label_pairs,
    open_label_set,
)


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


def test_label_format() -> None:
    assert LabelFormat["HOP"] == LabelFormat.HOP

    assert LabelFormat.parse("hop") == LabelFormat.HOP

    assert LabelFormat.parse("HOP") == LabelFormat.HOP

    assert "HOP" in LabelFormat._member_map_


def test_open_label_set_a2d(tmp_path: Path) -> None:
    """工厂按 A2D 格式打开数据集."""
    _write_sample(tmp_path, "a2d", 0, "x")

    rs = open_label_set(tmp_path, LabelFormat.A2D, 0)

    assert is_ok(rs)
    label_set = rs.unwrap()
    assert isinstance(label_set, A2dSampleSet)
    assert label_set.format() == LabelFormat.A2D
    assert len(label_set) == 1


def test_open_label_set_unregistered_format(tmp_path: Path) -> None:
    """未注册格式(IMAGENET 未入 _class_map)返回 Err.

    （HopSet 已于 2026-09-21 按裁决移除——零存活消费者的兼容存留；读取能力由 hop IO 函数承载）
    DARKNET 由导入 jxl.label.formats 注册——两者不再可用作未注册样例.
    """
    _write_sample(tmp_path, "imagenet", 0, "y")

    rs = open_label_set(tmp_path, LabelFormat.IMAGENET, 0)

    assert is_err(rs)


def test_meta_dataset_import_sample(tmp_path: Path) -> None:
    """MetaDataset 按 {fmt}_m{id} 布局导入样本."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"")
    save_json(_make_label(), src / "a.json").unwrap()

    ds = MetaDataset(tmp_path / "ds", "a2d", 3)
    assert MetaDataset.meta_dir_name("a2d", 3) == "a2d_m3"
    assert not ds.valid()

    ds.import_sample(src / "a.jpg", src / "a.json")

    assert ds.valid()
    assert (tmp_path / "ds" / "image" / "a.jpg").is_file()
    assert (tmp_path / "ds" / "a2d_m3" / "a.json").is_file()


def test_hop_label_path_and_roundtrip(tmp_path: Path) -> None:
    from jxl.label.dataset import hop_label_path_of, hop_load_label, hop_save_label

    img = tmp_path / "image" / "a.jpg"
    img.parent.mkdir()
    img.write_bytes(b"x")
    p = hop_label_path_of(img, 5)
    assert p.parent.name == "hop_m5" and p.name == "a.json"

    label = A2dImageLabel(user_agent="t")
    assert hop_load_label(img, 5).is_null()
    hop_save_label(label, img, 5)
    got = hop_load_label(img, 5).unwrap()
    assert got.user_agent == "t"


def test_hop_load_label_corrupt_file_raises_not_null(tmp_path: Path) -> None:
    """损坏的 HOP 标注文件必须显式报错并点名文件，不得吞成「无标注」（Null）.

    与 hop_save_label 的显式传播裁决同源：corrupt ≠ missing，静默当空样本
    即训练数据静默丢失.
    """
    from jxl.label.dataset import hop_label_path_of, hop_load_label

    img = tmp_path / "image" / "a.jpg"
    img.parent.mkdir()
    img.write_bytes(b"x")
    label_file = hop_label_path_of(img, 5)
    label_file.parent.mkdir()
    label_file.write_text("{ not json !!!", encoding="utf-8")

    with pytest.raises(Exception, match=str(label_file)):
        hop_load_label(img, 5)


def test_label_filter(tmp_path: Path) -> None:
    from jxl.label.dataset import LabelFilter, hop_save_label

    img_dir = tmp_path / "image"
    img_dir.mkdir()
    img = img_dir / "a.jpg"
    img.write_bytes(b"x")
    assert LabelFilter.ALL.has_label(img, 7)
    assert not LabelFilter.LABELED.has_label(img, 7)
    hop_save_label(A2dImageLabel(user_agent="t"), img, 7)
    assert LabelFilter.LABELED.has_label(img, 7)
    assert (tmp_path / "hop_m7").is_dir()


def test_hop_save_label_propagates_io_error(tmp_path: Path) -> None:
    """save_json 结果不吞：目标父目录被同名文件占据时显式炸出。"""
    import pytest

    from jxl.label.a2d import A2dImageLabel
    from jxl.label.dataset import hop_save_label

    img_dir = tmp_path / "image"
    img_dir.mkdir()
    img = img_dir / "a.jpg"
    img.write_bytes(b"x")
    blocker = tmp_path / "hop_m3"
    blocker.write_text("占位文件，使标注目录无法写入")

    with pytest.raises(Exception, match="保存 HOP 标注失败"):
        hop_save_label(A2dImageLabel(user_agent="t"), img, 3)
