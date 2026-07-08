"""link_samples 纯函数 + 集成测试。"""

from pathlib import Path

from jxl.bin.link_samples import build_link_map


def test_build_link_map_basic(tmp_path: Path) -> None:
    ds = tmp_path / "datasets" / "COCO"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    (ds / "images" / "a.jpg").touch()
    (ds / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.2")
    cfg = {"name": "t", "datasets": ["COCO"], "split": [8, 1, 1]}
    pairs = build_link_map(cfg, tmp_path / "datasets")
    assert len(pairs) == 1
    src_img, src_lbl, prefix = pairs[0]
    assert prefix == "COCO"
    assert src_img.name == "a.jpg"
    assert src_lbl.name == "a.txt"


def test_build_link_map_multi_datasets(tmp_path: Path) -> None:
    for ds_name in ["COCO", "MOT17"]:
        ds = tmp_path / "datasets" / ds_name
        (ds / "images").mkdir(parents=True)
        (ds / "labels").mkdir(parents=True)
        (ds / "images" / "x.jpg").touch()
        (ds / "labels" / "x.txt").write_text("0 0.5 0.5 0.1 0.1")
    cfg = {"name": "t", "datasets": ["COCO", "MOT17"], "split": [8, 1, 1]}
    pairs = build_link_map(cfg, tmp_path / "datasets")
    assert len(pairs) == 2
    assert {p[2] for p in pairs} == {"COCO", "MOT17"}


def test_build_link_map_empty(tmp_path: Path) -> None:
    cfg = {"name": "t", "datasets": [], "split": [8, 1, 1]}
    assert build_link_map(cfg, tmp_path / "datasets") == []
