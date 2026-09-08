"""link_samples 纯函数 + 集成测试。"""

import shutil
from pathlib import Path

from jxl.bin.link_samples import _relink, build_link_map, main


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


def test_relink_replaces_dangling_symlink(tmp_path: Path) -> None:
    """悬空 symlink(数据集迁移后旧链接)→ unlink 再链,不抛 FileExistsError。"""
    src = tmp_path / "new_root" / "img.jpg"
    src.parent.mkdir()
    src.write_text("new")
    dst = tmp_path / "out" / "img.jpg"
    dst.parent.mkdir()
    dst.symlink_to(tmp_path / "old_root" / "img.jpg")  # 悬空: old_root 不存在
    assert not dst.exists() and dst.is_symlink()
    _relink(dst, src)
    assert dst.is_symlink() and dst.read_text() == "new"


def test_relink_refreshes_valid_symlink(tmp_path: Path) -> None:
    """有效 symlink → 重链到当前 src(重跑刷新)。"""
    old_src = tmp_path / "old" / "img.jpg"
    new_src = tmp_path / "new" / "img.jpg"
    for p in (old_src, new_src):
        p.parent.mkdir()
        p.write_text(p.parent.name)
    dst = tmp_path / "out" / "img.jpg"
    dst.parent.mkdir()
    dst.symlink_to(old_src)
    _relink(dst, new_src)
    assert dst.read_text() == "new"


def test_relink_skips_real_file(tmp_path: Path) -> None:
    """真实文件占用 dst → 跳过不覆盖。"""
    src = tmp_path / "src" / "img.jpg"
    src.parent.mkdir()
    src.write_text("src")
    dst = tmp_path / "out" / "img.jpg"
    dst.parent.mkdir()
    dst.write_text("kept")
    _relink(dst, src)
    assert not dst.is_symlink() and dst.read_text() == "kept"


def test_main_rerun_after_dataset_migration(tmp_path: Path) -> None:
    """同一 out_dir 换 --datasets-root 重跑: 陈旧悬空链接被替换,不崩溃。"""
    for root in ("rootA", "rootB"):
        ds = tmp_path / root / "ds_a"
        (ds / "images").mkdir(parents=True)
        (ds / "labels").mkdir(parents=True)
        (ds / "images" / "img.jpg").write_text(root)
        (ds / "labels" / "img.txt").write_text("0 0.1 0.1 0.2 0.2")
    cfg = tmp_path / "exp.toml"
    cfg.write_text('name = "t"\ndatasets = ["ds_a"]\nsplit = [8, 1, 1]\n')
    out = tmp_path / "out"
    main(cfg, out, datasets_root=tmp_path / "rootA")
    shutil.rmtree(tmp_path / "rootA")  # 数据集整体迁移,旧位置删除
    main(cfg, out, datasets_root=tmp_path / "rootB")  # 修复前: FileExistsError
    img = out / "images" / "ds_a_img.jpg"
    assert img.is_symlink() and img.read_text() == "rootB"
