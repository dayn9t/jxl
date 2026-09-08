"""yolo_coreset 测试: 含点文件名代表存活(防全删) + embedding 外/非图片文件保留."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from jxl.bin.yolo_coreset import app


def _make_ds(root: Path, names: list[str]) -> None:
    """构造 images/{name} + labels/{stem}.txt 数据集."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for n in names:
        (root / "images" / n).touch()
        (root / "labels" / f"{Path(n).stem}.txt").write_text("0 0.5 0.5 0.1 0.1")


def _make_emb(npy: Path, names: list[str]) -> None:
    """embedding .npy + .txt(与 person_embed 输出对齐: basename 列表)."""
    rng = np.random.default_rng(0)
    np.save(npy, rng.random((len(names), 8), dtype=np.float32))
    npy.with_suffix(".txt").write_text("\n".join(names), encoding="utf-8")


def test_dotted_names_rep_survives(tmp_path: Path) -> None:
    """含点文件名(frame.0001.jpg): 代表必须存活 —— 曾 split('.')[0] 与 stem 失配致全删(-100%)."""
    names = ["frame.0001.jpg", "frame.0002.jpg"]
    _make_ds(tmp_path, names)
    emb = tmp_path / "emb.npy"
    _make_emb(emb, names)
    result = CliRunner().invoke(app, [str(emb), str(tmp_path), "--target", "1"])
    assert result.exit_code == 0
    imgs = sorted(p.name for p in (tmp_path / "images").glob("*"))
    lbls = sorted(p.name for p in (tmp_path / "labels").glob("*"))
    assert len(imgs) == 1 and imgs[0] in names
    assert lbls == [f"{Path(imgs[0]).stem}.txt"]


def test_out_of_scope_files_preserved(tmp_path: Path) -> None:
    """embedding 外新图 / 非图片文件 / 子目录 / classes.txt 保留, 只删集合内非代表."""
    names = ["a.jpg", "b.jpg"]
    _make_ds(tmp_path, names)
    (tmp_path / "images" / "new.jpg").touch()  # embedding 落后于数据集的新图
    (tmp_path / "images" / "notes.md").write_text("x")  # 非图片文件
    (tmp_path / "images" / "sub").mkdir()  # 子目录(曾 IsADirectoryError)
    (tmp_path / "labels" / "classes.txt").write_text("person")
    emb = tmp_path / "emb.npy"
    _make_emb(emb, names)
    result = CliRunner().invoke(app, [str(emb), str(tmp_path), "--target", "1"])
    assert result.exit_code == 0
    kept_imgs = {p.name for p in (tmp_path / "images").glob("*")}
    assert "new.jpg" in kept_imgs
    assert "notes.md" in kept_imgs
    assert (tmp_path / "images" / "sub").is_dir()
    kept_lbls = {p.name for p in (tmp_path / "labels").glob("*")}
    assert "classes.txt" in kept_lbls
    assert len(kept_lbls) == 2  # classes.txt + 1 个代表 label
    assert len(kept_imgs & set(names)) == 1  # 集合内非代表被删
