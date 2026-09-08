"""frame_sample 测试: stride 下界校验(防误删全量) + 采样删除行为。"""

from pathlib import Path

from typer.testing import CliRunner

from jxl.bin.frame_sample import app


def _make_ds(root: Path, frames: list[int], seq: str = "SEQ_01") -> None:
    """构造 {seq}_{frame}.jpg + 对应 label 的数据集."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for f in frames:
        (root / "images" / f"{seq}_{f:06d}.jpg").touch()
        (root / "labels" / f"{seq}_{f:06d}.txt").write_text("0 0.5 0.5 0.1 0.1")


def test_stride_zero_rejected(tmp_path: Path) -> None:
    """--stride 0 必须在入口被拒, 不得触碰任何文件."""
    _make_ds(tmp_path, [1, 2, 3])
    result = CliRunner().invoke(app, [str(tmp_path), "--stride", "0"])
    assert result.exit_code != 0
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 3


def test_stride_negative_rejected(tmp_path: Path) -> None:
    """--stride 负数曾致 keep 集为空、全量删除; 必须入口拒绝."""
    _make_ds(tmp_path, [1, 2, 3])
    result = CliRunner().invoke(app, [str(tmp_path), "--stride", "-1"])
    assert result.exit_code != 0
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 3
    assert len(list((tmp_path / "labels").glob("*.txt"))) == 3


def test_stride_sampling_deletes_images_and_labels(tmp_path: Path) -> None:
    _make_ds(tmp_path, [1, 2, 3, 4, 5])
    result = CliRunner().invoke(app, [str(tmp_path), "--stride", "2"])
    assert result.exit_code == 0
    kept = sorted(p.name for p in (tmp_path / "images").glob("*.jpg"))
    assert kept == ["SEQ_01_000001.jpg", "SEQ_01_000003.jpg", "SEQ_01_000005.jpg"]
    assert sorted(p.name for p in (tmp_path / "labels").glob("*.txt")) == [
        "SEQ_01_000001.txt",
        "SEQ_01_000003.txt",
        "SEQ_01_000005.txt",
    ]


def test_dry_run_keeps_all(tmp_path: Path) -> None:
    _make_ds(tmp_path, [1, 2, 3, 4])
    result = CliRunner().invoke(app, [str(tmp_path), "--stride", "2", "--dry-run"])
    assert result.exit_code == 0
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 4
