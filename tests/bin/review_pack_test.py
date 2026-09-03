"""review_pack 单测: tile 渲染纯函数(合成图) + CLI 端到端(tmp 目录)."""
from __future__ import annotations

from pathlib import Path

import orjson
from PIL import Image
from typer.testing import CliRunner

from jxl.bin.review_pack import MODEL_COLORS, app, render_tile


def test_model_colors_complete() -> None:
    assert set(MODEL_COLORS) == {"target", "yoloe", "gdino", "rfdetr", "la"}
    assert MODEL_COLORS["target"] == (0, 0, 0)
    assert MODEL_COLORS["yoloe"] == (60, 120, 255)
    assert MODEL_COLORS["gdino"] == (255, 200, 0)
    assert MODEL_COLORS["rfdetr"] == (0, 200, 0)
    assert MODEL_COLORS["la"] == (255, 40, 40)


def test_render_tile_sizes() -> None:
    im = Image.new("RGB", (320, 240), (0, 0, 0))
    out = render_tile(im, {"la": [(0.1, 0.1, 0.5, 0.5, 1.0)]}, "stem_a")
    assert out.width == 640  # tile 统一宽
    assert out.height > 240  # 含 header


def test_render_tile_marks_model_boxes() -> None:
    # 320x240 → 640x480, 框 (64,48,320,240); header 高 26 → 框左上角在 (64, 74)
    im = Image.new("RGB", (320, 240), (0, 0, 0))
    out = render_tile(im, {"la": [(0.1, 0.1, 0.5, 0.5, 1.0)]}, "stem_a")
    assert out.getpixel((64, 74)) == MODEL_COLORS["la"]  # 框边缘 = la 色
    assert out.getpixel((300, 200)) == (0, 0, 0)  # 框内部未填充


def _make_case(tmp_path: Path) -> tuple[Path, Path, str]:
    """造 consensus/review/manifest.jsonl(3 条, 1 条缺图) + 2 张合成帧图."""
    images = tmp_path / "frames"
    images.mkdir()
    for stem in ("a", "b"):
        Image.new("RGB", (320, 240), (10, 10, 10)).save(images / f"{stem}.jpg")
    consensus = tmp_path / "consensus"
    (consensus / "review").mkdir(parents=True)
    box = [0.1, 0.1, 0.5, 0.5, 1.0]
    rows = [
        {
            "image": "a.jpg",
            "score": 0.5,
            "target_boxes": [box],
            "validators": {"la": [box], "yoloe": []},
            "breakdown": {"fp_count": 1, "fn_count": 0},
        },
        {
            "image": "b.jpg",
            "score": 0.25,
            "target_boxes": [],
            "validators": {"rfdetr": [box]},
            "breakdown": {"fp_count": 0, "fn_count": 1},
        },
        {
            "image": "gone.jpg",
            "score": 0.75,
            "target_boxes": [],
            "validators": {},
            "breakdown": {"fp_count": 0, "fn_count": 0},
        },
    ]
    text = "\n".join(orjson.dumps(row).decode() for row in rows) + "\n"
    (consensus / "review" / "manifest.jsonl").write_text(text, encoding="utf-8")
    return consensus, images, text


def test_cli_end_to_end(tmp_path: Path) -> None:
    consensus, images, text = _make_case(tmp_path)
    out = tmp_path / "pack"
    r = CliRunner().invoke(app, [str(consensus), str(images), str(out), "--per-grid", "2"])
    assert r.exit_code == 0, r.output
    # 2 条可渲染, per-grid=2 → 恰 1 张网格(4 列 × tile 640)
    assert (out / "review_grid_001.jpg").exists()
    assert not (out / "review_grid_002.jpg").exists()
    assert Image.open(out / "review_grid_001.jpg").width == 2560
    # manifest 原样合并 + README 指引 + 缺图记录(不静默)
    assert (out / "manifest.jsonl").read_text(encoding="utf-8") == text
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "labels" in readme and "target" in readme
    missing = (out / "_missing.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(missing) == 1 and "gone.jpg" in missing[0]


def test_cli_empty_manifest_errors(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus"
    (consensus / "review").mkdir(parents=True)  # 无 manifest → 空输入
    r = CliRunner().invoke(app, [str(consensus), str(tmp_path / "imgs"), str(tmp_path / "pack")])
    assert r.exit_code == 1
