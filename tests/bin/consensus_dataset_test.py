"""consensus_dataset 单测: 源解析纯函数 + merge 端到端(合成三层小集)."""

from __future__ import annotations

from pathlib import Path

import orjson
import yaml
from PIL import Image
from typer.testing import CliRunner

from jxl.bin.consensus_dataset import (
    LabelKind,
    SourceSpec,
    app,
    boxes_from_xanylabel,
    merge,
)

_BOX_YOLO = "0 0.3 0.3 0.4 0.4\n"


def test_source_spec_from_str() -> None:
    """冒号 DSL 解析: 正常/缺分隔/含冒号路径(取最后分隔)均显式."""
    spec = SourceSpec.from_str("/a/b/labels:/c/imgs", LabelKind.YOLO)
    assert spec == (LabelKind.YOLO, Path("/a/b/labels"), Path("/c/imgs"))
    assert SourceSpec.from_str("/odd:name/labels:/imgs", LabelKind.YOLO).images_dir == Path(
        "/imgs"
    )
    for bad in ("no-colon", ":lead", "trail:"):
        try:
            SourceSpec.from_str(bad, LabelKind.YOLO)
            raise AssertionError(f"应拒绝: {bad}")
        except Exception as e:  # typer.BadParameter
            assert "格式" in str(e)


def test_boxes_from_xanylabel(tmp_path: Path) -> None:
    """polygon 顶点 → xyxy(乱序 min/max); rois 忽略; 空对象; 点数不足报错."""
    doc = {
        "version": "2.0",
        "rois": [[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]],
        "objects": [
            {
                "category": 0,
                "confidence": 0.9,
                "polygon": [
                    {"x": 0.633, "y": 0.425}, {"x": 0.35, "y": 0.12},
                    {"x": 0.35, "y": 0.425}, {"x": 0.633, "y": 0.12},
                ],
            }
        ],
    }
    f = tmp_path / "a.yaml"
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert boxes_from_xanylabel(f) == [(0.35, 0.12, 0.633, 0.425, 0.9)]
    doc["objects"] = []
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert boxes_from_xanylabel(f) == []
    doc["objects"] = [{"polygon": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}]}]
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    try:
        boxes_from_xanylabel(f)
        raise AssertionError("点数不足应抛 ValueError")
    except ValueError:
        pass


def _make_layers(tmp: Path) -> list[SourceSpec]:
    """合成三层: dump(2 帧, 1 行 L1 被滤) + yolo(1) + xanylabel(1)."""
    imgs = tmp / "images"
    imgs.mkdir()
    for i, stem in enumerate(("l0a", "l0b", "y1", "x1")):
        Image.new("RGB", (64, 64), (i * 40, 10, 10)).save(imgs / f"{stem}.jpg")
    dump = tmp / "validators.jsonl"
    rows = [
        {"stem": "l0a", "target": [[0.1, 0.1, 0.5, 0.5, 0.9]], "level": "L0"},
        {"stem": "l0b", "target": [], "level": "L1"},  # 非 L0 → 滤除
    ]
    dump.write_text("\n".join(orjson.dumps(r).decode() for r in rows), encoding="utf-8")
    labels = tmp / "yolo_labels"
    labels.mkdir()
    (labels / "y1.txt").write_text(_BOX_YOLO, encoding="utf-8")
    xdir = tmp / "xany"
    xdir.mkdir()
    (xdir / "x1.yaml").write_text(
        yaml.safe_dump({
            "objects": [
                {"category": 0, "polygon": [
                    {"x": 0.1, "y": 0.1}, {"x": 0.4, "y": 0.1},
                    {"x": 0.4, "y": 0.5}, {"x": 0.1, "y": 0.5},
                ]}
            ]
        }),
        encoding="utf-8",
    )
    return [
        SourceSpec(LabelKind.DUMP, dump, imgs),
        SourceSpec(LabelKind.YOLO, labels, imgs),
        SourceSpec(LabelKind.XANYLABEL, xdir, imgs),
    ]


def test_merge_three_layers(tmp_path: Path) -> None:
    """三层合并: 平铺 3 帧 symlink + 标注内容 + classes.txt + data.yaml."""
    out = tmp_path / "ds"
    stats = merge(_make_layers(tmp_path), out, ["person"], frozenset({"L0"}))
    assert [(s.kind, s.frames, s.boxes) for s in stats] == [
        (LabelKind.DUMP, 1, 1),
        (LabelKind.YOLO, 1, 1),
        (LabelKind.XANYLABEL, 1, 1),
    ]
    all_imgs = sorted((out / "all/images").glob("*.jpg"))
    assert [p.stem for p in all_imgs] == ["l0a", "x1", "y1"]
    assert all(p.is_symlink() for p in all_imgs)
    # dump 框 → YOLO 行; xanylabel 四点 → 中心/宽高; yolo 往返无损
    assert (out / "all/labels/l0a.txt").read_text(encoding="utf-8").strip() == (
        "0 0.300000 0.300000 0.400000 0.400000"
    )
    assert (out / "all/labels/x1.txt").read_text(encoding="utf-8").strip() == (
        "0 0.250000 0.300000 0.300000 0.400000"
    )
    assert (out / "all/labels/y1.txt").read_text(encoding="utf-8") == (
        "0 0.300000 0.300000 0.400000 0.400000"
    )
    assert (out / "classes.txt").read_text(encoding="utf-8") == "person\n"
    data = (out / "data.yaml").read_text(encoding="utf-8")
    assert data.startswith(f"path: {out.resolve()}")
    assert "train: train/images" in data and "  0: person" in data


def test_merge_rejects_conflict_and_empty(tmp_path: Path) -> None:
    """stem 跨层冲突 → ValueError(消息含两来源); 零源/零帧拒绝."""
    sources = _make_layers(tmp_path)
    (tmp_path / "yolo_labels/l0a.txt").write_text(_BOX_YOLO, encoding="utf-8")
    try:
        merge(sources, tmp_path / "ds2", ["person"], frozenset({"L0"}))
        raise AssertionError("冲突应抛 ValueError")
    except ValueError as e:
        assert "冲突" in str(e) and "dump" in str(e) and "yolo" in str(e)
    try:
        merge([], tmp_path / "ds3", ["person"], frozenset({"L0"}))
        raise AssertionError("零源应抛 ValueError")
    except ValueError:
        pass


def test_cli_smoke(tmp_path: Path) -> None:
    """CLI 薄壳: Sequence 多值选项 + 冲突时 exit 1 与错误消息."""
    imgs = tmp_path / "images"
    imgs.mkdir()
    Image.new("RGB", (64, 64)).save(imgs / "a.jpg")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text(_BOX_YOLO, encoding="utf-8")
    r = CliRunner().invoke(app, [
        str(tmp_path / "ds"), "--yolo", f"{labels}:{imgs}", "--classes", "person",
    ])
    assert r.exit_code == 0, r.output
    assert "合计 1 帧" in r.output
    r2 = CliRunner().invoke(app, [
        str(tmp_path / "ds4"), "--yolo", f"{labels}:{imgs}", "--yolo", f"{labels}:{imgs}",
    ])
    assert r2.exit_code == 1
    assert "冲突" in r2.output
