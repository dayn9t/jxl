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
    labeled_boxes_from_xanylabel,
    merge,
    near_dup_pairs,
)

_BOX_YOLO = "0 0.3 0.3 0.4 0.4\n"
# 两框 (0.1,0.1,0.5,0.5) 与 (0.105,...) → IoU≈0.9515 ≥ NEAR_DUP_IOU(0.95), 触发守卫
_NEAR_YOLO = _BOX_YOLO + "0 0.305 0.305 0.4 0.4\n"
# 错位 0.01 → IoU≈0.906 < 0.95, 守卫放行
_FAR_YOLO = _BOX_YOLO + "0 0.31 0.31 0.4 0.4\n"


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


def test_labeled_boxes_from_xanylabel(tmp_path: Path) -> None:
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
    assert labeled_boxes_from_xanylabel(f) == [((0.35, 0.12, 0.633, 0.425, 0.9), 0)]
    doc["objects"] = []
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert labeled_boxes_from_xanylabel(f) == []
    doc["objects"] = [{"polygon": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}]}]
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    try:
        labeled_boxes_from_xanylabel(f)
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


def test_near_dup_pairs_pure() -> None:
    """近重复对纯函数: 同类高 IoU 命中; 异类/低于阈值不命中; 阈值可调."""
    box = (0.1, 0.1, 0.5, 0.5, 1.0)
    dup = (0.105, 0.105, 0.505, 0.505, 1.0)  # IoU≈0.9515
    far = (0.11, 0.11, 0.51, 0.51, 1.0)  # IoU≈0.906
    other_cls = (0.1, 0.1, 0.5, 0.5, 1.0)
    boxes: list[tuple[tuple[float, float, float, float, float], int]] = [
        (box, 0),
        (dup, 0),
        (far, 0),
        (other_cls, 1),
    ]
    pairs = near_dup_pairs(boxes)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 2)]
    assert pairs[0][2] >= 0.95
    # 同几何异类: 不判定(cls 不同)
    assert near_dup_pairs([(box, 0), (other_cls, 1)]) == []
    # 阈值下调: far 对也被捕获
    assert (0, 2) in [(i, j) for i, j, _ in near_dup_pairs(boxes, 0.9)]


def test_merge_rejects_intra_frame_near_dup(tmp_path: Path) -> None:
    """帧内近重复守卫: IoU≥0.95 → ValueError(报 stem+框对), 违规帧不落盘; 放行阀有效."""
    imgs = tmp_path / "images"
    imgs.mkdir()
    Image.new("RGB", (64, 64)).save(imgs / "a_ok.jpg")
    Image.new("RGB", (64, 64), (7, 7, 7)).save(imgs / "b_near.jpg")
    Image.new("RGB", (64, 64), (3, 3, 3)).save(imgs / "c_far.jpg")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a_ok.txt").write_text(_BOX_YOLO, encoding="utf-8")
    (labels / "b_near.txt").write_text(_NEAR_YOLO, encoding="utf-8")
    (labels / "c_far.txt").write_text(_FAR_YOLO, encoding="utf-8")  # 阈值下, 不触发
    sources = [SourceSpec(LabelKind.YOLO, labels, imgs)]
    out = tmp_path / "ds"
    try:
        merge(sources, out, ["person"], frozenset())
        raise AssertionError("帧内近重复应抛 ValueError")
    except ValueError as e:
        assert "b_near" in str(e) and "近重复" in str(e) and "IoU" in str(e)
    assert (out / "all/labels/a_ok.txt").exists()  # 违规帧之前的帧已落盘
    assert not (out / "all/labels/b_near.txt").exists()  # 先检后写: 违规帧不落盘
    assert not (out / "all/labels/c_far.txt").exists()  # 中止于违规帧
    # 逃生阀: allow_near_dup=True 放行, 两框均写出
    out2 = tmp_path / "ds_allow"
    stats = merge(sources, out2, ["person"], frozenset(), allow_near_dup=True)
    assert sum(s.boxes for s in stats) == 5
    near_lines = (out2 / "all/labels/b_near.txt").read_text(encoding="utf-8").strip()
    assert len(near_lines.splitlines()) == 2
    # 阈值下帧(IoU≈0.906)默认放行
    labels_far = tmp_path / "labels_far"
    labels_far.mkdir()
    Image.new("RGB", (64, 64), (5, 5, 5)).save(imgs / "c.jpg")
    (labels_far / "c.txt").write_text(_FAR_YOLO, encoding="utf-8")
    out3 = tmp_path / "ds_far"
    stats_far = merge(
        [SourceSpec(LabelKind.YOLO, labels_far, imgs)], out3, ["person"], frozenset()
    )
    assert stats_far[0].boxes == 2


def test_cli_allow_near_dup_flag(tmp_path: Path) -> None:
    """CLI 逃生阀: 默认拒绝(exit 1 + 近重复消息), --allow-near-dup 放行(exit 0)."""
    imgs = tmp_path / "images"
    imgs.mkdir()
    Image.new("RGB", (64, 64)).save(imgs / "a.jpg")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text(_NEAR_YOLO, encoding="utf-8")
    base = [str(tmp_path / "ds"), "--yolo", f"{labels}:{imgs}", "--classes", "person"]
    r = CliRunner().invoke(app, base)
    assert r.exit_code == 1
    assert "近重复" in r.output and "a" in r.output
    r2 = CliRunner().invoke(app, [*base, "--allow-near-dup"])
    assert r2.exit_code == 0, r2.output
    assert "合计 1 帧" in r2.output


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


def test_merge_rerun_overwrites(tmp_path: Path) -> None:
    """同 out_dir 重跑安全: 冲突中止留半写 all/, 修复后重跑(标准恢复路径)覆盖旧产物."""
    sources = _make_layers(tmp_path)
    conflict = tmp_path / "yolo_labels/l0a.txt"
    conflict.write_text(_BOX_YOLO, encoding="utf-8")
    out = tmp_path / "ds"
    try:
        merge(sources, out, ["person"], frozenset({"L0"}))
        raise AssertionError("冲突应抛 ValueError")
    except ValueError:
        pass
    assert (out / "all/labels/l0a.txt").exists()  # 半写状态(冲突帧前的帧已落盘)
    conflict.unlink()
    for _ in range(2):  # 恢复重跑 + 成功后重跑(全量 symlink 已存在)
        stats = merge(sources, out, ["person"], frozenset({"L0"}))
        assert sum(s.frames for s in stats) == 3
    assert all(p.is_symlink() for p in (out / "all/images").glob("*.jpg"))
    assert (out / "all/labels/x1.txt").read_text(encoding="utf-8").strip() == (
        "0 0.250000 0.300000 0.300000 0.400000"
    )


def test_multiclass_and_guards(tmp_path: Path) -> None:
    """多类: xanylabel category → cls, yolo cls 保留; 守卫: 路径校验/图 stem 重复."""
    imgs = tmp_path / "images"
    imgs.mkdir()
    Image.new("RGB", (64, 64)).save(imgs / "a.jpg")
    Image.new("RGB", (64, 64), (9, 9, 9)).save(imgs / "a.png")  # 同 stem 异扩展名
    try:
        merge([SourceSpec(LabelKind.YOLO, tmp_path, imgs)], tmp_path / "x", ["p"], frozenset())
        raise AssertionError("图 stem 重复应失败")
    except ValueError as e:
        assert "stem 重复" in str(e)
    imgs.joinpath("a.png").unlink()
    (tmp_path / "multi").mkdir()
    labels = tmp_path / "multi_labels"
    labels.mkdir()
    (labels / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")  # cls 1
    xdir = tmp_path / "multi_xany"
    xdir.mkdir()
    (xdir / "b.yaml").write_text(yaml.safe_dump({
        "objects": [{"category": 1, "polygon": [
            {"x": 0.1, "y": 0.1}, {"x": 0.3, "y": 0.1}, {"x": 0.3, "y": 0.3}, {"x": 0.1, "y": 0.3},
        ]}]
    }), encoding="utf-8")
    Image.new("RGB", (64, 64)).save(imgs / "b.jpg")
    out = tmp_path / "ds_multi"
    stats = merge(
        [SourceSpec(LabelKind.YOLO, labels, imgs), SourceSpec(LabelKind.XANYLABEL, xdir, imgs)],
        out, ["cat", "dog"], frozenset(),
    )
    assert sum(s.frames for s in stats) == 2
    assert (out / "all/labels/a.txt").read_text(encoding="utf-8") == "1 0.500000 0.500000 0.200000 0.200000"
    assert (out / "all/labels/b.txt").read_text(encoding="utf-8") == "1 0.200000 0.200000 0.200000 0.200000"
    # 路径校验: typo 目录 fail-fast
    try:
        merge([SourceSpec(LabelKind.YOLO, tmp_path / "nope", imgs)], tmp_path / "y", ["p"], frozenset())
        raise AssertionError("typo 路径应失败")
    except ValueError as e:
        assert "不存在" in str(e)


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


def test_merge_rerun_removed_layer_drops_stale_frames(tmp_path: Path) -> None:
    """撤层重跑: all/ 全清重建, 被撤层帧不残留(否则统计与文件不符, 旧帧流入训练)."""
    sources = _make_layers(tmp_path)
    out = tmp_path / "ds"
    merge(sources, out, ["person"], frozenset({"L0"}))
    assert sorted(p.stem for p in (out / "all/labels").glob("*.txt")) == ["l0a", "x1", "y1"]
    stats = merge([sources[0], sources[2]], out, ["person"], frozenset({"L0"}))  # 撤 yolo 层
    assert sum(s.frames for s in stats) == 2
    assert sorted(p.stem for p in (out / "all/labels").glob("*.txt")) == ["l0a", "x1"]
    assert sorted(p.stem for p in (out / "all/images").glob("*.jpg")) == ["l0a", "x1"]
