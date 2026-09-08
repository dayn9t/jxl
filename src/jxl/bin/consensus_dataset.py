#!/usr/bin/env python3
"""Consensus-Dataset: 共识标注管线产物层 → darknet 平铺集 + classes.txt + data.yaml.

把标注管线的分层产物合并为可直接训练的数据集(2026-09-06 n001 dataset_v2
流程的工具化——当时为内联脚本):

标注源(LabelKind)三格式, 统一展开为 (stem, 带类框, 图片), cls 全链保留:
- `--dump <jsonl>:<images_dir>`: det_mine --dump-validators 产物按 level 过滤.
  领域约定: **L0 = 确认样本而非丢弃**(五模型全一致, 标注 = target 框;
  难例挖掘视角的"丢弃"是旧账本误记, 2026-09-06 修正); dump 无类别 → cls 恒 0
- `--yolo <labels_dir>:<images_dir>`: 现成 YOLO labels(det_mine batch /
  doubao·vlm confirmed / 人工 YOLO 终版), 可重复; cls 保留原值(多类可用),
  坐标经解析统一 6 位小数格式(数值等价)
- `--xanylabel <yaml_dir>:<images_dir>`: X-AnyLabeling 标注(version 2.0,
  objects[].polygon 归一化顶点 → xyxy, category → cls; rois 为监测区不属标注)

产出 `<out>/all/{images(symlink), labels}` + `classes.txt` + `data.yaml`
(data.yaml 的 train/val/test 为相对路径, 与划分比例无关, 划分前生成零信息缺失).
链路: 本工具(合并+配置) → jxl_split(纯划分) → yolo_train(训练), 三工具三职责.

用法:
    consensus_dataset <out_dir> \
        --dump consensus/validators_all.jsonl:images --dump-level L0 \
        --yolo consensus/batch_0000/labels:consensus/batch_0000/images ... \
        --yolo arbitration/confirmed/labels:images \
        --xanylabel hardcase_labels:images --classes person
"""

import shutil
from collections.abc import Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NamedTuple, Self

import orjson
import typer
import yaml

from jxl.det.hardmine import Box

app = typer.Typer(add_completion=False, help="共识管线产物层 → 平铺集 + data.yaml")

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class LabelKind(StrEnum):
    """标注源格式."""

    DUMP = "dump"
    YOLO = "yolo"
    XANYLABEL = "xanylabel"


type LabeledBox = tuple[Box, int]
"""带类框: (归一化 xyxy 框, cls id)."""

type StemBoxes = tuple[str, list[LabeledBox], Path]
"""单帧标注: (stem, 带类框列表, 图片路径)."""


class SourceSpec(NamedTuple):
    """一个标注源: 标注路径 + 配对图片目录(冒号 DSL 显式解析)."""

    kind: LabelKind
    labels_path: Path
    images_dir: Path

    @classmethod
    def from_str(cls, spec: str, kind: LabelKind) -> Self:
        """`<标注路径>:<images_dir>` → SourceSpec(路径含冒号时取最后分隔)."""
        left, sep, right = spec.rpartition(":")
        if not sep or not left or not right:
            raise typer.BadParameter(f"层须为 <标注路径>:<images_dir> 格式: {spec}")
        return cls(kind, Path(left), Path(right))


class LayerStats(NamedTuple):
    """单层合并统计."""

    kind: LabelKind
    frames: int
    boxes: int


def img_by_stem(images_dir: Path) -> dict[str, Path]:
    """图片目录递归索引 {stem: path}; 同 stem 多图(不同扩展名)即失败(防配错图)."""
    out: dict[str, Path] = {}
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        if p.stem in out:
            raise ValueError(f"图片目录 stem 重复: {p.stem} ({out[p.stem]} 与 {p})")
        out[p.stem] = p
    return out


def parse_labeled_yolo(text: str, name: str) -> list[LabeledBox]:
    """YOLO label 文本(cls cx cy w h) → 带类框(cls 保留, conf 不可恢复恒 1.0)."""
    out: list[LabeledBox] = []
    for ln in text.splitlines():
        parts = ln.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{name}: 非 5 字段行: {ln[:50]}")
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        out.append(((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 1.0), cls))
    return out


def labeled_to_text(boxes: Sequence[LabeledBox]) -> str:
    """带类框 → YOLO label 文本(与 to_yolo_label 同 6 位小数格式, cls per 框)."""
    lines = []
    for (x1, y1, x2, y2, _conf), cls in boxes:
        lines.append(
            f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
            f"{x2 - x1:.6f} {y2 - y1:.6f}"
        )
    return "\n".join(lines)


def iter_dump(spec: SourceSpec, levels: frozenset[str]) -> Iterator[StemBoxes]:
    """dump 层展开: jsonl 行按 level 过滤, 标注 = target 框(cls 恒 0——dump 无类别)."""
    imgs = img_by_stem(spec.images_dir)
    for line in spec.labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = orjson.loads(line)
        if row.get("level") not in levels:
            continue
        stem = str(row["stem"])
        if stem not in imgs:
            raise ValueError(f"dump 缺图: {stem} (images_dir={spec.images_dir})")
        boxes = [tuple(b) for b in row["target"]]
        yield stem, [(b, 0) for b in boxes], imgs[stem]


def iter_yolo(spec: SourceSpec) -> Iterator[StemBoxes]:
    """现成 YOLO labels 层展开(cls 保留原值, 多类数据集可用)."""
    imgs = img_by_stem(spec.images_dir)
    for lbl in sorted(spec.labels_path.glob("*.txt")):
        img = imgs.get(lbl.stem)
        if img is None:
            raise ValueError(f"labels 缺图: {lbl.stem} (images_dir={spec.images_dir})")
        text = lbl.read_text(encoding="utf-8")
        yield lbl.stem, parse_labeled_yolo(text, lbl.name), img


def labeled_boxes_from_xanylabel(yaml_path: Path) -> list[LabeledBox]:
    """X-AnyLabeling 标注 YAML(version 2.0) → [带类框].

    objects[].polygon 归一化顶点 → xyxy(min/max), category → cls;
    rois 为监测区不属标注, 忽略.
    """
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    boxes: list[LabeledBox] = []
    for obj in doc.get("objects", []):
        pts = [(float(p["x"]), float(p["y"])) for p in obj.get("polygon", [])]
        if len(pts) < 3:
            raise ValueError(f"{yaml_path.name}: polygon 点数 {len(pts)} < 3")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = (min(xs), min(ys), max(xs), max(ys), float(obj.get("confidence", 1.0)))
        boxes.append((box, int(obj.get("category", 0))))
    return boxes


def iter_xanylabel(spec: SourceSpec) -> Iterator[StemBoxes]:
    """X-AnyLabeling 层展开: 逐 YAML objects → 带类框(category → cls)."""
    imgs = img_by_stem(spec.images_dir)
    for yp in sorted(spec.labels_path.glob("*.yaml")):
        img = imgs.get(yp.stem)
        if img is None:
            raise ValueError(f"xanylabel 缺图: {yp.stem} (images_dir={spec.images_dir})")
        yield yp.stem, labeled_boxes_from_xanylabel(yp), img


def iter_layer(spec: SourceSpec, levels: frozenset[str]) -> Iterator[StemBoxes]:
    """任一标注源 → 统一 (stem, 框, 图) 流(三格式在此归一)."""
    match spec.kind:
        case LabelKind.DUMP:
            yield from iter_dump(spec, levels)
        case LabelKind.YOLO:
            yield from iter_yolo(spec)
        case LabelKind.XANYLABEL:
            yield from iter_xanylabel(spec)


def _write_frame(all_dir: Path, stem: str, boxes: list[LabeledBox], img: Path) -> int:
    """写单帧(label 文本 + 图 symlink), 返回框数."""
    (all_dir / "labels" / f"{stem}.txt").write_text(
        labeled_to_text(boxes), encoding="utf-8"
    )
    (all_dir / "images" / img.name).symlink_to(img.resolve())
    return len(boxes)


def register_stem(seen: dict[str, LabelKind], stem: str, kind: LabelKind) -> None:
    """注册 stem → 层映射; 跨层重复立即失败(不静默覆盖)."""
    if (prev := seen.get(stem)) is not None:
        raise ValueError(f"stem 冲突: {stem} 同时出现于 {prev.value} 与 {kind.value}")
    seen[stem] = kind


_DATA_YAML = """path: {root}
train: train/images
val: val/images
test: test/images
names:
{names}
"""


def merge(
    sources: Sequence[SourceSpec],
    out_dir: Path,
    classes: Sequence[str],
    levels: frozenset[str],
) -> list[LayerStats]:
    """按序合并层 → <out>/all/ + classes.txt + data.yaml, 返回各层统计.

    冲突检测与写入同帧同步(先检后写, 失败即中止). 失败可留半写 all/,
    同 out_dir 重跑全清 all/ 重建——冲突修复后重跑是标准恢复路径.
    """
    if not sources:
        raise ValueError("零标注源")
    for spec in sources:  # 入口校验: typo 路径立即失败, 不静默产出零帧层
        if not spec.labels_path.exists():
            raise ValueError(f"层标注路径不存在: {spec.labels_path} ({spec.kind.value})")
        if not spec.images_dir.is_dir():
            raise ValueError(f"层图片目录不存在: {spec.images_dir} ({spec.kind.value})")
    seen: dict[str, LabelKind] = {}
    stats: list[LayerStats] = []
    all_dir = out_dir / "all"
    if all_dir.exists():
        # 重跑全清: 撤层/缩量后陈旧帧不残留(否则统计与文件不符, 旧帧经 jxl_split
        # 流入训练); train/val/test 为 jxl_split 产物, 由其 remake_dirs 自清
        shutil.rmtree(all_dir)
    (all_dir / "images").mkdir(parents=True)
    (all_dir / "labels").mkdir(parents=True)
    for spec in sources:
        frames = boxes = 0
        for stem, bs, img in iter_layer(spec, levels):
            register_stem(seen, stem, spec.kind)
            boxes += _write_frame(all_dir, stem, bs, img)
            frames += 1
        stats.append(LayerStats(spec.kind, frames, boxes))
    total = sum(s.frames for s in stats)
    if not total:
        raise ValueError("零帧输出, 拒绝生成数据集")
    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (out_dir / "data.yaml").write_text(
        _DATA_YAML.format(root=out_dir.resolve(), names=names), encoding="utf-8"
    )
    return stats


@app.command()
def main(
    out_dir: Annotated[Path, typer.Argument(help="输出数据集目录")],
    dump: Annotated[
        list[str],
        typer.Option("--dump", help="dump 层 <validators_all.jsonl>:<images_dir>"),
    ] = [],
    dump_level: Annotated[
        list[str],
        typer.Option(
            "--dump-level",
            help="dump 层纳入的 level(默认 L0; L0=确认样本——五模型全一致)",
        ),
    ] = ["L0"],
    yolo: Annotated[
        list[str], typer.Option("--yolo", help="YOLO labels 层 <labels_dir>:<images_dir>")
    ] = [],
    xanylabel: Annotated[
        list[str],
        typer.Option("--xanylabel", help="X-AnyLabeling 层 <yaml_dir>:<images_dir>"),
    ] = [],
    classes: Annotated[
        list[str], typer.Option("--classes", help="类名表(按 cls id 序)")
    ] = ["person"],
) -> None:
    """合并分层标注 → 平铺集 + classes.txt + data.yaml(配合 jxl_split + yolo_train)."""
    specs = [
        *(SourceSpec.from_str(s, LabelKind.DUMP) for s in dump),
        *(SourceSpec.from_str(s, LabelKind.YOLO) for s in yolo),
        *(SourceSpec.from_str(s, LabelKind.XANYLABEL) for s in xanylabel),
    ]
    if not specs:
        typer.secho("至少指定一个层(--dump/--yolo/--xanylabel)", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    try:
        stats = merge(specs, out_dir, classes, frozenset(dump_level))
    except (ValueError, orjson.JSONDecodeError, yaml.YAMLError) as e:
        typer.secho(f"合并失败: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    for s in stats:
        typer.echo(f"  {s.kind.value:10s} {s.frames:6d} 帧 / {s.boxes:6d} 框")
    typer.secho(
        f"合计 {sum(s.frames for s in stats)} 帧 / {sum(s.boxes for s in stats)} 框 → {out_dir}"
        "(all/ 平铺 + data.yaml; jxl_split 划分后 yolo_train 训练)",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
