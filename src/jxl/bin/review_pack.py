#!/usr/bin/env python3
"""Review-Pack: det_mine review 清单 → 多模型色框网格 + manifest + 审核指引.

每条 review 图按模型色(target 黑/yoloe 蓝/gdino 黄/rfdetr 绿/la 红)叠加检测框 +
header(stem + 争议分), 每 per-grid 张拼一张网格(review_grid_001.jpg ...), 供人工
快速过审; manifest.jsonl 原样合并复制, README.txt 给色图例与操作指引.

用法:
    review_pack <consensus_dir> <images_dir> <out_dir> [--per-grid 20]
"""

from pathlib import Path
from typing import Annotated, NamedTuple

import orjson
import typer
from PIL import Image

from jxl.bin.det_mine import gather_images
from jxl.det.hardmine import Box
from jxl.det.viz import RGB, draw_boxes, grid, label_header, scale_to_width

app = typer.Typer(add_completion=False, help="review 清单 → 多模型色框网格审核材料.")

MODEL_COLORS: dict[str, RGB] = {
    "target": (0, 0, 0),
    "yoloe": (60, 120, 255),
    "gdino": (255, 200, 0),
    "rfdetr": (0, 200, 0),
    "la": (255, 40, 40),
}
_TILE_W = 640
_GRID_COLS = 4
_HEADER_STEM_MAX = 28


class ReviewEntry(NamedTuple):
    """manifest 单行解析结果(det_mine review/manifest.jsonl 行格式)."""

    image: str
    score: float
    boxes_by_model: dict[str, list[Box]]


def _coerce_boxes(raw: object) -> list[Box]:
    """json 值 → Box 列表(边界校验: 每框 5 元数值)."""
    if not isinstance(raw, list):
        raise ValueError(f"manifest 框字段非数组: {raw!r}")
    out: list[Box] = []
    for b in raw:
        if not isinstance(b, list) or len(b) != 5:
            raise ValueError(f"manifest 框非 5 元组: {b!r}")
        out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])))
    return out


def parse_entry(line: str) -> ReviewEntry:
    """manifest 单行(json) → ReviewEntry; target_boxes + validators 并入逐模型框."""
    d: dict[str, object] = orjson.loads(line)
    boxes: dict[str, list[Box]] = {"target": _coerce_boxes(d["target_boxes"])}
    raw_vs = d["validators"]
    if not isinstance(raw_vs, dict):
        raise ValueError(f"manifest validators 非对象: {raw_vs!r}")
    for name, raw in raw_vs.items():
        boxes[str(name)] = _coerce_boxes(raw)
    raw_score = d["score"]
    if not isinstance(raw_score, (int, float)):
        raise ValueError(f"manifest score 非数值: {raw_score!r}")
    return ReviewEntry(str(d["image"]), float(raw_score), boxes)


def render_tile(
    img: Image.Image, boxes_by_model: dict[str, list[Box]], stem: str
) -> Image.Image:
    """单图 tile: 统一宽 640 → 按 MODEL_COLORS 逐模型叠框 → 顶部 header.

    stem 参数即标题文本(调用方拼好 f"{stem[:28]} s{score:.2f}").
    """
    im = scale_to_width(img, _TILE_W)
    for name, color in MODEL_COLORS.items():
        if name in boxes_by_model:
            im = draw_boxes(im, boxes_by_model[name], color)
    return label_header(im, stem)


def find_manifests(consensus_dir: Path) -> list[Path]:
    """manifest 定位: 优先 <dir>/review/manifest.jsonl, 否则递归 glob manifest*.jsonl."""
    primary = consensus_dir / "review" / "manifest.jsonl"
    if primary.is_file():
        return [primary]
    return sorted(consensus_dir.rglob("manifest*.jsonl"))


def _readme_text(consensus_dir: Path, per_grid: int) -> str:
    """README.txt 内容: 色图例 + 文件说明 + 审核操作指引."""
    words = {"target": "黑", "yoloe": "蓝", "gdino": "黄", "rfdetr": "绿", "la": "红"}
    legend = "\n".join(
        f"  {name:<7} {words[name]} rgb{color}" for name, color in MODEL_COLORS.items()
    )
    return f"""n001 共识标注 — 人工审核材料
================================

色图例(每张图叠加各模型检测框, target 为被校验基准):
{legend}

文件说明:
  review_grid_001.jpg ...  审核网格, 每张 {per_grid} 图({_GRID_COLS} 列);
                           标题行 = stem(截 {_HEADER_STEM_MAX} 字符) + s争议分(越高分歧越大);
  manifest.jsonl           review 条目原样合并(image/score/target_boxes/validators/breakdown);
  _missing.jsonl           图片缺失被跳过的条目(原样行), 排查 images_dir 后重跑本工具。

审核操作:
  1. 逐张查看网格图, 按标题行 stem 定位对应条目与原图;
  2. 多数模型一致、分歧可接受 → 无需处理, 维持共识自动标注;
  3. 标注需修正 → 直接修改 {consensus_dir}/labels/<stem>.txt(YOLO 归一化 xywh);
  4. 整图拒收(误检严重/无法标注) → 将该 stem 记入拒收清单, 不入池。
"""


@app.command()
def run(
    consensus_dir: Annotated[Path, typer.Argument(help="det_mine 共识输出目录(含 review/)")],
    images_dir: Annotated[Path, typer.Argument(help="帧图目录(按 stem 递归找图)")],
    out_dir: Annotated[Path, typer.Argument(help="审核材料输出目录")],
    per_grid: Annotated[int, typer.Option("--per-grid", help="每张网格图条目数")] = 20,
) -> None:
    """review 清单 → 多模型色框网格 + manifest + README 人工审核包."""
    if per_grid < 1:
        typer.secho(f"--per-grid 须 >=1: {per_grid}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    lines = [
        ln
        for m in find_manifests(consensus_dir)
        for ln in m.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        typer.secho(f"无 review manifest 条目: {consensus_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    entries = [parse_entry(ln) for ln in lines]
    unknown = sorted({n for e in entries for n in e.boxes_by_model} - set(MODEL_COLORS))
    if unknown:
        typer.secho(f"警告: 未知模型名(不画框): {unknown}", fg=typer.colors.YELLOW, err=True)

    img_by_stem = {p.stem: p for p in gather_images(images_dir)}
    tiles: list[Image.Image] = []
    missing: list[str] = []
    for ln, entry in zip(lines, entries, strict=True):
        stem = Path(entry.image).stem
        img_path = img_by_stem.get(stem)
        if img_path is None:
            typer.secho(f"警告: 图片缺失跳过 {entry.image}", fg=typer.colors.YELLOW, err=True)
            missing.append(ln)
            continue
        header = f"{stem[:_HEADER_STEM_MAX]} s{entry.score:.2f}"
        img = Image.open(img_path).convert("RGB")
        tiles.append(render_tile(img, entry.boxes_by_model, header))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing:
        (out_dir / "_missing.jsonl").write_text("\n".join(missing) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(_readme_text(consensus_dir, per_grid), encoding="utf-8")
    if not tiles:
        typer.secho(
            f"全部 {len(lines)} 条目图片缺失, 未产出网格", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)

    n_grids = 0
    for i in range(0, len(tiles), per_grid):
        n_grids += 1
        grid(tiles[i : i + per_grid], cols=_GRID_COLS, tile_w=_TILE_W).save(
            out_dir / f"review_grid_{n_grids:03d}.jpg", quality=88
        )
    typer.secho(
        f"review_pack: {len(tiles)}/{len(lines)} 条 → {n_grids} 网格, "
        f"缺图 {len(missing)} → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
