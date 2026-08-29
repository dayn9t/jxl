#!/usr/bin/env python3
"""La-Eval: la 重标注 vs 基准标注 逐图 IoU 对比 + 可视化抽检。

数值层: 逐图 greedy_match(IoU≥thr) → 框级 precision/recall/F1 + 图级一致率 +
分歧清单 jsonl(按 missed+extra 降序)。可视化层: PIL 网格对比图(绿=基准, 红=la),
分歧 top-N + 全体随机 N(固定 seed 可复现)。

基准非 GT(本身是多模型共识+豆包的自动标注)——结果是两套自动标注的一致性度量。

用法:
    la_eval <base_labels_dir> <la_labels_dir> <images_dir> <out_dir> [--iou 0.5]
"""

import random
import time
from pathlib import Path
from typing import Annotated, NamedTuple

import orjson
import typer
from PIL import Image, ImageDraw, ImageFont

from jxl.bin.det_mine import gather_images
from jxl.det.hardmine import greedy_match, parse_yolo_label

app = typer.Typer(add_completion=False, help="la 重标注 vs 基准对比评估.")

_PREVIEW_W = 640
_GRID_COLS = 4
_COLORS = {"base": (0, 200, 0), "la": (255, 40, 40)}


class ImageDiff(NamedTuple):
    """单图对比结果: matched 双方互认; missed 基准有 la 无(漏检); extra la 有基准无(多检)."""

    stem: str
    n_base: int
    n_la: int
    matched: int
    missed: list[int]
    extra: list[int]


def compare_label_text(base_txt: str, la_txt: str, iou_thr: float) -> ImageDiff:
    """两组 YOLO 标注文本 → ImageDiff(greedy_match IoU≥thr)."""
    base = parse_yolo_label(base_txt)
    la = parse_yolo_label(la_txt)
    matched, unmatched_base, unmatched_la = greedy_match(base, la, iou_thr)
    return ImageDiff(
        stem="",
        n_base=len(base),
        n_la=len(la),
        matched=len(matched),
        missed=unmatched_base,
        extra=unmatched_la,
    )


def aggregate(diffs: list[ImageDiff]) -> dict[str, object]:
    """逐图 diff 列表 → 聚合指标(框级 P/R/F1 + 图级一致率)."""
    n_base = sum(d.n_base for d in diffs)
    n_la = sum(d.n_la for d in diffs)
    matched = sum(d.matched for d in diffs)
    precision = matched / n_la if n_la else 0.0
    recall = matched / n_base if n_base else 0.0
    f1 = 2 * precision * recall / (precision + recall) if matched else 0.0
    perfect = sum(1 for d in diffs if not d.missed and not d.extra)
    return {
        "images": len(diffs),
        "perfect_images": perfect,
        "perfect_ratio": perfect / len(diffs) if diffs else 0.0,
        "base_boxes": n_base,
        "la_boxes": n_la,
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "la_missed_boxes": sum(len(d.missed) for d in diffs),
        "la_extra_boxes": sum(len(d.extra) for d in diffs),
    }


def draw_one(img_path: Path, d: ImageDiff, base_boxes: list, la_boxes: list) -> Image.Image:
    """单图绘制: 缩到预览宽, 绿框=基准, 红框=la, 顶部 stem+分歧数."""
    im = Image.open(img_path).convert("RGB")
    ratio = _PREVIEW_W / im.width
    im = im.resize((_PREVIEW_W, max(1, int(im.height * ratio))))
    draw = ImageDraw.Draw(im)
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()
    for b in base_boxes:
        x1, y1, x2, y2 = (v * im.width if i % 2 == 0 else v * im.height
                          for i, v in enumerate(b[:4]))
        draw.rectangle((x1, y1, x2, y2), outline=_COLORS["base"], width=3)
    for b in la_boxes:
        x1, y1, x2, y2 = (v * im.width if i % 2 == 0 else v * im.height
                          for i, v in enumerate(b[:4]))
        draw.rectangle((x1, y1, x2, y2), outline=_COLORS["la"], width=3)
    draw.rectangle((0, 0, im.width, 26), fill=(0, 0, 0))
    draw.text((4, 4), f"{d.stem[:28]} m{len(d.missed)}/e{len(d.extra)}", font=font,
              fill=(255, 255, 255))
    return im


def grid(images: list[Image.Image]) -> Image.Image:
    """等高拼网格(cols 固定, 行数自适应)."""
    h = max(im.height for im in images) + 26
    rows = (len(images) + _GRID_COLS - 1) // _GRID_COLS
    canvas = Image.new("RGB", (_PREVIEW_W * _GRID_COLS, h * rows), (30, 30, 30))
    for i, im in enumerate(images):
        canvas.paste(im, ((i % _GRID_COLS) * _PREVIEW_W, (i // _GRID_COLS) * h))
    return canvas


@app.command()
def run(
    base_labels: Annotated[Path, typer.Argument(help="基准 labels 目录(YOLO txt)")],
    la_labels: Annotated[Path, typer.Argument(help="la labels 目录(YOLO txt)")],
    images_dir: Annotated[Path, typer.Argument(help="图片目录(递归)")],
    out_dir: Annotated[Path, typer.Argument(help="输出目录(report/jsonl/jpg)")],
    iou: Annotated[float, typer.Option("--iou", help="IoU 匹配阈值")] = 0.5,
    top: Annotated[int, typer.Option("--top", help="分歧 top-N 可视化")] = 20,
    rand: Annotated[int, typer.Option("--rand", help="随机抽检 N 张")] = 20,
    seed: Annotated[int, typer.Option("--seed", help="随机抽检种子(可复现)")] = 42,
) -> None:
    """la 重标注 vs 基准: 数值一致性 + 可视化抽检."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_by_stem = {p.stem: p for p in gather_images(images_dir)}
    base_by_stem = {p.stem: p for p in base_labels.glob("*.txt")}
    la_by_stem = {p.stem: p for p in la_labels.glob("*.txt")}
    stems = sorted(set(base_by_stem) & set(la_by_stem) & set(img_by_stem))
    only = {"base_only": sorted(set(base_by_stem) - set(la_by_stem)),
            "la_only": sorted(set(la_by_stem) - set(base_by_stem))}

    diffs: list[ImageDiff] = []
    t0 = time.monotonic()
    for stem in stems:
        base_txt = base_by_stem[stem].read_text(encoding="utf-8")
        la_txt = la_by_stem[stem].read_text(encoding="utf-8")
        d = compare_label_text(base_txt, la_txt, iou)
        diffs.append(d._replace(stem=stem))
    typer.secho(f"对比 {len(diffs)} 图 in {time.monotonic()-t0:.1f}s", fg=typer.colors.CYAN)

    report = {"iou": iou, "la_missing": len(only["la_only"]), **aggregate(diffs)}
    (out_dir / "eval_report.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2))

    divergent = sorted((d for d in diffs if d.missed or d.extra),
                       key=lambda d: len(d.missed) + len(d.extra), reverse=True)
    with (out_dir / "divergent.jsonl").open("w", encoding="utf-8") as f:
        for d in divergent:
            f.write(orjson.dumps(d._asdict()).decode() + "\n")
    typer.secho(
        f"P={report['precision']:.4f} R={report['recall']:.4f} F1={report['f1']:.4f} "
        f"图一致率={report['perfect_ratio']:.4f} 分歧图={len(divergent)}",
        fg=typer.colors.GREEN)

    def render(sample: list[ImageDiff], name: str) -> None:
        if not sample:
            return
        tiles = [draw_one(img_by_stem[d.stem], d,
                          parse_yolo_label(base_by_stem[d.stem].read_text(encoding="utf-8")),
                          parse_yolo_label(la_by_stem[d.stem].read_text(encoding="utf-8")))
                 for d in sample]
        grid(tiles).save(out_dir / name, quality=88)

    render(divergent[:top], "preview_divergent_top.jpg")
    render(random.Random(seed).sample(diffs, min(rand, len(diffs))),
           "preview_random.jpg")
    typer.secho(f"产出 → {out_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
