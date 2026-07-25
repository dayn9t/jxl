#!/usr/bin/env python3
"""评估 grounding 结果 vs rmb_yolo GT：定位 IoU/召回/精度/F1 + 面额准确率。

读 ground_notes 产出的 ndjson + GT labels，按图 stem 匹配，
每图做 GT↔检测框贪心匹配（IoU≥阈值），汇总定位指标与面额准确率。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import typer

from jxl.det.box_utils import xyxy_iou

app = typer.Typer(add_completion=False, help="评估 grounding vs GT：IoU/召回/精度/F1/面额准确率。")

CANON = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]
_DENOM_RE = re.compile(r"(?<!\d)(100|50|20|10|5|1)(?!\d)")


def denom_of(label: str) -> str | None:
    m = _DENOM_RE.search(label.lower())
    return f"{m.group(1)}yuan" if m else None


def load_gt(label_dir: Path) -> dict[str, list[tuple[int, list[float]]]]:
    """Stem -> [(cls, [x1,y1,x2,y2])]，YOLO cx cy w h → xyxy。"""
    out: dict[str, list[tuple[int, list[float]]]] = {}
    for f in label_dir.glob("*.txt"):
        boxes: list[tuple[int, list[float]]] = []
        for line in f.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            cx, cy, w, h = (float(x) for x in p[1:5])
            boxes.append((cls, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
        out[f.stem] = boxes
    return out


@app.command()
def run(
    grounding: Path = typer.Option(..., "--grounding", help="ground_notes 产出的 ndjson。"),
    label_dir: Path = typer.Option(Path("assets/rmb_yolo/labels/valid"), "--labels", help="GT labels 目录。"),
    iou: float = typer.Option(0.5, "--iou", help="匹配 IoU 阈值。"),
) -> None:
    """汇总定位与面额指标。"""
    gts = load_gt(label_dir)
    rows = [json.loads(line) for line in grounding.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_gt = n_det = tp = denom_ok = denom_total = n_imgs = n_err = 0
    matched_iou: list[float] = []
    for r in rows:
        stem = Path(r["image"]).stem
        if "error" in r:
            n_err += 1
            continue
        gts_img = gts.get(stem, [])
        dets = r.get("detections", [])
        n_gt += len(gts_img)
        n_det += len(dets)
        n_imgs += 1
        pairs = [
            (xyxy_iou(gb, d["bbox"]), gi, di)
            for gi, (_, gb) in enumerate(gts_img)
            for di, d in enumerate(dets)
        ]
        pairs.sort(reverse=True)
        used_g: set[int] = set()
        used_d: set[int] = set()
        for iov, gi, di in pairs:
            if iov < iou:
                break
            if gi in used_g or di in used_d:
                continue
            used_g.add(gi)
            used_d.add(di)
            tp += 1
            matched_iou.append(iov)
            d_denom = denom_of(dets[di]["label"])
            if d_denom is not None:
                denom_total += 1
                if d_denom == CANON[gts_img[gi][0]]:
                    denom_ok += 1
    rec = tp / max(n_gt, 1)
    prec = tp / max(n_det, 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    miou = sum(matched_iou) / len(matched_iou) if matched_iou else 0.0
    typer.secho(f"{grounding.name}  IoU≥{iou}", fg=typer.colors.CYAN)
    typer.echo(f"  图数 {n_imgs} (错误 {n_err}) | GT框 {n_gt} | 检测框 {n_det}")
    typer.echo(f"  召回 {rec:.3f} | 精度 {prec:.3f} | F1 {f1:.3f} | 匹配均IoU {miou:.3f}")
    typer.echo(f"  面额准确 {denom_ok}/{denom_total} = {denom_ok / max(denom_total, 1):.3f}")


if __name__ == "__main__":
    app()
