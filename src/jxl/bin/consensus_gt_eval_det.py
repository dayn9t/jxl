"""共识 GT × 检测器对比评估（2026-09-13）。

用 vlm_consensus_gt 产出的 trusted GT 评估检测器（v3/v4）在帧集上的
召回（GT 命中率）与疑似误检（检出无 GT 匹配），输出双模型对照报告。

评估口径（部署对齐）：
  consensus GT 的 box_norm（0-1 全图）× 检测结果（全图坐标）
  GT 命中 = 存在检出框 IoU≥0.5；检出无匹配 = 疑似 FP（或 VLM 集体漏，见 low_agreement 佐证）

用法：
  uv run --project . python -m jxl.bin.consensus_gt_eval_det \
      GT_TRUSTED_JSONL --det weights.pt [weights2.pt ...] --out report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import orjson
import typer
from PIL import Image
from ultralytics import YOLO

from jxl.bin.vlm_pool import iou


def _crop_image(path: Path, rect: tuple[float, float, float, float]) -> Path:
    """裁出部署 ROI（缓存于同目录 .crop.jpg，幂等复用）。"""
    out = path.with_suffix(".crop.jpg")
    if not out.exists():
        rx, ry, rw, rh = rect
        Image.open(path).convert("RGB").crop(
            (int(rx), int(ry), int(rx + rw), int(ry + rh))).save(out, quality=90)
    return out

app = typer.Typer(help="共识 GT vs 检测器：召回/疑似误检对照")


@app.command()
def run(
    gt_jsonl: Path = typer.Argument(..., help="vlm_consensus_gt 的 gt_trusted.jsonl"),
    det: list[Path] = typer.Option(..., "--det", help="检测权重（可多份对照）"),
    out: Path = typer.Option(Path("gt_eval_report.json"), help="报告输出"),
    conf: float = typer.Option(0.5, help="检测置信阈（对齐部署）"),
    iou_th: float = typer.Option(0.5, help="GT 命中 IoU 阈"),
    limit: int = typer.Option(0, help=">0 只评前 N 帧"),
    crop_rect: str = typer.Option("", help="部署 ROI 'x,y,w,h'（全图坐标）；提供=部署口径：GT 滤进 rect、帧裁 rect 推理"),
) -> None:
    """逐帧双模型对照：GT 召回 / 疑似 FP / 框数变化。"""
    rows = [orjson.loads(l) for l in gt_jsonl.open() if l.strip()]
    if limit > 0:
        rows = rows[:limit]
    rect: tuple[float, float, float, float] | None = None
    if crop_rect:
        rx, ry, rw, rh = (float(v) for v in crop_rect.split(","))
        rect = (rx, ry, rw, rh)
    models = {str(p): YOLO(str(p)) for p in det}
    report: dict[str, dict] = {str(p): {"tp": 0, "n_gt": 0, "n_det": 0, "fps": []}
                               for p in det}
    for r in rows:
        W, H = float(r["row"]["width"]), float(r["row"]["height"])
        img = r["row"]["image"]
        if rect is not None:
            rx, ry, rw, rh = rect
            keep = []
            for b in r["consensus"]:
                n = b["box_norm"]
                cx, cy = (n[0] + n[2]) / 2 * W, (n[1] + n[3]) / 2 * H
                if rx <= cx < rx + rw and ry <= cy < ry + rh:  # 框中心在 rect 内才计入
                    keep.append({"box_norm": [round((n[0] * W - rx) / rw, 4),
                                              round((n[1] * H - ry) / rh, 4),
                                              round((n[2] * W - rx) / rw, 4),
                                              round((n[3] * H - ry) / rh, 4)]})
            gt = [tuple(b["box_norm"]) for b in keep]
            pred_img: str | Path = _crop_image(Path(img), rect)
        else:
            gt = [tuple(b["box_norm"]) for b in r["consensus"]]
            pred_img = img
        for p, model in models.items():
            res = model.predict(pred_img, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
            dets = [tuple(float(v) for v in b) for b in res.boxes.xyxy]
            if rect is not None:  # 检出（rect 局部像素）→ 归一化与 GT 同域
                dets = [(d[0] / rw, d[1] / rh, d[2] / rw, d[3] / rh) for d in dets]
            rep = report[p]
            rep["n_gt"] += len(gt)
            rep["n_det"] += len(dets)
            used: set[int] = set()
            for g in gt:
                hit = next((i for i, d in enumerate(dets)
                            if i not in used and iou(d, g) >= iou_th), None)
                if hit is not None:
                    used.add(hit)
                    rep["tp"] += 1
            for i, d in enumerate(dets):
                if i not in used:
                    rep["fps"].append({"image": img, "bbox": [round(v, 1) for v in d],
                                       "conf": round(float(res.boxes.conf[i]), 3)})
    summary = {}
    for p, rep in report.items():
        rc = rep["tp"] / rep["n_gt"] if rep["n_gt"] else 0.0
        summary[p] = {"weight": Path(p).name, "gt_recall@0.5": round(rc, 4),
                      "n_gt": rep["n_gt"], "n_det": rep["n_det"],
                      "suspected_fp": len(rep["fps"])}
        rep["fps"] = rep["fps"][:50]  # 报告只留前 50 条疑似 FP 明细
    out.write_bytes(orjson.dumps({"summary": summary,
                                  "detail": {k: {"fps": v["fps"]} for k, v in report.items()}},
                                 option=orjson.OPT_INDENT_2))
    typer.echo(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())
    typer.secho(f"报告 → {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
