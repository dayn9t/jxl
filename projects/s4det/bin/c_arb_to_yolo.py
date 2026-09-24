#!/usr/bin/env python3
"""s4det 任务书 C 步骤 3 产物转换——VLM 仲裁 gt_trusted → darknet labels 层.

输入 = vlm_consensus_gt 的 gt_trusted.jsonl({row:{image,stem}, consensus:[{box_norm,status}]})。
按 status 分两层落盘: trusted → <out>/trusted_yolo/labels, arbited → <out>/arbited_yolo/labels
(层级分开, 训练合并可分层取舍; images 由 consensus_dataset --yolo 的池目录提供)。

典型用法:
    python c_arb_to_yolo.py <mining/{leaf}/arbitration> <mining/{leaf}> --cls-id 0
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import orjson
import typer
from loguru import logger

app = typer.Typer(add_completion=False, help="任务书 C: 仲裁 GT → darknet labels.")


def convert(arb_dir: Path, out_root: Path, cls_id: int) -> dict[str, int]:
    """gt_trusted.jsonl → 按 status 分层的 darknet labels; 返回统计."""
    stats: Counter = Counter()
    src = arb_dir / "gt_trusted.jsonl"
    layers = {
        "trusted": out_root / "trusted_yolo" / "labels",
        "arbited": out_root / "arbited_yolo" / "labels",
    }
    for d in layers.values():
        d.mkdir(parents=True, exist_ok=True)
    for ln in src.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = orjson.loads(ln)
        stem = r["row"]["stem"]
        by_status: dict[str, list[str]] = {}
        for c in r.get("consensus", []):
            x1, y1, x2, y2 = c["box_norm"]
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            by_status.setdefault(c["status"], []).append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        for status, lines in by_status.items():
            dst = layers[status] / f"{stem}.txt"
            dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats[status] += len(lines)
    return dict(stats)


@app.command()
def main(
    arb_dir: Annotated[Path, typer.Argument(help="vlm_consensus_gt 输出目录")],
    out_root: Annotated[Path, typer.Argument(help="labels 落盘根目录")],
    cls_id: Annotated[int, typer.Option(help="该叶类在 classes 表位次")],
) -> None:
    """仲裁 GT → 分层 darknet labels."""
    stats = convert(arb_dir, out_root, cls_id)
    logger.info("转换 {} | cls_id={} | {}", arb_dir.name, cls_id, stats)


if __name__ == "__main__":
    app()
