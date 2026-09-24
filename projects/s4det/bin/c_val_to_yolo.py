#!/usr/bin/env python3
"""s4det 任务书 C——validators.jsonl 指定 level 行 → darknet labels 层.

用途: det_mine 的 labels/ 目录混含多帧级(L1+review), 直接 --yolo 会与仲裁层
stem 冲突; 本工具按 level 精确导出(target 字段框, x1y1x2y2+conf 归一化 →
darknet cx cy w h), 保证层间 stem 互斥。

典型用法:
    python c_val_to_yolo.py <validators.jsonl> <labels_out> --level L1 --cls-id 0
"""

from pathlib import Path
from typing import Annotated

import orjson
import typer
from loguru import logger

app = typer.Typer(add_completion=False, help="任务书 C: validators 行 → darknet labels.")


def export(src: Path, out: Path, level: str, cls_id: int) -> tuple[int, int]:
    """按 level 过滤导出; 返回 (帧数, 框数)."""
    out.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    n_boxes = 0
    for ln in src.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = orjson.loads(ln)
        if r.get("level") != level or not (r.get("target") or []):
            continue
        lines = []
        for b in r["target"]:
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            lines.append(f"{cls_id} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
        (out / f"{r['stem']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_frames += 1
        n_boxes += len(lines)
    return n_frames, n_boxes


@app.command()
def main(
    validators: Annotated[Path, typer.Argument(help="det_mine validators.jsonl")],
    out: Annotated[Path, typer.Argument(help="labels 输出目录")],
    level: Annotated[str, typer.Option(help="帧级过滤(L0/L1/review)")],
    cls_id: Annotated[int, typer.Option(help="classes 表位次")],
) -> None:
    """导出指定层为 darknet labels."""
    n_frames, n_boxes = export(validators, out, level, cls_id)
    logger.info("[level={}] {} 帧 / {} 框 → {}", level, n_frames, n_boxes, out)


if __name__ == "__main__":
    app()
