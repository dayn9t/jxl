#!/usr/bin/env python3
"""sample_prune CLI: 训练样本簇内削减(近重复优先)+隔离池. spec 见 docs/2026-09-09-训练样本量控制机制设计.md."""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from jxl.sample_prune import decide

app = typer.Typer(help="样本簇内削减+隔离池(plan/apply/pool-review)")


@app.callback()
def main() -> None:
    """样本簇内削减+隔离池. 需显式子命令(plan/apply/pool-review), 保持多命令组形态."""


@app.command()
def plan(
    ds_dir: Annotated[Path, typer.Argument(help="数据集目录(images/+labels/)")],
    embeddings: Annotated[Path, typer.Option(help="帧 DINOv2 embeddings.npy, 行序=stems 排序")],
    confs: Annotated[Path, typer.Option(help="gencheck_infer 格式 jsonl")],
    ratio: Annotated[float, typer.Option(help="削减比例 0-1")] = 0.2,
    out: Annotated[Path, typer.Option(help="plan.json 输出")] = Path("prune_plan.json"),
) -> None:
    stems = sorted(p.stem for p in (ds_dir / "images").glob("*.jpg"))
    emb = np.load(embeddings).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    if len(emb) != len(stems):
        raise SystemExit(f"FATAL: embeddings {len(emb)} != 帧数 {len(stems)} (行序=stems 排序)")
    conf_by_stem: dict[str, float] = {}
    for line in confs.open(encoding="utf-8"):
        r = json.loads(line)
        conf_by_stem[r["stem"]] = max(r.get("confs") or [0.0])
    confs_sorted = [conf_by_stem.get(s, 0.0) for s in stems]
    p = decide(emb, confs_sorted, ratio)
    plan_doc = {"stems": stems, "keep": list(p.keep), "pool": list(p.pool), "meta": list(p.meta)}
    out.write_text(json.dumps(plan_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    typer.echo(f"plan: {len(stems)} 帧 -> keep {len(p.keep)} / pool {len(p.pool)} -> {out}")


if __name__ == "__main__":
    app()
