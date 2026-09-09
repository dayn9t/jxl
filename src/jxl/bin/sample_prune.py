#!/usr/bin/env python3
"""sample_prune CLI: 训练样本簇内削减(近重复优先)+隔离池. spec 见 docs/2026-09-09-训练样本量控制机制设计.md."""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from jcx.sys.fs import files_in

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
    stems = [f.stem for f in files_in(ds_dir / "images", ".jpg")]
    emb = np.load(embeddings).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    if len(emb) != len(stems):
        raise SystemExit(f"FATAL: embeddings {len(emb)} != 帧数 {len(stems)} (行序=stems 排序)")
    # 行序交叉校验: embed_dino 产物 <emb>.txt 与数据集排序逐一比对, 防同数错序静默错绑
    sidecar = embeddings.with_suffix(".txt")
    if sidecar.is_file():
        emb_stems = [Path(line).stem
                     for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
        # strict=False: 长度不等本身即不一致, 由下方 len 比对单独报出
        diffs = [f"行{i}: 数据集 {s} / 侧车 {e}"
                 for i, (s, e) in enumerate(zip(stems, emb_stems, strict=False)) if s != e][:3]
        if diffs or len(emb_stems) != len(stems):
            raise SystemExit(
                f"FATAL: {sidecar.name} 行序与数据集 stems 排序不一致 "
                f"(侧车 {len(emb_stems)} 行 / 数据集 {len(stems)} 帧): {'; '.join(diffs)}"
            )
    conf_by_stem: dict[str, float] = {}
    for line in confs.open(encoding="utf-8"):
        r = json.loads(line)
        conf_by_stem[r["stem"]] = max(r.get("confs") or [0.0])
    confs_sorted = [conf_by_stem.get(s, 0.0) for s in stems]
    p = decide(emb, confs_sorted, ratio)
    plan_doc = {"stems": stems, "keep": list(p.keep), "pool": list(p.pool), "meta": list(p.meta)}
    out.write_text(json.dumps(plan_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    # stems 侧车: 行序=嵌入行序, 供 apply 阶段回核行-茎对齐
    out.with_name(out.name + ".stems.txt").write_text("\n".join(stems), encoding="utf-8")
    typer.echo(f"plan: {len(stems)} 帧 -> keep {len(p.keep)} / pool {len(p.pool)} -> {out}")


if __name__ == "__main__":
    app()
