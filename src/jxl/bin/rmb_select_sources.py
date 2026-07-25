#!/usr/bin/env python3
"""从 VLM 打标结果筛选高质量钱币作可灵图生图参考源。

读 describe ndjson，过滤（真实人民币 + 适合合成 + 完整 + 无伪影），
按面额平衡选 N 张，偏好平铺/纯背景/高清，复制到 assets/sources_selected/{denom}/。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="筛选高质量钱币作可灵参考源。")

CANON = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]
_SUIT = {"good": 0, "ok": 1, "bad": 2}
_ANGLE = {"flat": 0, "tilted": 1, "oblique": 2}
_BG = {"plain": 0, "simple": 1, "complex": 2}
_SHARP = {"high": 0, "medium": 1, "low": 2}


@app.command()
def run(
    ndjson: Path = typer.Option(Path("assets/descriptions/bat67-rmb-dataset.ndjson"), "--ndjson", help="describe 产出。"),
    src_root: Path = typer.Option(Path("assets/datasets/bat67-rmb-dataset"), "--src-root", help="image 字段相对此根。"),
    out: Path = typer.Option(Path("assets/sources_selected"), "--out"),
    per_denom: int = typer.Option(30, "--per-denom", help="每面额选多少张。"),
) -> None:
    rows = [json.loads(line) for line in ndjson.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_denom: dict[str, list[tuple[int, dict]]] = {d: [] for d in CANON}
    for r in rows:
        if r.get("error") or not r.get("is_real_rmb"):
            continue
        if r.get("synthesis_suitability") == "bad" or r.get("completeness") != "full" or r.get("has_artifact"):
            continue
        denom = r.get("denomination")
        if denom not in ("1", "5", "10", "20", "50", "100"):
            continue
        key = f"{denom}yuan"
        # 偏好分（越小越好）：适合度为主，其次平铺/纯背景/高清
        score = (_SUIT.get(r.get("synthesis_suitability"), 2) * 100
                 + _ANGLE.get(r.get("view_angle"), 2) * 10
                 + _BG.get(r.get("background_complexity"), 2)
                 + _SHARP.get(r.get("sharpness"), 2))
        by_denom[key].append((score, r))

    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for d in CANON:
        picked = sorted(by_denom[d], key=lambda x: x[0])[:per_denom]
        (out / d).mkdir(parents=True, exist_ok=True)
        for _, r in picked:
            src = src_root / r["image"]
            if not src.exists():
                continue
            dst = out / d / Path(r["image"]).name
            shutil.copy2(src, dst)
            manifest.append({
                "denom": d, "src": str(src), "out": str(dst),
                "side": r.get("side"), "view_angle": r.get("view_angle"),
                "suitability": r.get("synthesis_suitability"),
            })
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.secho(f"筛选 {len(manifest)} 张参考源 → {out}", fg=typer.colors.GREEN)
    for d in CANON:
        typer.echo(f"  {d}: {sum(1 for m in manifest if m['denom'] == d)}")


if __name__ == "__main__":
    app()
