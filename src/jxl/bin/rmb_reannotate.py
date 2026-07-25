#!/usr/bin/env python3
"""可灵生成图 reannotation + 面额校验 → YOLO 检测集。

对 assets/synth/ 可灵生成图：豆包 grounding 出 bbox+面额，校验检测面额 ∈ manifest
记录的参考图面额（不符的框丢弃；整图无有效框则丢弃图），通过的组织成 YOLO
images+labels（cls=面额id），供并入 rmb_yolo 训练。

⚠️ 依赖可灵生成图（assets/synth/），当前阻塞等可灵充值。
复用 ground_notes 的豆包 grounding。
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

import httpx
import typer

from jxl.bin.rmb_ground import Backend, ground_one, load_backend

app = typer.Typer(add_completion=False, help="可灵生成图 reannotation + 面额校验 → YOLO。")

CANON = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]
DENOM_ID = {d: i for i, d in enumerate(CANON)}
_DENOM_RE = re.compile(r"(?<!\d)(100|50|20|10|5|1)(?!\d)")


def denom_of(label: str) -> str | None:
    m = _DENOM_RE.search(label.lower())
    return f"{m.group(1)}yuan" if m else None


def ref_denoms_from_manifest(synth: Path) -> dict[str, set[str]]:
    """生成图 stem → 参考面额集合。manifest 缺失则报错退出（No Silent Degradation）。"""
    mp = synth / "manifest.json"
    if not mp.exists():
        typer.secho(f"manifest 缺失: {mp}，无法做面额校验。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    out: dict[str, set[str]] = {}
    for m in json.loads(mp.read_text(encoding="utf-8")):
        ds: set[str] = {d for r in m.get("refs", []) if (d := denom_of(Path(r).parent.name)) is not None}
        for fp in m.get("out", []):
            out[Path(fp).stem] = ds
    return out


@app.command()
def run(
    synth: Path = typer.Option(Path("assets/synth"), "--synth", help="可灵生成图目录。"),
    out: Path = typer.Option(Path("assets/rmb_synth"), "--out", help="输出 YOLO 集根。"),
    backend: Backend = typer.Option(Backend.DOUBAO, "-b", "--backend", help="grounding 后端。"),
    concurrency: int = typer.Option(6, "--concurrency"),
) -> None:
    base, key, model = load_backend(backend, "", "")
    imgs = sorted(p for p in synth.glob("*.jpg"))
    if not imgs:
        typer.secho(f"{synth} 无生成图（等可灵充值生成）。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    ref_map = ref_denoms_from_manifest(synth)
    matched = sum(1 for p in imgs if p.stem in ref_map)
    if matched < len(imgs):
        typer.secho(f"警告：{len(imgs) - matched}/{len(imgs)} 张生成图在 manifest 无参考记录，将被 drop。", fg=typer.colors.YELLOW)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    typer.secho(f"reannotation {len(imgs)} 张生成图 @ {backend.value}", fg=typer.colors.CYAN)

    async def main() -> list[tuple[Path, list, str | None]]:
        results: list[tuple[Path, list, str | None]] = []
        async with httpx.AsyncClient() as client:
            tasks = [ground_one(client, sem, p, base, key, model) for p in imgs]
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
        return results

    results = asyncio.run(main())
    n_keep = n_drop = 0
    for path, dets, err in results:
        if err or not dets:
            n_drop += 1
            continue
        allowed = ref_map.get(path.stem)  # 参考面额集合；无记录则 drop 整图（No Silent Degradation）
        if allowed is None:
            n_drop += 1
            continue
        lines: list[str] = []
        for d in dets:
            den = denom_of(d.label)
            if den is None or den not in DENOM_ID or den not in allowed:
                continue  # 面额不符参考图 → 丢弃该框
            x1, y1, x2, y2 = d.bbox
            x1, x2 = min(x1, x2), max(x1, x2)  # 规整，防负 w/h
            y1, y2 = min(y1, y2), max(y1, y2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            lines.append(f"{DENOM_ID[den]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not lines:
            n_drop += 1
            continue
        shutil.copy2(path, out / "images" / path.name)
        (out / "labels" / (path.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_keep += 1
    typer.secho(f"保留 {n_keep} | 丢弃 {n_drop} → {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
