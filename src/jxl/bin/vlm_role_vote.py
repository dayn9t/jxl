"""role 分类多模型投票（2026-09-14，代替人工复审的主路径）。

对 seed_verdicts.jsonl 中 verdict=uncertain 的 crop 跑四模型 role 投票
（池/共识规则见 vlm_pool.role_consensus——知识依据 KB vlm-vision-grounding）：
  trusted / arbited → 自动定类入池（verdict 非 uncertain 的）
  split → 真分歧残量（人工池，预计小量）

用法：
  uv run --project . python -m jxl.bin.vlm_role_vote \
      [--pool ub_true|all|head] [--limit N] --out OUT_JSONL
  --pool ub_true 只投 upper_body=true 的 405 张（默认，高价值层）
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import httpx
import orjson
import typer

from jxl.bin.vlm_pool import (
    CANDIDATES,
    call_role_vote,
    image_data_url,
    role_consensus,
)

app = typer.Typer(help="role 分类多模型投票（uncertain 复审自动化）")

R2 = Path("/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/crop640_persons/gencheck/iapx_round2")


def load_crops(pool: str) -> list[dict]:
    verd = [orjson.loads(l) for l in (R2 / "seed_verdicts.jsonl").open() if l.strip()]
    unc = [v for v in verd if v["verdict"] == "uncertain"]
    if pool == "ub_true":
        unc = [v for v in unc if v.get("upper_body")]
    elif pool == "head":
        unc = [v for v in unc if "头顶" in v.get("reason", "") or "头部" in v.get("reason", "")]
    elif pool == "p1":
        unc = [v for v in unc if str(v.get("obs", "")).startswith("p1")]
    elif pool == "p1_cust":
        # p1e/p1d 旧 4 类 prompt 判 customer 的 crop——七分类重投挖制服（teller/manager/security）
        unc = [v for v in verd
               if str(v.get("obs", "")).startswith("p1") and v["verdict"] == "customer"]
    idx: dict[str, Path] = {}
    for p in (R2 / "seed_crops").glob("*/*.jpg"):
        idx.setdefault(p.stem, p)
    out = []
    for v in unc:
        p = idx.get(v["uid"])
        if p:
            out.append(v | {"path": str(p)})
    return out


@app.command()
def run(
    pool: str = typer.Option("ub_true", help="ub_true=仅上半身疑难层 / head=仅头顶层 / all=全部"),
    models: str = typer.Option("qwen38-local,qwen-flash,doubao-vl,glm-flash"),
    out: Path = typer.Option(R2 / "role_vote_uncertain.jsonl", "--out"),
    concurrency: int = typer.Option(6),
    limit: int = typer.Option(0, help=">0 冒烟只跑 N 张"),
) -> None:
    """四模型逐张投票 → role_consensus → 自动定类 / 分歧残量统计。"""
    aliases = [m.strip() for m in models.split(",") if m.strip()]
    crops = load_crops(pool)
    if limit > 0:
        crops = crops[:limit]
    if not crops:
        raise typer.BadParameter("池为空")
    typer.echo(f"pool={pool} crops={len(crops)} models={aliases}", err=True)
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, c: dict) -> dict:
        url = image_data_url(Path(c["path"]))
        async with sem:
            votes = await asyncio.gather(*[call_role_vote(client, a, url) for a in aliases])
        vd = {a: v for a, v, _ in votes}
        errs = [f"{a}:{r}" for a, v, r in votes if v == "api_error"]
        ok_votes = {a: v for a, v in vd.items() if v not in ("api_error", "parse_error")}
        status, verdict = (("abstain", "") if len(ok_votes) < 2 else role_consensus(ok_votes))
        return {"uid": c["uid"], "obs": c.get("obs", ""), "spark_reason": c.get("reason", ""),
                "votes": vd, "errors": errs, "status": status, "verdict": verdict}

    async def _all() -> list[dict]:
        async with httpx.AsyncClient() as client:
            res = []
            for i in range(0, len(crops), concurrency):
                batch = crops[i:i + concurrency]
                res.extend(await asyncio.gather(*[one(client, r) for r in batch]))
                typer.echo(f"{min(i + concurrency, len(crops))}/{len(crops)}", err=True)
            return res

    results = asyncio.run(_all())
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    stats: Counter = Counter(r["status"] for r in results)
    verdicts = Counter(r["verdict"] for r in results if r["status"] in ("trusted", "arbited"))
    typer.echo(orjson.dumps({"status": dict(stats),
                             "auto_verdicts": dict(verdicts),
                             "manual_residual": stats["split"] + stats["abstain"]},
                            option=orjson.OPT_INDENT_2).decode())
    typer.secho(f"投票明细 → {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
