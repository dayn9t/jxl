"""多 VLM 共识 GT 构造（2026-09-13，代替人工标注的审核标准）。

对帧清单逐帧跑投票池 person grounding（池配置/协议/共识规则见 vlm_pool.py，
KB 引用亦在彼处），产出分级共识 GT：
  trusted   强票 ≥2/3（默认三强中 ≥2 同意，IoU≥0.5 聚类）→ 直接入库候选
  arbited   强票 1 + 仲裁票 1（第四意见救回）→ 入库候选（标记弱共识）
  low_agreement → 人工池（小量，路由人工）
弃权票（调用失败）只缩小选民池，不当空票。

帧清单来源两种：
  a) 现成 jsonl：{"image": 绝对路径, "width", "height", ...meta}
  b) --from-manifest iapx manifest.jsonl --date D --top-minutes N：
     自动挑该日期含人帧最多的 N 个分钟窗，全部含人帧入清单（确定性选窗）

用法：
  uv run --project . python -m jxl.bin.vlm_consensus_gt FRAMES_JSONL \
      --out-dir DIR [--models ...] [--concurrency 6]
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
    consensus_boxes,
    image_data_url,
    call_person,
)

app = typer.Typer(help="多 VLM 共识 GT 构造（trusted/arbited/low_agreement 分级）")


def frames_from_manifest(manifest: Path, date: str, top_minutes: int) -> list[dict]:
    """确定性选窗：该日期按分钟聚合含人帧数，取 top-N 分钟窗的全部含人帧。"""
    rows = [orjson.loads(l) for l in manifest.open() if l.strip()]
    rows = [r for r in rows if r["date"] == date and r.get("n_persons", 0) >= 1]
    per_min: Counter[str] = Counter(r["wallclock"][:5] for r in rows)
    top = {m for m, _ in per_min.most_common(top_minutes)}
    base = manifest.parent  # manifest 在 samples/ 下且 file 自带 "samples/" 前缀 → 取其父目录
    return [{"image": str(base.parent / r["file"]), "width": 1920, "height": 1080,
             "date": r["date"], "wallclock": r["wallclock"], "minute": r["wallclock"][:5],
             "n_det": r["n_persons"]} for r in rows if r["wallclock"][:5] in top]


@app.command()
def run(
    frames_jsonl: Path = typer.Argument(..., help="帧清单 jsonl 或 --from-manifest 的 manifest 路径"),
    out_dir: Path = typer.Option(..., "--out-dir", help="输出目录（gt_trusted/low_agreement/report）"),
    models: str = typer.Option("qwen35b-local,qwen-flash,doubao-vl,glm-flash",
                               help="投票池别名（默认=2026-09-13 标定池）"),
    concurrency: int = typer.Option(6, help="跨帧并发（帧内四模型并行）"),
    limit: int = typer.Option(0, help=">0 只跑前 N 帧（冒烟）"),
    from_manifest: bool = typer.Option(False, "--from-manifest", help="输入是 iapx manifest，配合 --date/--top-minutes"),
    date: str = typer.Option("", help="--from-manifest：日期"),
    top_minutes: int = typer.Option(3, help="--from-manifest：取含人帧最多的 N 个分钟窗"),
) -> None:
    """跑共识：四模型逐帧投票 → 分级聚类 → GT/人工池/报告三件套。"""
    aliases = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [a for a in aliases if a not in CANDIDATES]
    if unknown:
        raise typer.BadParameter(f"未知候选: {unknown}；可选: {list(CANDIDATES)}")
    missing = [CANDIDATES[a]["key_env"] for a in aliases
               if CANDIDATES[a]["key_env"] and not __import__("os").environ.get(CANDIDATES[a]["key_env"])]
    if missing:
        raise typer.BadParameter(f"env 缺失（只列变量名）: {missing}")
    if from_manifest:
        rows = frames_from_manifest(frames_jsonl, date, top_minutes)
        typer.echo(f"manifest 选窗: {date} top{top_minutes}min → {len(rows)} 帧")
    else:
        rows = [orjson.loads(l) for l in frames_jsonl.open() if l.strip()]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise typer.BadParameter("帧清单为空")
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, r: dict) -> dict:
        url = image_data_url(Path(r["image"]))
        async with sem:
            votes = await asyncio.gather(*[call_person(client, a, url) for a in aliases])
        boxes = consensus_boxes(list(votes))
        return {"row": r,
                "votes": [{"alias": v.alias, "ok": v.ok, "n": len(v.boxes), "err": v.error}
                          for v in votes],
                "boxes": [{"box_norm": [round(v, 4) for v in c.box],
                           "status": c.status, "strong": c.strong_votes,
                           "arbiter": c.arbiter_votes, "members": list(c.members)}
                          for c in boxes]}

    async def _all() -> list[dict]:
        async with httpx.AsyncClient() as client:
            out = []
            for i in range(0, len(rows), concurrency):
                batch = rows[i:i + concurrency]
                out.extend(await asyncio.gather(*[one(client, r) for r in batch]))
                typer.echo(f"{min(i + concurrency, len(rows))}/{len(rows)}", err=True)
            return out

    results = asyncio.run(_all())
    trusted, low, stats = [], [], Counter()
    for res in results:
        stats["frames"] += 1
        ok_votes = [v for v in res["votes"] if v["ok"]]
        if len(ok_votes) < 2:
            stats["frames_abstain"] += 1  # 弃权过半，共识不成立 → 人工池
        statuses = {b["status"] for b in res["boxes"]}
        keep: list[dict] = []
        for b in res["boxes"]:
            stats[f"box_{b['status']}"] += 1
            if b["status"] != "low_agreement":
                keep.append(b)
        if keep and len(ok_votes) >= 2:
            trusted.append({"row": res["row"], "consensus": keep})
            stats["frames_trusted"] += 1
        else:
            low.append({"row": res["row"], "votes": res["votes"],
                        "boxes": res["boxes"]})
            stats["frames_low"] += 1
    _dump(out_dir / "gt_trusted.jsonl", trusted)
    _dump(out_dir / "low_agreement.jsonl", low)
    report = {"pool": {a: CANDIDATES[a]["model"] for a in aliases},
              "frames": len(rows), "stats": dict(stats),
              "rule": "strong>=2/3 trusted; strong==1+arbiter arbited; else low_agreement"}
    (out_dir / "consensus_report.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2))
    typer.echo(orjson.dumps(report["stats"], option=orjson.OPT_INDENT_2).decode())
    typer.secho(f"GT → {out_dir}/gt_trusted.jsonl | 人工池 → low_agreement.jsonl",
                fg=typer.colors.GREEN)


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    app()
