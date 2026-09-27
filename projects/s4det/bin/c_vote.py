#!/usr/bin/env python3
"""agree=false 行对象级类别投票（批 2 前置，C-execution 批 1 遗留 #3）.

manifest_leaf.jsonl 的 agree=false 行（平台 code 与 llm 类别不一致，类别悬置）
→ bbox ±10% 裁切（vlabel.crop 标准函数，bbox-crop-expansion 规范）
→ vlm_pool 池内模型菜单分类投票 → role_consensus：
    trusted/arbited → 自动定类（sidecar 票据，批 2 挖掘装配时回填类别先验）
    split           → 真分歧 → 隔离台账（gate=category-vote, priority=high）
    abstain         → 有效票 <2（API 故障），只记票据，重跑补票
断点重入：按 (event,rid,object_id) 行键跳过已投行。输出 sidecar，不改写 manifest；
训练框仍以共识挖掘为准（本票只修类别先验，不直接进训练集）。
菜单单一数据源 = projects/s4det/taxonomy.md 附录 A（导出件 taxonomy_leaves.json，
taxonomy 演进时同步重导出）。

典型用法:
    python c_vote.py --limit 6          # 冒烟
    python c_vote.py                    # 全量（断点重入）
    python c_vote.py --keys-from K --out SIDECAR   # 定向重投（键文件每行一个 event|rid|object_id）
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import httpx
import orjson
import typer
from a_common import DATA_ROOT, LEDGER, QuarantineEntry, append_quarantine, read_jsonl
from loguru import logger
from PIL import Image
from vlabel.crop import be_uniform, crop_expanded

from jxl.bin.vlm_pool import (
    CANDIDATES,
    call_role_vote,
    image_data_url,
    role_consensus,
)

app = typer.Typer(add_completion=False, help="agree=false 行类别投票.")

PREANNOT = DATA_ROOT / "preannot"
VOTE_BATCH = "2026-09-25-catvote"
MENU_PATH = Path(__file__).resolve().parents[1] / "taxonomy_leaves.json"
EXPAND_RATIO = 0.10
"""bbox 各边外延比例（bbox-crop-expansion 规范红线值，不放大）."""

MENU_PROMPT = """你是城市事件域标注员。判断裁片中的主要目标属于以下哪一类（只选一类，verdict 必须填英文键名）：
{menu}
uncertain=以上都不是，或目标不可辨/证据不足
仅输出JSON：{{"verdict":"<英文键名>","reason":"10字内"}}"""


def row_key(r: dict) -> str:
    """行键：事件|资源行|对象（断点重入与台账 id 共用）."""
    return f"{r['event']}|{r['rid']}|{r['object_id']}"


def load_menu(path: Path) -> tuple[str, frozenset[str]]:
    """taxonomy 导出件 → (prompt 菜单串, 合法 verdict 集)."""
    data = orjson.loads(path.read_bytes())
    menu = "\n".join(f"- {name}：{zh}" for name, zh in data["menu"])
    valid = frozenset({name for name, _zh in data["menu"]} | {"uncertain"})
    return menu, valid


def make_crop(r: dict, crops_dir: Path) -> Path:
    """manifest 行 → bbox±10% crop 文件（失败异常上抛，由调用方逐条落账）."""
    im = Image.open(r["img_path"]).convert("RGB")
    x1, y1, x2, y2 = r["box_norm"]
    crop = crop_expanded(
        im, (x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height),
        be_uniform(EXPAND_RATIO),
    )
    dst = crops_dir / f"{r['event'].replace('/', '_')}_r{r['rid']}_o{r['object_id']}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst, quality=90)
    return dst


@app.command()
def run(
    manifest: Annotated[Path, typer.Option(help="叶类投影 manifest")] = PREANNOT / "manifest_leaf.jsonl",
    out: Annotated[Path, typer.Option(help="投票 sidecar（追加+断点重入）")] = PREANNOT / "votes_agree_false.jsonl",
    crops_dir: Annotated[Path, typer.Option(help="裁片工作目录")] = PREANNOT / "vote_crops",
    models: Annotated[str, typer.Option(help="池成员逗号分隔")] = "qwen-flash,qwen38-local,doubao-vl,glm-flash",
    keys_from: Annotated[
        Path | None,
        typer.Option(help="只投此文件所列行键(每行一个 event|rid|object_id)"),
    ] = None,
    concurrency: Annotated[int, typer.Option()] = 6,
    limit: Annotated[int, typer.Option(help=">0 冒烟只跑 N 行")] = 0,
) -> None:
    """agree=false 行 → crop → 池投票 → role_consensus → sidecar + 隔离台账."""
    aliases = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [a for a in aliases if a not in CANDIDATES]
    if unknown:
        raise typer.BadParameter(f"未知池成员: {unknown}（合法: {list(CANDIDATES)}）")
    menu_text, valid = load_menu(MENU_PATH)
    prompt = MENU_PROMPT.format(menu=menu_text)
    rows = [r for r in read_jsonl(manifest) if r.get("agree") is False]
    if keys_from is not None:
        wanted = {
            ln.strip()
            for ln in keys_from.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        rows = [r for r in rows if row_key(r) in wanted]
        missed = wanted - {row_key(r) for r in rows}
        if missed:
            raise typer.BadParameter(
                f"keys-from 含 {len(missed)} 个键不在 agree=false 行中(示例 {sorted(missed)[:3]})"
            )
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise typer.BadParameter("agree=false 行为空")
    done: set[str] = set()
    if out.exists():
        for line in out.open():
            if line.strip():
                done.add(str(orjson.loads(line)["key"]))
    todo = [r for r in rows if row_key(r) not in done]
    logger.info(f"agree=false 总 {len(rows)} | 已投 {len(done)} | 待投 {len(todo)} | 池 {aliases}")

    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, r: dict) -> dict:
        key = row_key(r)
        base = {"key": key, "event": r["event"], "rid": r["rid"], "object_id": r["object_id"],
                "leaf": r.get("leaf", ""), "code": r.get("code"), "code_verif": r.get("code_verif")}
        try:
            url = image_data_url(make_crop(r, crops_dir))
        except Exception as e:  # 单样本裁切失败不卡批
            return base | {"votes": {}, "errors": [f"crop:{type(e).__name__}"],
                           "status": "img_error", "verdict": ""}
        async with sem:
            votes = await asyncio.gather(
                *[call_role_vote(client, a, url, prompt=prompt, valid=valid) for a in aliases])
        vd = {a: v for a, v, _ in votes}
        errs = [f"{a}:{reason}" for a, v, reason in votes if v in ("api_error", "parse_error")]
        ok = {a: v for a, v in vd.items() if v not in ("api_error", "parse_error")}
        status, verdict = (("abstain", "") if len(ok) < 2 else role_consensus(ok))
        return base | {"votes": vd, "errors": errs, "status": status, "verdict": verdict}

    async def _all() -> list[dict]:
        res: list[dict] = []
        async with httpx.AsyncClient() as client:
            with out.open("a") as f:
                for i in range(0, len(todo), concurrency):
                    batch = todo[i:i + concurrency]
                    batch_res = await asyncio.gather(*[one(client, r) for r in batch])
                    for br in batch_res:
                        f.write(json.dumps(br, ensure_ascii=False) + "\n")
                    f.flush()
                    res.extend(batch_res)
                    logger.info(f"{min(i + concurrency, len(todo))}/{len(todo)}")
        return res

    results = asyncio.run(_all())
    stats: Counter = Counter(r["status"] for r in results)
    verdicts: Counter = Counter(r["verdict"] for r in results if r["status"] in ("trusted", "arbited"))
    quar = [QuarantineEntry(id=f"{VOTE_BATCH}:{r['key']}", batch=VOTE_BATCH, gate="category-vote",
                            verdict=f"split votes={r['votes']}", model="vlm_pool", priority="high")
            for r in results if r["status"] == "split"]
    quar += [QuarantineEntry(id=f"{VOTE_BATCH}:{r['key']}", batch=VOTE_BATCH, gate="category-vote-crop",
                             verdict=f"{r['errors'][0]} | src={r['event']}", model="rule", priority="low")
             for r in results if r["status"] == "img_error"]
    append_quarantine(LEDGER, quar)
    typer.echo(orjson.dumps({"status": dict(stats), "auto_verdicts": dict(verdicts),
                             "manual_residual": stats["split"], "quarantined": len(quar)},
                            option=orjson.OPT_INDENT_2).decode())
    typer.secho(f"投票明细 → {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
