#!/usr/bin/env python3
"""s4det M0 烟火独立线 phase 1——弱标签候选池装配.

输入 m0_scan.py 的 matches.jsonl → 命中 fire/smoke 词族的事件资源图,
symlink 扁平名 {date}_{event_id}_r{stem}.jpg(与 c_project.py 候选池同约定,
事件图原名恒为 {rid}.jpg 跨事件重名, n001 同坑)入 candidates/images/,
逐图一行 manifest.jsonl + pool_stats.json。

来源判据(README 同步): 文本侧弱标签 = 平台对象 Description / scene prompt /
llm_entries 双模型语句命中烟火词族(m0_scan.py WORD_FAMILIES)——D2 无在役
先验、mobile 域无 ET10 事件, 候选池只能弱标签投影起步, phase 2 共识挖掘
(yoloe+gdino)负责真值化。

典型用法:
    python m0_pool.py --matches /mnt/data/s4det/m0-fire/scan/matches.jsonl \
                      --out-dir /mnt/data/s4det/m0-fire/candidates
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import orjson
import typer
from loguru import logger
from pydantic import BaseModel

app = typer.Typer(add_completion=False, help="M0 phase 1: 候选池装配.")

CANDIDATE_FAMILIES = {"fire", "smoke"}


class PoolRow(BaseModel):
    """candidates/manifest.jsonl 行模型(逐图)."""

    event: str
    domain: str
    date: str
    event_id: str
    event_type: int
    type_name: str
    families: list[str]
    flat_name: str
    src_path: str


@app.command()
def main(
    matches: Annotated[Path, typer.Option(help="m0_scan.py 产物 matches.jsonl")],
    out_dir: Annotated[Path, typer.Option(help="候选池输出目录")],
    families: Annotated[str, typer.Option(help="进池词族, 逗号分隔")] = "fire,smoke",
) -> None:
    """装配候选池: images/ symlink + manifest.jsonl + pool_stats.json."""
    want = set(families.split(","))
    unknown = want - CANDIDATE_FAMILIES
    if unknown:
        raise typer.BadParameter(f"非候选词族: {unknown}")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[PoolRow] = []
    per_domain: Counter[str] = Counter()
    per_fam: Counter[str] = Counter()
    with matches.open("rb") as f:
        for line in f:
            rec = orjson.loads(line)
            hit = [x for x in rec["families"] if x in want]
            if not hit:
                continue
            per_domain[rec["domain"]] += 1
            for x in hit:
                per_fam[x] += 1
            for p in rec["image_paths"]:
                src = Path(p)
                if not src.is_file():
                    logger.warning("源图缺失(跳过): {}", src)
                    continue
                # 扁平名: .../{domain}/{date}/{event_id}/{resources|images}/{stem}.jpg
                parts = src.parts
                flat = f"{parts[-4]}_{parts[-3]}_r{src.stem}.jpg"
                dst = img_dir / flat
                if not dst.is_symlink():
                    dst.symlink_to(src)
                rows.append(
                    PoolRow(
                        event=f"{rec['date']}/{rec['event_id']}", domain=rec["domain"],
                        date=rec["date"], event_id=rec["event_id"],
                        event_type=rec["event_type"], type_name=rec["type_name"],
                        families=rec["families"], flat_name=flat, src_path=str(src),
                    )
                )
    with (out_dir / "manifest.jsonl").open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r.model_dump()) + b"\n")
    stats = {
        "batch": "2026-09-27-m0-phase1-pool",
        "matches": str(matches),
        "families": sorted(want),
        "n_events": int(sum(per_domain.values())),
        "n_images": len(rows),
        "n_events_per_domain": dict(per_domain),
        "n_events_per_family": dict(per_fam),
        "script": "m0_pool.py",
    }
    (out_dir / "pool_stats.json").write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    logger.info("候选池: {} 事件 / {} 图 → {}", stats["n_events"], len(rows), out_dir)


if __name__ == "__main__":
    app()
