#!/usr/bin/env python3
"""批 2 review 帧仲裁（vlm_pool 四票 grounding + 校验器位置支持判定）.

输入: 各链 consensus/review/*.jpg（det_mine 高争议帧）+ validators.jsonl review 行
（校验器检测位置先验）。
规则（批 1 同语义，2026-09-26 设计落档 C-execution）:
    consensus_boxes 分级 → trusted/arbited 且 ≥1 校验器位置支持(IoU≥0.4)
        → 仲裁 GT → <chain>/consensus/arbitration_labels/<stem>.txt(cls=output_cls_id)
    无校验器支持(vlm_only 幻觉防线) / split / 有效票<2
        → 人工池（帧级清单, 不直通 GT）
输出:
    <chain>/consensus/arbitration_labels/   仲裁 GT(cls 保真, c_assemble 消费)
    arbitration_votes.jsonl                 每帧票据台账(断点重入 + 模型对拍)
    arbitration_human.jsonl                 人工池帧清单(含未裁决聚类框)

典型用法:
    python c_arbitrate.py --limit 6     # 冒烟
    python c_arbitrate.py               # 全量(断点重入)
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import httpx
import orjson
import typer
from loguru import logger
from vlabel.crop import xyxy_iou

from jxl.bin.vlm_pool import (
    build_target_prompt,
    call_person,
    consensus_boxes,
    image_data_url,
)
from jxl.target import load_target

app = typer.Typer(add_completion=False, help="批 2 review 帧仲裁.")

MINING = Path("/mnt/data/s4det/p3-seedset/batch2/mining")
PROFILES = Path(__file__).resolve().parents[1] / "targets"
VOTES = MINING / "arbitration_votes.jsonl"
HUMAN = MINING / "arbitration_human.jsonl"
MODELS = ["qwen-flash", "doubao-vl", "qwen38-local", "glm-flash"]
CHAIN_SRC = {
    "ebike": "vehicle_pool", "garbage_bag": "garbage_pool",
    "bicycle": "vehicle_pool", "car": "vehicle_pool", "shared_bike": "vehicle_pool",
    "tricycle": "vehicle_pool",
    "carton": "garbage_pool", "foam_box": "garbage_pool", "garbage_pile": "garbage_pool",
    "leaves_pile": "garbage_pool", "plastic_loose": "garbage_pool",
    "tarp_covered": "garbage_pool",
    "construction_fence": "vendor_pool", "piled_goods": "vendor_pool",
    "scaffolding": "vendor_pool", "shop_display": "vendor_pool",
    "stall_cart": "vendor_pool", "vendor_booth": "vendor_pool",
}
IOU_SUPPORT = 0.4
"""仲裁框须与 ≥1 校验器检出 IoU≥此值（vlm_only 幻觉防线, 批 1 语义）."""


def _write_label(path: Path, boxes: list[tuple], cls_id: int) -> None:
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]  # consensus 框为 4 元组(中位数, 无 conf)
        lines.append(f"{cls_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                     f"{x2 - x1:.6f} {y2 - y1:.6f}")
    path.write_text("\n".join(lines) + "\n")


@app.command()
def run(
    models: Annotated[str, typer.Option(help="池成员逗号分隔")] = ",".join(MODELS),
    concurrency: Annotated[int, typer.Option()] = 6,
    limit: Annotated[int, typer.Option(help=">0 冒烟只跑 N 帧")] = 0,
) -> None:
    """review 帧 → 四票 grounding → 位置支持判定 → 仲裁 GT / 人工池."""
    aliases = [m.strip() for m in models.split(",") if m.strip()]
    profs = {c: load_target(c, PROFILES / f"{c}.toml") for c in CHAIN_SRC}
    tasks: list[tuple[str, Path, str, int, list[tuple]]] = []
    for chain, _pool in CHAIN_SRC.items():
        cons = MINING / chain / "consensus"
        (cons / "arbitration_labels").mkdir(parents=True, exist_ok=True)
        priors: dict[str, list[tuple]] = {}
        for line in (cons / "validators.jsonl").open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("level") == "review":
                priors[r["stem"]] = [tuple(b) for vb in r["validators"].values() for b in vb]
        for img in sorted((cons / "review").glob("*.jpg")):
            tasks.append((chain, img, profs[chain].yolo_text,
                          profs[chain].output_cls_id, priors.get(img.stem, [])))
    if limit > 0:
        tasks = tasks[:limit]

    done: set[str] = set()
    if VOTES.exists():
        for line in VOTES.open():
            if line.strip():
                r = orjson.loads(line)
                if "chain" in r:
                    done.add(f"{r['chain']}:{r['stem']}")
    todo = [t for t in tasks if f"{t[0]}:{t[1].stem}" not in done]
    logger.info("review 帧 {}（已完成 {}）| 池 {}", len(todo), len(done), aliases)
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, chain: str, img: Path,
                  yolo_text: str, cls_id: int, val_boxes: list[tuple]) -> dict:
        url = image_data_url(img)
        async with sem:
            votes = await asyncio.gather(
                *[call_person(client, a, url, prompt=build_target_prompt(yolo_text))
                  for a in aliases])
        vd = {v.alias: [list(b) for b in v.boxes] for v in votes if v.ok}
        errs = [f"{v.alias}:{v.error}" for v in votes if not v.ok]
        cboxes = consensus_boxes(list(votes))
        gt, unsupported = [], []
        for c in cboxes:
            supported = any(xyxy_iou(c.box[:4], vb[:4]) >= IOU_SUPPORT for vb in val_boxes)
            if c.status in ("trusted", "arbited") and supported:
                gt.append(c)
            else:
                unsupported.append({"box": list(c.box), "status": c.status,
                                    "members": list(c.members)})
        if gt:
            _write_label(MINING / chain / "consensus" / "arbitration_labels" / f"{img.stem}.txt",
                         [c.box for c in gt], cls_id)
        return {"chain": chain, "stem": img.stem, "votes": vd, "errors": errs,
                "consensus": [{"box": list(c.box), "status": c.status,
                               "members": list(c.members)} for c in cboxes],
                "gt_boxes": len(gt), "unsupported": unsupported,
                "verdict": "gt" if gt else "human"}

    async def _all() -> list[dict]:
        res: list[dict] = []
        async with httpx.AsyncClient() as client:
            with VOTES.open("a") as f:
                for i in range(0, len(todo), concurrency):
                    batch = todo[i:i + concurrency]
                    rows = await asyncio.gather(*[one(client, *t) for t in batch])
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    res.extend(rows)
                    logger.info("{}/{}", min(i + concurrency, len(todo)), len(todo))
        return res

    results = asyncio.run(_all())
    stats: Counter = Counter(r["verdict"] for r in results)
    # 人工池清单自完整 VOTES 台账重建（与本次结果解耦：全量重跑/续跑不截空已有人工池）
    votes_rows = ([orjson.loads(line) for line in VOTES.open() if line.strip()]
                  if VOTES.exists() else [])
    human = [{"chain": r["chain"], "stem": r["stem"], "reason": "无支持共识框/分歧",
              "unsupported": r["unsupported"]} for r in votes_rows if r.get("verdict") == "human"]
    with HUMAN.open("wb") as f:
        f.write(b"\n".join(orjson.dumps(h) for h in human) + b"\n")
    n_labels = sum(1 for c in CHAIN_SRC
                   for _ in (MINING / c / "consensus" / "arbitration_labels").glob("*.txt"))
    typer.echo(orjson.dumps(
        {"frames": len(results), "verdict": dict(stats),
         "gt_label_files": n_labels, "human_frames": len(human)},
        option=orjson.OPT_INDENT_2).decode())
    typer.secho(f"仲裁 GT → 各链 arbitration_labels/ | 人工池 → {HUMAN}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
