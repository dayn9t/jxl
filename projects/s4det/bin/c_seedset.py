#!/usr/bin/env python3
"""s4det 批 2 全局种子集装配——~5,000 帧（含批 1 已入 1,600），锚组配额+双重去重.

锚组（粗类族，细类归属由批 2 挖掘的类专用校验器完成——弱标签粒度不可细分）：
    vehicle_pool  ET1+2   1,000
    garbage_pool  ET3+11  1,000
    vendor_pool   ET6+7+12 900
    phenomenon_pool ET14+17+18 400
    negative（无锚事件帧）250
去重：每事件 ≤1 帧、(日,机,3h) ≤2、组内 DINO cos≥0.95 近重复剔除。

典型用法:
    python c_seedset.py  # 幂等
"""

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import orjson
import typer
from a_common import DATA_ROOT, read_jsonl
from loguru import logger
from PIL import Image

app = typer.Typer(add_completion=False, help="批 2 全局种子集装配.")

POOL_QUOTA = {
    "vehicle_pool": {"ets": {1, 2}, "quota": 1000},
    "garbage_pool": {"ets": {3, 11}, "quota": 1000},
    "vendor_pool": {"ets": {6, 7, 12}, "quota": 900},
    "phenomenon_pool": {"ets": {14, 17, 18}, "quota": 400},
}
NEG_FRAMES = 250
OVERSAMPLE = 1.15
PER_GROUP_CAP = 8
"""(日,机,3h) 同组帧上限. mobile 巡查编组制: ET1/2 全部 14,049 事件仅落在
143 个组(每天约 2 个组出动), 组内=沿途不同目标——cap 过小即全局干涸
(cap=2 实测容量 286, 2026-09-25); 8×143=1,144 满足单轮预算, 真重复仍由语义去重兜底."""


def build_candidates(
    idx: dict[str, dict], exclude_events: set[str]
) -> dict[str, list[dict]]:
    """事件索引 → 锚组候选帧（每事件 1 帧，原图在位）."""
    cand: dict[str, list[dict]] = defaultdict(list)
    seen_events: set[str] = set()
    for ev, r in idx.items():
        eid = ev.split("/")[1]
        if eid in exclude_events or ev in seen_events:
            continue
        ets = {int(r.get("event_type", 0))}
        img = r.get("image_path")
        if not img or not Path(img).is_file():
            continue
        for pool, spec in POOL_QUOTA.items():
            if ets & spec["ets"]:
                cand[pool].append(
                    {
                        "img": str(img),
                        "event": ev,
                        "pool": pool,
                        "date": ev.split("/")[0],
                        "device": str(r.get("device_id", "")),
                        "hour3": int(r.get("hour", 0)) // 3,
                    }
                )
                seen_events.add(ev)
                break
    return cand


def pick(
    frames: list[dict], quota: int, rng: random.Random, used_groups: dict
) -> list[dict]:
    """组内随机选择，(日,机,3h) ≤2，事件 ≤1."""
    rng.shuffle(frames)
    picked: list[dict] = []
    for f in frames:
        if len(picked) >= math.ceil(quota * OVERSAMPLE):
            break
        g = (f["date"], f["device"], f["hour3"])
        if used_groups[g] >= PER_GROUP_CAP:
            continue
        used_groups[g] += 1
        picked.append(f)
    return picked


def semantic_dedup(frames: list[dict], emb_path: Path) -> list[dict]:
    """组内 DINO cos≥0.95 贪心去重（embedding 缓存于 emb_path）."""
    import faiss
    import torch
    from transformers import AutoImageProcessor, AutoModel

    files = [f["img"] for f in frames]
    if emb_path.is_file():
        emb = np.load(emb_path)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        net = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(files), 64):
                imgs = []
                for p in files[i : i + 64]:
                    with Image.open(p) as im:
                        imgs.append(im.convert("RGB"))
                inp = proc(images=imgs, return_tensors="pt").to(dev)
                out = net(**inp).last_hidden_state[:, 0]
                chunks.append(torch.nn.functional.normalize(out, dim=-1).cpu())
                if (i // 64) % 10 == 0:
                    logger.info("embed {}/{}", min(i + 64, len(files)), len(files))
        emb = torch.cat(chunks).numpy().astype(np.float32)
        np.save(emb_path, emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    _, nbrs = index.search(emb, 16)
    suppressed: set[int] = set()
    keep: list[int] = []
    for i in range(len(files)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in nbrs[i][1:]:
            if j >= 0 and float(emb[i] @ emb[int(j)]) >= 0.95:
                suppressed.add(int(j))
    kept = [frames[i] for i in keep]
    logger.info("语义去重: {} → {}", len(frames), len(kept))
    return kept


@app.command()
def main(
    index: Annotated[Path, typer.Option(help="事件索引")] = DATA_ROOT
    / "p0-cluster"
    / "event_index.jsonl",
    out_dir: Annotated[Path, typer.Option(help="种子集输出目录")] = DATA_ROOT
    / "p3-seedset"
    / "batch2"
    / "seedset",
    seed: Annotated[int, typer.Option(help="抽样种子")] = 42,
) -> None:
    """装配批 2 种子集（锚组配额 + 结构/语义双重去重）."""
    rng = random.Random(seed)
    (out_dir / "work").mkdir(parents=True, exist_ok=True)  # embedding 缓存目录前置(np.save 不建父目录)
    idx = {f"{r['date']}/{r['event_id']}": r for r in read_jsonl(index)}
    exclude_events = {
        p.stem.split("_")[2]  # {leaf}_{date}_{event_id}_r{rid} → 取 event_id 段
        for p in (DATA_ROOT / "p3-seedset" / "batch1" / "pools").rglob("*.jpg")
    }
    cand = build_candidates(idx, exclude_events)

    used_groups: dict[tuple, int] = defaultdict(int)
    picked: list[dict] = []
    for pool, spec in POOL_QUOTA.items():
        got = pick(cand.get(pool, []), spec["quota"], rng, used_groups)
        # 超采候选先做语义去重，再截到配额
        got = semantic_dedup(got, out_dir / "work" / f"dino_{pool}.npy")[
            : spec["quota"]
        ]
        picked.extend(got)
        logger.info("[{}] 配额 {} → 取 {}", pool, spec["quota"], len(got))

    # 负样本
    pos_events = {f["event"] for f in picked} | exclude_events
    neg_pool = [
        (ev, r)
        for ev, r in idx.items()
        if ev.split("/")[1] not in exclude_events
        and ev not in pos_events
        and r.get("image_path")
    ]
    rng.shuffle(neg_pool)
    negs = 0
    for ev, r in neg_pool:
        g = (ev.split("/")[0], str(r.get("device_id", "")), int(r.get("hour", 0)) // 3)
        if used_groups[g] >= 2:
            continue
        used_groups[g] += 1
        picked.append(
            {
                "img": str(r["image_path"]),
                "event": ev,
                "pool": "negative",
                "date": ev.split("/")[0],
                "device": str(r.get("device_id", "")),
                "hour3": int(r.get("hour", 0)) // 3,
            }
        )
        negs += 1
        if negs >= NEG_FRAMES:
            break

    imgs = out_dir / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    manifest = []
    for f in picked:
        p = Path(f["img"])
        parts = p.parts
        flat = f"{parts[-4]}_{parts[-3]}_r{p.stem}.jpg"
        dst = imgs / flat
        if not dst.is_symlink():
            dst.symlink_to(p)
        manifest.append(
            {
                "image": str(dst),
                "event": f["event"],
                "pool": f["pool"],
            }
        )
    (out_dir / "manifest.jsonl").write_bytes(
        b"\n".join(orjson.dumps(m) for m in manifest) + b"\n"
    )
    # 锚组池视图: pools/<pool>/ 扁平 symlink(det_mine 类链 src; manifest 为单一数据源)
    pools_root = out_dir / "pools"
    for m in manifest:
        pd = pools_root / m["pool"]
        pd.mkdir(parents=True, exist_ok=True)
        dst = pd / Path(m["image"]).name
        if not dst.is_symlink():
            dst.symlink_to(Path(m["image"]).resolve())
    by_pool = defaultdict(int)
    for m in manifest:
        by_pool[m["pool"]] += 1
    stats = {
        "budget": 5000,
        "batch1_committed": 1600,
        "new_frames": len(picked),
        "by_pool": dict(sorted(by_pool.items())),
        "seed": seed,
    }
    (out_dir / "seedset_stats.json").write_bytes(
        orjson.dumps(stats, option=orjson.OPT_INDENT_2)
    )
    logger.info(
        "种子集新增 {}（全局 {}）| {}", len(picked), 1600 + len(picked), dict(by_pool)
    )


if __name__ == "__main__":
    app()
