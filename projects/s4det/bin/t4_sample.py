#!/usr/bin/env python3
"""T4-b1——人工 GT 参照集抽样装配（tasks/T4-eval-gt.md 装配侧）.

分层 = 叶类(配额 ebike 120/garbage_bag 80) × 尺寸段(对角线像素: small<32 /
medium 32-96 / large>96, 按原图换算) × 源族(ET2 族/ET3 族/其他)。
抽样: 种子集事件全排除(按 event_id); 层内随机; (日×机×3h)≤2; 空标负样本
帧占 10-15%。产出 {data_root}/eval-gt/t4-b1/(images+manifest+README) 与
workbench 轮(119 帧兜底裁决任务包)。

典型用法:
    python t4_sample.py  # 幂等, 种子固定可复现
"""

import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import orjson
import typer
from a_common import DATA_ROOT, read_jsonl
from loguru import logger
from PIL import Image

app = typer.Typer(add_completion=False, help="T4-b1 eval GT 抽样装配.")

QUOTA = {"ebike": 120, "garbage_bag": 80}
"""叶类配额(T4 规范 §1; 批 1 两类 200 帧)."""

NEG_FRAMES = 25
"""空标负样本帧数(200 的 12.5%, 落 10-15% 带)."""

SEEDSET_EXCLUDE = DATA_ROOT / "p3-seedset" / "batch1" / "pools"


def size_band(diag_px: float) -> str:
    """T4 尺寸段: small<32 / medium 32-96 / large>96(对角线像素)."""
    if diag_px < 32:
        return "small"
    return "medium" if diag_px <= 96 else "large"


def source_family(etype: int) -> str:
    """源族: ET2 族/ET3 族/其他(防单一来源偏)."""
    if etype == 2:
        return "et2"
    return "et3" if etype == 3 else "other"


@app.command()
def main(
    index: Annotated[Path, typer.Option(help="事件索引")] = DATA_ROOT
    / "p0-cluster"
    / "event_index.jsonl",
    manifest_leaf: Annotated[Path, typer.Option(help="叶类投影 manifest")] = DATA_ROOT
    / "preannot"
    / "manifest_leaf.jsonl",
    out_dir: Annotated[Path, typer.Option(help="eval-gt 输出目录")] = DATA_ROOT
    / "eval-gt"
    / "t4-b1",
    seed: Annotated[int, typer.Option(help="抽样种子")] = 42,
) -> None:
    """抽样 + 装配 + workbench 任务包."""
    idx = {f"{r['date']}/{r['event_id']}": r for r in read_jsonl(index)}
    seedset_events = {
        "_".join(p.stem.split("_")[:2]) for p in SEEDSET_EXCLUDE.rglob("*.jpg")
    }
    rows = [r for r in read_jsonl(manifest_leaf) if r.get("leaf") in QUOTA]
    cand: dict[str, list[dict]] = defaultdict(list)
    imgs_wh: dict[str, tuple[int, int]] = {}
    for r in rows:
        ev = str(r["event"])
        if ev.split("/")[1] in seedset_events:
            continue
        img = str(r["img_path"])
        if img not in imgs_wh:
            with Image.open(img) as im:
                imgs_wh[img] = im.size
        w, h = imgs_wh[img]
        x1, y1, x2, y2 = r["box_norm"]
        diag = math.hypot((x2 - x1) * w, (y2 - y1) * h)
        etype = int(idx.get(ev, {}).get("event_type", 0))  # type: ignore[call-overload]
        cand[r["leaf"]].append(
            {
                "img": img,
                "event": ev,
                "w": w,
                "h": h,
                "band": size_band(diag),
                "family": source_family(etype),
                "device": str(idx.get(ev, {}).get("device_id", "")),  # type: ignore[call-overload]
                "hour3": int(str(idx.get(ev, {}).get("hour", 0))) // 3,  # type: ignore[call-overload]
                "date": ev.split("/")[0],
            }
        )
    rng = random.Random(seed)
    picked: list[dict] = []
    used_groups: dict[tuple[str, str, int], int] = defaultdict(int)
    for leaf, quota in QUOTA.items():
        strata: dict[tuple, list[dict]] = defaultdict(list)
        for r in cand[leaf]:
            strata[(r["band"], r["family"])].append(r)
        # 比例分配(最大余数法), 层内随机, 相机去过密
        total = sum(len(v) for v in strata.values())
        raw = {k: len(v) / total * quota for k, v in strata.items()}
        alloc = {k: int(v) for k, v in raw.items()}
        remain = quota - sum(alloc.values())
        for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[:remain]:
            alloc[k] += 1
        for (band, family), members in sorted(strata.items()):
            rng.shuffle(members)
            got = 0
            for r in members:
                if got >= alloc[(band, family)]:
                    break
                g = (r["date"], r["device"], r["hour3"])
                if used_groups[g] >= 2:
                    continue
                used_groups[g] += 1
                picked.append(r | {"leaf": leaf, "band": band, "family": family})
                got += 1
    # 空标负样本: 两类都无投影行的事件帧(且不在种子集)
    pos_events = {r["event"] for r in rows}
    neg_pool = [
        ev
        for ev, r in idx.items()
        if ev.split("/")[1] not in seedset_events and ev not in pos_events
    ]
    rng.shuffle(neg_pool)
    negatives = 0
    for ev in neg_pool:
        r = idx[ev]
        img = Path(str(r["image_path"]))
        if not img.is_file():
            continue
        g = (ev.split("/")[0], str(r["device_id"]), int(str(r["hour"])) // 3)
        if used_groups[g] >= 2:
            continue
        used_groups[g] += 1
        with Image.open(img) as im:
            w, h = im.size
        picked.append(
            {
                "img": str(img),
                "event": ev,
                "w": w,
                "h": h,
                "band": "-",
                "family": "-",
                "device": str(r["device_id"]),
                "hour3": int(str(r["hour"])) // 3,
                "date": ev.split("/")[0],
                "leaf": "negative",
            }
        )
        negatives += 1
        if negatives >= NEG_FRAMES:
            break
    assemble(out_dir, picked, seed)
    build_workbench(
        DATA_ROOT
        / "p3-seedset"
        / "batch1"
        / "review"
        / "tasks"
        / "r01_2026-09-25_t4b1_fallback",
        rng,
    )


def assemble(out_dir: Path, picked: list[dict], seed: int) -> None:
    """eval-gt/t4-b1 落盘: images symlink + manifest + README 警示."""
    imgs = out_dir / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for r in picked:
        p = Path(r["img"])
        parts = p.parts
        flat = f"{parts[-4]}_{parts[-3]}_r{p.stem}.jpg"
        dst = imgs / flat
        if not dst.is_symlink():
            dst.symlink_to(p)
        manifest.append(
            {
                "image": str(dst),
                "event": r["event"],
                "leaf": r["leaf"],
                "size_band": r["band"],
                "source_family": r["family"],
            }
        )
    (out_dir / "manifest.jsonl").write_bytes(
        b"\n".join(orjson.dumps(m) for m in manifest) + b"\n"
    )
    digest = hashlib.sha256((out_dir / "manifest.jsonl").read_bytes()).hexdigest()
    leaf_counts: dict[str, int] = defaultdict(int)
    for m in manifest:
        leaf_counts[m["leaf"]] += 1
    (out_dir / "README.md").write_text(
        f"""# T4-b1 人工 GT 参照集（装配完成，待人审标注）

> **⚠️ 评测专用——禁止任何形式入训练**（含经由 isolation 失效的间接泄漏）。

- 帧数 {len(manifest)}（正样本 {len(manifest) - leaf_counts.get("negative", 0)} + 空标负样本 {leaf_counts.get("negative", 0)}）
- 分层: 叶类配额 {QUOTA} × 尺寸段(small<32/medium 32-96/large>96 对角线) × 源族(ET2/ET3/其他)
- 标注协议: annotation-guideline.md v1; 10% 双人一致性 ≥85% 闸(T4 规范 §3)
- manifest sha256: {digest[:16]}… | 种子 42 可复现 | 生成: projects/s4det/bin/t4_sample.py
""",
        encoding="utf-8",
    )
    logger.info("T4-b1 装配 {} 帧 | 分布 {}", len(manifest), dict(leaf_counts))


def build_workbench(round_dir: Path, rng: random.Random) -> None:
    """workbench 轮: 119 帧低共识兜底裁决任务包(人类窗口开启即用)."""
    round_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for leaf in ["ebike", "garbage_bag"]:
        p = (
            DATA_ROOT
            / "p3-seedset"
            / "batch1"
            / "mining"
            / leaf
            / "arbitration"
            / "low_agreement.jsonl"
        )
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = orjson.loads(ln)
            samples.append(
                {
                    "id": r["row"]["stem"],
                    "images": [r["row"]["image"]],
                    "meta": {"leaf": leaf},
                }
            )
    task = {
        "name": "fallback_low_agreement",
        "title": "P3 批 1 低共识帧兜底裁决",
        "intro": "VLM 仲裁无法达成共识的帧。判定画面主类并转标注；难定帧选 unsure 转人工画框。",
        "hint": "ebike=两轮电动车/摩托；garbage_bag=袋装/编织袋垃圾",
        "options": [
            {
                "key": "has_target",
                "label": "含目标,转标注",
                "desc": "画面含该类可标目标",
            },
            {"key": "empty", "label": "空帧", "desc": "无可标目标(负样本)"},
            {"key": "unsure", "label": "难定", "desc": "转 X-AnyLabeling 人工画框终裁"},
        ],
        "samples": samples,
    }
    (round_dir / "fallback_low_agreement.json").write_text(
        orjson.dumps(task, option=orjson.OPT_INDENT_2).decode(), encoding="utf-8"
    )
    logger.info("workbench 轮 {} 任务包 {} 样本", round_dir.name, len(samples))


if __name__ == "__main__":
    app()
