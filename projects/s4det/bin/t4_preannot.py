#!/usr/bin/env python3
"""T4-b1 预标注装配（用户准则 2026-09-26：人不凭空标——多模型预标 + 人编辑 + 对拍）.

vlm_pool 四票池（MODELS，三强 + glm 仲裁）对 eval-gt/t4-b1 的 225 帧逐类 grounding（每帧×2 类），
consensus_boxes 合成：
    trusted/arbited → label=类名      （共识底稿，用户复核/删改）
    low_agreement   → label=类名?     （候选框，description=来源票，待用户裁决）
    无任何框        → 不写标注文件    （空卷=负样本；用户翻过即生效）

产物（labels/ 为交互层，votes 为机器侧单一数据源——X-AnyLabeling 误覆盖不丢数据）：
    labels/<图名>.json       X-AnyLabeling v4 格式（schema 抄上游 assets/demo.json）
    preannot_votes.jsonl     每帧×每类×每模型原始归一化框（模型能力对拍数据源）

score 子命令：用户编辑完后，以 labels/ 中真实类框（`?` 后缀候选不计）为 GT，
对每模型（及池共识）算 IoU≥0.4 一对一匹配 P/R/F1——模型能力账 → 权重差异化依据。

典型用法:
    python t4_preannot.py run                # 生成预标注（断点重入；已有 json 的帧跳过=尊重人编辑）
    python t4_preannot.py score              # 用户编辑完后对拍各模型
"""

import asyncio
import json
from collections import Counter
from pathlib import Path

import httpx
import orjson
import typer
from a_common import DATA_ROOT, read_jsonl
from loguru import logger
from PIL import Image

from jxl.bin.vlm_pool import (
    CANDIDATES,
    build_target_prompt,
    call_person,
    consensus_boxes,
    greedy_iou_match,
    image_data_url,
)

app = typer.Typer(add_completion=False, help="T4-b1 预标注（多模型预标 + 人编辑 + 对拍）.")

T4 = DATA_ROOT / "eval-gt" / "t4-b1"
LABELS = T4 / "labels"
VOTES = T4 / "preannot_votes.jsonl"
MODELS = ["qwen-flash", "doubao-vl", "qwen38-local", "glm-flash"]
TARGETS = {
    "ebike": "electric bike or motorcycle",  # taxonomy v1: motorbike 并入 ebike
    "garbage_bag": "garbage bag or woven sack",
}
GT_LABELS = set(TARGETS)  # score 时算真值的真实类（`?` 后缀候选不计）
IOU_TH = 0.4
STRONG_MODELS = {a for a, s in CANDIDATES.items() if s["strength"] == "strong"}
"""候选框上屏门槛：仅强模型单票（glm 旧代 0.33 → 换代后仲裁票 0.482（T7'），单票噪声过大——留台账对拍，不上屏）。"""


def render_labels(votes_rows: list[dict]) -> int:
    """votes 台账 → labels/*.json（已有文件跳过=尊重人编辑）。返回新写文件数."""
    by_frame: dict[str, list[dict]] = {}
    for r in votes_rows:
        by_frame.setdefault(r["basename"], []).append(r)
    written = 0
    for stem, rows in by_frame.items():
        out = LABELS / f"{stem}.json"
        if out.exists():
            continue
        img = T4 / "images" / f"{stem}.jpg"
        with Image.open(img) as im:
            w, h = im.size
        shapes = []
        for r in rows:
            cls = r["cls"]
            for c in r["consensus"]:
                cand = c["status"] == "low_agreement"
                if cand and not set(c["members"]) <= STRONG_MODELS:
                    continue  # 弱模型单票不上屏（台账保留, score 照算）
                desc = ("共识: " if not cand else "候选: ") + "+".join(c["members"])
                shapes.append(_shape(cls + ("?" if cand else ""), c["box"], w, h, desc))
        if shapes:
            out.write_text(json.dumps(_xany_json(img, shapes, w, h), ensure_ascii=False))
            written += 1
    return written


def _xany_json(img: Path, shapes: list[dict], width: int, height: int) -> dict:
    """X-AnyLabeling v4 标注文件（schema 单一数据源 = 上游 assets/demo.json）."""
    return {
        "version": "4.0.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": img.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "text": "",
    }


def _shape(label: str, box: tuple, width: int, height: int, desc: str) -> dict:
    """归一化 xyxy → rectangle shape（4 角点绝对像素, TL,TR,BR,BL）."""
    x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    return {
        "label": label,
        "score": None,
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": desc,
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def _load_gt(labels_dir: Path) -> dict[tuple[str, str], list[tuple]]:
    """labels/*.json 真实类框 → {(basename, cls): [归一化 xyxy]}（`?` 候选不计）."""
    gt: dict[tuple[str, str], list[tuple]] = {}
    for f in sorted(labels_dir.glob("*.json")):
        d = json.loads(f.read_text())
        w, h = d.get("imageWidth", 0), d.get("imageHeight", 0)
        if not w or not h:
            continue
        for s in d.get("shapes", []):
            lab = str(s["label"])
            if lab not in GT_LABELS:
                continue
            (x1, y1), (_x2, _), (x3, y2), _ = s["points"]
            gt.setdefault((f.stem, lab), []).append(
                (x1 / w, y1 / h, x3 / w, y2 / h))
    return gt


@app.command()
def run(
    models: str = typer.Option(",".join(MODELS), help="池成员逗号分隔"),
    concurrency: int = typer.Option(6),
    limit: int = typer.Option(0, help=">0 冒烟只跑 N 帧"),
) -> None:
    """生成预标注：四票池逐类 grounding → 共识 → labels/*.json + votes 台账."""
    aliases = [m.strip() for m in models.split(",") if m.strip()]
    frames = read_jsonl(T4 / "manifest.jsonl")
    if limit > 0:
        frames = frames[:limit]
    LABELS.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if VOTES.exists():
        done = {f"{r['basename']}:{r['cls']}" for r in read_jsonl(VOTES)}
    # 每帧只预标被抽样的叶类（负样本帧不预标=全空卷，用户翻过即负）——
    # 跨类候选为纯噪声源；GT 局限：正样本帧内偶见的其他类目标靠用户手动补框
    todo = [
        (f, str(f["leaf"]))
        for f in frames
        if f.get("leaf") in TARGETS
        and f"{Path(f['image']).stem}:{f['leaf']}" not in done
    ]
    logger.info("帧 {} | 票任务 {}（已完成 {}）| 池 {}", len(frames), len(todo), len(done), aliases)
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, f: dict, cls: str) -> dict:
        img = Path(f["image"])
        url = image_data_url(img)
        async with sem:
            votes = await asyncio.gather(
                *[call_person(client, a, url, prompt=build_target_prompt(TARGETS[cls]))
                  for a in aliases])
        vd = {v.alias: [list(b) for b in v.boxes] for v in votes if v.ok}
        errs = [f"{v.alias}:{v.error}" for v in votes if not v.ok]
        cboxes = consensus_boxes(list(votes))
        return {"basename": img.stem, "event": f["event"], "leaf": f.get("leaf", ""),
                "cls": cls, "votes": vd, "errors": errs,
                "consensus": [{"box": list(c.box), "status": c.status,
                               "members": list(c.members)} for c in cboxes]}

    async def _all() -> list[dict]:
        res: list[dict] = []
        async with httpx.AsyncClient() as client:
            with VOTES.open("a") as f:
                for i in range(0, len(todo), concurrency):
                    batch = todo[i:i + concurrency]
                    rows = await asyncio.gather(*[one(client, f_, c) for f_, c in batch])
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    res.extend(rows)
                    logger.info("{}/{}", min(i + concurrency, len(todo)), len(todo))
        return res

    results = asyncio.run(_all())
    n_json = len(list(LABELS.glob("*.json")))
    written = render_labels(read_jsonl(VOTES))
    status = Counter(c["status"] for r in results for c in r["consensus"])
    typer.echo(orjson.dumps(
        {"votes_rows": len(results), "consensus": dict(status),
         "label_files": n_json + written, "rendered_now": written},
        option=orjson.OPT_INDENT_2).decode())
    typer.secho(f"预标注 → {LABELS}", fg=typer.colors.GREEN)


@app.command()
def score() -> None:
    """用户编辑后对拍：per-model（及池共识）vs 人工终版 GT 的 P/R/F1."""
    gt = _load_gt(LABELS)
    if not gt:
        raise typer.BadParameter(f"labels/ 无真实类标注（{GT_LABELS}）——用户尚未编辑?")
    votes_idx = {(r["basename"], r["cls"]): r for r in read_jsonl(VOTES)}
    n_gt = sum(len(v) for v in gt.values())
    rows: list[dict] = []
    for name in [*MODELS, "consensus"]:
        tp = fp = 0
        ious_all: list[float] = []
        for (stem, cls), gt_boxes in gt.items():
            row = votes_idx.get((stem, cls))
            if name == "consensus":
                pred = [tuple(c["box"]) for c in (row or {}).get("consensus", [])
                        if c["status"] in ("trusted", "arbited")]
            else:
                pred = [list(b) for b in (row or {}).get("votes", {}).get(name, [])]
            tp_m, n_pred, _n_gt, ious = greedy_iou_match(pred, gt_boxes, IOU_TH)
            tp += tp_m
            fp += n_pred - tp_m
            ious_all.extend(ious)
        fn = n_gt - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        rows.append({"model": name, "P": round(p, 3), "R": round(r_, 3), "F1": round(f1, 3),
                     "IoU_mean": round(sum(ious_all) / len(ious_all), 3) if ious_all else 0.0})
    typer.echo(orjson.dumps({"gt_boxes": n_gt, "iou_th": IOU_TH, "leaderboard": rows},
                            option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    app()
