#!/usr/bin/env python3
"""T4 预标注装配（用户准则 2026-09-26：人不凭空标——多模型预标 + 人编辑 + 对拍）.

vlm_pool 四票池（MODELS，三强 + glm 仲裁）对 eval-gt/{t4-b1,t4-b2} 的帧逐类
grounding，consensus_boxes 合成：
    trusted/arbited → label=类名      （共识底稿，用户复核/删改）
    low_agreement   → label=类名?     （候选框，description=来源票，待用户裁决）
    无任何框        → 不写标注文件    （空卷=负样本；用户翻过即生效）

分辨率门控（用户裁决 2026-09-27）：检测层专用 960×540 缩图实体（样本侧已物理
重制， t4_sample 落盘时缩）——原图永不动（下游 crop/VLM 细判用高分辨率）；
--downscale 仅在样本图未达标时再缩（达标即 1），杜绝 ~10px 小目标幻觉识别。

产物（labels/ 为交互层，votes 为机器侧单一数据源——X-AnyLabeling 误覆盖不丢数据）：
    labels/<图名>.json       X-AnyLabeling v4 格式（schema 抄上游 assets/demo.json）
    preannot_votes.jsonl     每帧×每类×每模型原始归一化框（模型能力对拍数据源）

score 子命令：用户编辑完后，以 labels/ 中真实类框（`?` 后缀候选不计）为 GT，
对每模型（及池共识）算 IoU≥0.4 一对一匹配 P/R/F1——模型能力账 → 权重差异化依据。

目标描述单一数据源 = targets/<类>.toml 的 vlm_prompt（检测语义 + 「排除:」行，
garbage_bag 背包事故后用户裁决每类必带排除项）——本工具把 det_mine 链的
[0,1] bbox 契约行替换为池契约(0-1000 + persons 键)后送池。

典型用法:
    python t4_preannot.py run                       # t4-b1（断点重入；已有 json 跳过=尊重人编辑）
    python t4_preannot.py run --batch t4-b2         # t4-b2
    python t4_preannot.py score [--batch t4-b2]     # 用户编辑完后对拍各模型
"""

import asyncio
import json
import tomllib
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
    call_person,
    consensus_boxes,
    greedy_iou_match,
    image_data_url,
)

app = typer.Typer(add_completion=False, help="T4 预标注（多模型预标 + 人编辑 + 对拍）.")

TARGETS_DIR = Path("/home/jiang/cc/py/jxl/projects/s4det/targets")
"""叶类 profile 目录(vlm_prompt = 检测语义+排除项单一数据源)."""

B2_LEAVES = (
    "car",
    "shared_bike",
    "bicycle",
    "tricycle",
    "carton",
    "foam_box",
    "garbage_pile",
    "leaves_pile",
    "tarp_covered",
    "construction_fence",
    "piled_goods",
    "scaffolding",
    "shop_display",
    "stall_cart",
    "vendor_booth",
)
"""t4-b2 类序(与 t4_sample.QUOTA_B2 一致; plastic_loose 存量 0 无评估源)."""

MODELS = ["qwen-flash", "doubao-vl", "qwen38-local", "glm-flash"]

POOL_JSON = (
    "对每个目标输出边界框, 坐标归一化到 0-1000(相对图像宽高, 左上原点), "
    "格式 [x1,y1,x2,y2]。\n"
    '仅输出 JSON: {"persons": [{"bbox_2d": [x1,y1,x2,y2]}]}'
)
"""池输出契约(与 vlm_pool.PERSON_PROMPT 逐字对齐——解析层按此提框)."""


def pool_prompt(vlm_prompt: str) -> str:
    """toml vlm_prompt → 池 prompt: 剥 det_mine 的 [0,1] 契约行, 换池契约.

    检测句/要点行(含「排除:」)原样保留——语义单一数据源在 toml。
    """
    lines: list[str] = []
    for ln in vlm_prompt.strip().splitlines():
        s = ln.strip()
        if not s or s.startswith(("每个目标一个对象", "- bbox 归一化坐标")):
            continue
        if s in ("仅输出 JSON 数组。", "- 若无, 输出 []"):
            continue
        if s.startswith("检测图中"):
            s = s.split("。严格只输出")[0] + "。"
            lines.append(s)
        else:
            lines.append(s)
    if not lines or not lines[0].startswith("检测图中"):
        raise ValueError(f"vlm_prompt 无检测句(模板漂移?): {vlm_prompt[:80]!r}")
    return "\n".join(lines) + "\n" + POOL_JSON


def load_prompts(leaves: tuple[str, ...] | list[str]) -> dict[str, str]:
    """targets/<类>.toml 的 vlm_prompt → {leaf: 池 prompt}(缺失即炸)."""
    prompts: dict[str, str] = {}
    for leaf in leaves:
        with (TARGETS_DIR / f"{leaf}.toml").open("rb") as f:
            prompts[leaf] = pool_prompt(tomllib.load(f)["vlm_prompt"])
    return prompts


def batch_paths(batch: str) -> tuple[Path, dict[str, str]]:
    """批次 → (eval 目录, {leaf: 池 prompt}); labels/votes 子路径由调用方拼."""
    if batch == "t4-b1":
        return DATA_ROOT / "eval-gt" / "t4-b1", load_prompts(["ebike", "garbage_bag"])
    if batch == "t4-b2":
        return DATA_ROOT / "eval-gt" / "t4-b2", load_prompts(list(B2_LEAVES))
    raise typer.BadParameter(f"未知批次: {batch}(可选 t4-b1/t4-b2)")


IOU_TH = 0.4
STRONG_MODELS = {a for a, s in CANDIDATES.items() if s["strength"] == "strong"}
"""候选框上屏门槛：仅强模型单票（glm 旧代 0.33 → 换代后仲裁票 0.482（T7'），单票噪声过大——留台账对拍，不上屏）。"""


def render_labels(t4: Path, targets: dict[str, str], votes_rows: list[dict]) -> int:
    """votes 台账 → labels/*.json（已有文件跳过=尊重人编辑）。返回新写文件数."""
    by_frame: dict[str, list[dict]] = {}
    for r in votes_rows:
        by_frame.setdefault(r["basename"], []).append(r)
    written = 0
    for stem, rows in by_frame.items():
        out = t4 / "labels" / f"{stem}.json"
        if out.exists():
            continue
        img = t4 / "images" / f"{stem}.jpg"
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
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(_xany_json(img, shapes, w, h), ensure_ascii=False)
            )
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


def _load_gt(
    labels_dir: Path, gt_labels: set[str]
) -> dict[tuple[str, str], list[tuple]]:
    """labels/*.json 真实类框 → {(basename, cls): [归一化 xyxy]}（`?` 候选不计）."""
    gt: dict[tuple[str, str], list[tuple]] = {}
    for f in sorted(labels_dir.glob("*.json")):
        d = json.loads(f.read_text())
        w, h = d.get("imageWidth", 0), d.get("imageHeight", 0)
        if not w or not h:
            continue
        for s in d.get("shapes", []):
            lab = str(s["label"])
            if lab not in gt_labels:
                continue
            (x1, y1), (_x2, _), (x3, y2), _ = s["points"]
            gt.setdefault((f.stem, lab), []).append((x1 / w, y1 / h, x3 / w, y2 / h))
    return gt


@app.command()
def run(
    batch: str = typer.Option("t4-b1", help="批次(t4-b1/t4-b2)"),
    models: str = typer.Option(",".join(MODELS), help="池成员逗号分隔"),
    concurrency: int = typer.Option(6),
    limit: int = typer.Option(0, help=">0 冒烟只跑 N 帧"),
    downscale: int = typer.Option(
        1, help="分辨率门控: 样本图未达标时再缩 ÷N(960×540 达标即 1)"
    ),
) -> None:
    """生成预标注：四票池逐类 grounding → 共识 → labels/*.json + votes 台账."""
    t4, prompts = batch_paths(batch)
    labels = t4 / "labels"
    votes_path = t4 / "preannot_votes.jsonl"
    aliases = [m.strip() for m in models.split(",") if m.strip()]
    frames = read_jsonl(t4 / "manifest.jsonl")
    if limit > 0:
        frames = frames[:limit]
    labels.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if votes_path.exists():
        done = {f"{r['basename']}:{r['cls']}" for r in read_jsonl(votes_path)}
    # 每帧只预标被抽样的叶类（负样本帧不预标=全空卷，用户翻过即负）——
    # 跨类候选为纯噪声源；GT 局限：正样本帧内偶见的其他类目标靠用户手动补框
    todo = [
        (f, str(f["leaf"]))
        for f in frames
        if f.get("leaf") in prompts
        and f"{Path(f['image']).stem}:{f['leaf']}" not in done
    ]
    logger.info(
        "[{}] 帧 {} | 票任务 {}（已完成 {}）| 池 {} | downscale {}",
        batch,
        len(frames),
        len(todo),
        len(done),
        aliases,
        downscale,
    )
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, f: dict, cls: str) -> dict:
        img = Path(f["image"])
        url = image_data_url(img, downscale=downscale)
        async with sem:
            votes = await asyncio.gather(
                *[call_person(client, a, url, prompt=prompts[cls]) for a in aliases]
            )
        vd = {v.alias: [list(b) for b in v.boxes] for v in votes if v.ok}
        errs = [f"{v.alias}:{v.error}" for v in votes if not v.ok]
        cboxes = consensus_boxes(list(votes))
        return {
            "basename": img.stem,
            "event": f["event"],
            "leaf": f.get("leaf", ""),
            "cls": cls,
            "votes": vd,
            "errors": errs,
            "consensus": [
                {"box": list(c.box), "status": c.status, "members": list(c.members)}
                for c in cboxes
            ],
        }

    async def _all() -> list[dict]:
        res: list[dict] = []
        async with httpx.AsyncClient() as client:
            with votes_path.open("a") as f:
                for i in range(0, len(todo), concurrency):
                    chunk = todo[i : i + concurrency]
                    rows = await asyncio.gather(
                        *[one(client, f_, c) for f_, c in chunk]
                    )
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    res.extend(rows)
                    logger.info("{}/{}", min(i + concurrency, len(todo)), len(todo))
        return res

    results = asyncio.run(_all())
    n_json = len(list(labels.glob("*.json")))
    written = render_labels(t4, prompts, read_jsonl(votes_path))
    status = Counter(c["status"] for r in results for c in r["consensus"])
    typer.echo(
        orjson.dumps(
            {
                "batch": batch,
                "votes_rows": len(results),
                "consensus": dict(status),
                "label_files": n_json + written,
                "rendered_now": written,
            },
            option=orjson.OPT_INDENT_2,
        ).decode()
    )
    typer.secho(f"预标注 → {labels}", fg=typer.colors.GREEN)


@app.command()
def score(batch: str = typer.Option("t4-b1", help="批次(t4-b1/t4-b2)")) -> None:
    """用户编辑后对拍：per-model（及池共识）vs 人工终版 GT 的 P/R/F1."""
    t4, targets = batch_paths(batch)
    gt_labels = set(targets)  # 算真值的真实类（`?` 后缀候选不计）
    gt = _load_gt(t4 / "labels", gt_labels)
    if not gt:
        raise typer.BadParameter(f"labels/ 无真实类标注（{gt_labels}）——用户尚未编辑?")
    votes_idx = {
        (r["basename"], r["cls"]): r for r in read_jsonl(t4 / "preannot_votes.jsonl")
    }
    n_gt = sum(len(v) for v in gt.values())
    rows: list[dict] = []
    for name in [*MODELS, "consensus"]:
        tp = fp = 0
        ious_all: list[float] = []
        for (stem, cls), gt_boxes in gt.items():
            row = votes_idx.get((stem, cls))
            if name == "consensus":
                pred = [
                    tuple(c["box"])
                    for c in (row or {}).get("consensus", [])
                    if c["status"] in ("trusted", "arbited")
                ]
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
        rows.append(
            {
                "model": name,
                "P": round(p, 3),
                "R": round(r_, 3),
                "F1": round(f1, 3),
                "IoU_mean": round(sum(ious_all) / len(ious_all), 3)
                if ious_all
                else 0.0,
            }
        )
    typer.echo(
        orjson.dumps(
            {"batch": batch, "gt_boxes": n_gt, "iou_th": IOU_TH, "leaderboard": rows},
            option=orjson.OPT_INDENT_2,
        ).decode()
    )


if __name__ == "__main__":
    app()
