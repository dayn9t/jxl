#!/usr/bin/env python3
"""T4 eval 参照集抽样装配（tasks/T4-eval-gt.md 装配侧; b1+b2）.

分层 = 叶类配额 × 尺寸段(对角线像素: small<32 / medium 32-96 / large>96,
按原图换算) × 源族; 层内随机; (日×机×3h)≤2。

- t4-b1: 叶类 ebike 120/garbage_bag 80, 源 = preannot 弱标签投影(manifest_leaf),
  种子集事件全排除; 空标负样本 25 帧; 附 workbench 轮(119 帧兜底裁决任务包)。
- t4-b2: 批 2 其余 15 类 400 帧, 源 = 批 2 种子集挖掘正本(mining/dataset/all,
  YOLO 标签)——批 2 类弱标签池已被挖掘链消费一空(merge_input 空, preannot
  投影无批 2 类), 种子集是唯一有框源。**帧与将来训练集同源 → manifest 即
  训练排除清单**(终版合并切分时按 event 排除, README 显式警示)。
  源族 = 挖掘池(vehicle/garbage/vendor/phenomenon, seedset manifest 的 pool
  字段)——批 2 帧跨五池, 比批 1 的 ET 族更有分层信息量。负样本配额本批跳过
  (eval 专用负样本池另建, 后补)。plastic_loose 存量 0 → 本批无评估源。
  images = 检测层专用 960×540 实体缩图(用户裁决 2026-09-27 分辨率门控,
  同 t4-b1 重制口径: 宽 >960 LANCZOS 等比缩 q88, 原图永不动)。

典型用法:
    python t4_sample.py                # t4-b1, 幂等, 种子固定可复现
    python t4_sample.py --batch t4-b2  # t4-b2
"""

import hashlib
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import orjson
import typer
from a_common import DATA_ROOT, read_jsonl
from loguru import logger
from PIL import Image

app = typer.Typer(add_completion=False, help="T4 eval GT 抽样装配(b1/b2).")

QUOTA = {"ebike": 120, "garbage_bag": 80}
"""t4-b1 叶类配额(T4 规范 §1; 批 1 两类 200 帧)."""

QUOTA_B2 = {
    # 400 帧派生(T4-eval-gt §1 均摊 + 存量自适应, 存量=种子集逐类有框帧数):
    # 高产 6 类均摊 255 帧 | 低产 9 类(≤47)全取 145 帧 | plastic_loose 0 无源。
    # garbage_pile 名义 40 实取 39(t4-b1 互斥排除占 1)——car +1 补足总量 400,
    # 低产类不硬凑不虚构。配额口径 = (帧,类) 对(同帧多类各计一行, 见 README)。
    "car": 44,
    "shared_bike": 43,
    "bicycle": 43,
    "tarp_covered": 42,
    "tricycle": 42,
    "carton": 41,
    "stall_cart": 47,
    "garbage_pile": 40,
    "construction_fence": 29,
    "piled_goods": 14,
    "shop_display": 5,
    "leaves_pile": 3,
    "vendor_booth": 3,
    "scaffolding": 3,
    "foam_box": 2,
}
"""t4-b2 叶类配额; 全取类 = 配额即存量(不硬凑不虚构)."""

NEG_FRAMES = 25
"""t4-b1 空标负样本帧数(200 的 12.5%, 落 10-15% 带); t4-b2 跳过(负池另建后补)."""

SEEDSET_EXCLUDE = DATA_ROOT / "p3-seedset" / "batch1" / "pools"

B2_DATASET = DATA_ROOT / "p3-seedset" / "batch2" / "mining" / "dataset"
B2_SEEDSET_MANIFEST = DATA_ROOT / "p3-seedset" / "batch2" / "seedset" / "manifest.jsonl"


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
    batch: Annotated[str, typer.Option(help="批次(t4-b1/t4-b2)")] = "t4-b1",
    index: Annotated[Path, typer.Option(help="事件索引")] = DATA_ROOT
    / "p0-cluster"
    / "event_index.jsonl",
    manifest_leaf: Annotated[
        Path, typer.Option(help="叶类投影 manifest(b1 源)")
    ] = DATA_ROOT / "preannot" / "manifest_leaf.jsonl",
    out_dir: Annotated[Path, typer.Option(help="eval-gt 输出目录")] = DATA_ROOT
    / "eval-gt"
    / "t4-b1",
    seed: Annotated[int, typer.Option(help="抽样种子")] = 42,
) -> None:
    """抽样 + 装配(b1 另附 workbench 任务包)."""
    if batch == "t4-b1":
        sample_b1(index, manifest_leaf, out_dir, seed)
    elif batch == "t4-b2":
        sample_b2(index, DATA_ROOT / "eval-gt" / "t4-b2", seed)
    else:
        raise typer.BadParameter(f"未知批次: {batch}(可选 t4-b1/t4-b2)")


def sample_b1(index: Path, manifest_leaf: Path, out_dir: Path, seed: int) -> None:
    """t4-b1: preannot 弱标签投影抽样(种子集排除) + 负样本 + workbench 轮."""
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


def sample_b2(index: Path, out_dir: Path, seed: int) -> None:
    """t4-b2: 批 2 种子集挖掘正本抽样(15 类 400 帧, manifest=训练排除清单)."""
    idx = {f"{r['date']}/{r['event_id']}": r for r in read_jsonl(index)}
    pool_of = {str(r["event"]): str(r["pool"]) for r in read_jsonl(B2_SEEDSET_MANIFEST)}
    # t4-b1 eval 帧排除集(首跑实测重合 3 帧: 批 2 池与批 1 弱标签池事件级巧合;
    # 评估集帧级互斥, 排除后源内重抽——存量富余, 不影响配额)
    b1_stems = {
        Path(str(r["image"])).stem
        for r in read_jsonl(DATA_ROOT / "eval-gt" / "t4-b1" / "manifest.jsonl")
    }
    names: dict[int, str] = {}
    for ln in (B2_DATASET / "data.yaml").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and ln[0].isdigit():
            k, v = ln.split(":", 1)
            names[int(k)] = v.strip()
    quota_cls = set(QUOTA_B2)
    absent = quota_cls - set(names.values())
    if absent:
        raise typer.BadParameter(f"配额类不在 data.yaml: {sorted(absent)}")
    # 每帧每类候选: YOLO 标签(cx cy w h) → 该类最大框对角线定尺寸段
    cand: dict[str, list[dict]] = defaultdict(list)
    for lf in sorted((B2_DATASET / "all" / "labels").glob("*.txt")):
        stem = lf.stem
        if stem in b1_stems:
            continue
        date, event_id = stem.split("_")[:2]
        ev = f"{date}/{event_id}"
        row = idx.get(ev)
        if row is None or ev not in pool_of:
            raise typer.BadParameter(f"批 2 帧缺索引/pool 映射: {ev}")
        img = B2_DATASET / "all" / "images" / f"{stem}.jpg"
        with Image.open(img) as im:
            w, h = im.size
        boxes: dict[str, float] = {}  # leaf → 该类最大框对角线
        for ln in lf.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            cid, _cx, _cy, bw, bh = ln.split()
            leaf = names[int(cid)]
            if leaf not in quota_cls:
                continue
            diag = math.hypot(float(bw) * w, float(bh) * h)
            boxes[leaf] = max(boxes.get(leaf, 0.0), diag)
        for leaf, diag in boxes.items():
            cand[leaf].append(
                {
                    "img": str(img.resolve()),
                    "stem": stem,
                    "event": ev,
                    "w": w,
                    "h": h,
                    "band": size_band(diag),
                    "family": pool_of[ev],
                    "device": str(row["device_id"]),
                    "hour3": int(str(row["hour"])) // 3,
                    "date": date,
                }
            )
    rng = random.Random(seed)
    picked: list[dict] = []
    used_groups: dict[tuple[str, str, int], int] = defaultdict(int)
    relaxed = 0  # 去密豁免补齐帧数(低产类全取可能超组限, 如实记录)
    stock_note: dict[str, int] = {}
    for leaf in QUOTA_B2:  # 按配额表序(高产→低产)消费组配额
        quota = QUOTA_B2[leaf]
        members = cand[leaf]
        stock_note[leaf] = len(members)
        full_take = len(members) <= quota
        strata: dict[tuple, list[dict]] = defaultdict(list)
        for r in members:
            strata[(r["band"], r["family"])].append(r)
        if full_take:
            alloc = {k: len(v) for k, v in strata.items()}
        else:  # 比例分配(最大余数法)
            total = sum(len(v) for v in strata.values())
            raw = {k: len(v) / total * quota for k, v in strata.items()}
            alloc = {k: int(v) for k, v in raw.items()}
            remain = quota - sum(alloc.values())
            for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[
                :remain
            ]:
                alloc[k] += 1
        skip: list[dict] = []
        got = 0
        for band_fam in sorted(strata):
            ms = strata[band_fam]
            rng.shuffle(ms)
            n = 0
            for r in ms:
                if n >= alloc[band_fam]:
                    break
                g = (r["date"], r["device"], r["hour3"])
                if used_groups[g] >= 2:
                    skip.append(r)
                    continue
                used_groups[g] += 1
                picked.append(r | {"leaf": leaf})
                n += 1
                got += 1
        for r in skip:  # 去密未取尽 → 补齐(全取类必须取满; 均摊类保配额)
            if got >= quota:
                break
            used_groups[(r["date"], r["device"], r["hour3"])] += 1
            picked.append(r | {"leaf": leaf})
            got += 1
            relaxed += 1
        if got < quota:
            logger.warning("t4-b2 {} 去密+豁免后仅 {} 帧(< 配额 {})", leaf, got, quota)
    assemble_b2(out_dir, picked, seed, stock_note, relaxed)


def write_det_frame(src: Path, dst: Path, max_width: int = 960) -> None:
    """检测层专用帧落盘(用户裁决 2026-09-27): 宽 > max_width 等比 LANCZOS 缩 + q88,
    达标字节拷贝——实体文件非 symlink; 原图永不动(下游 crop/VLM 细判用高分辨率)."""
    if dst.is_symlink():  # 旧口径(symlink 原图)清除, 换实体缩图
        dst.unlink()
    if dst.is_file():
        return
    with Image.open(src) as im:
        if im.width > max_width:
            s = max_width / im.width
            im = im.convert("RGB").resize(
                (max_width, round(im.height * s)), Image.LANCZOS
            )
            im.save(dst, format="JPEG", quality=88)
            return
    shutil.copyfile(src, dst)


def assemble_b2(
    out_dir: Path,
    picked: list[dict],
    seed: int,
    stock_note: dict[str, int],
    relaxed: int,
) -> None:
    """eval-gt/t4-b2 落盘: images 960×540 实体 + labels.txt + manifest + README 警示."""
    imgs = out_dir / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    wanted = {f"{r['stem']}.jpg" for r in picked}
    for stale in imgs.iterdir():  # 幂等: 清上次跑遗留的孤儿(symlink/实体, 文件级)
        if stale.name not in wanted:
            stale.unlink()
    manifest: list[dict] = []
    for r in picked:
        dst = imgs / f"{r['stem']}.jpg"
        write_det_frame(Path(r["img"]), dst)
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
    # labels.txt: X-AnyLabeling 标签菜单(正类 + `?` 候选两形式, 对齐 window1 修复后实践)
    classes = [c for c in QUOTA_B2 if leaf_counts.get(c)]
    menu = [ln for c in classes for ln in (c, f"{c}?")]
    (out_dir / "labels.txt").write_text("\n".join(menu) + "\n", encoding="utf-8")
    # 与 t4-b1 eval 帧互斥断言(源级已排除; 非 0 即 bug, fail fast)
    overlap = {Path(m["image"]).stem for m in manifest} & {
        Path(str(r["image"])).stem
        for r in read_jsonl(DATA_ROOT / "eval-gt" / "t4-b1" / "manifest.jsonl")
    }
    if overlap:
        raise RuntimeError(f"t4-b2 与 t4-b1 eval 帧重合 {len(overlap)} 帧(应互斥)")
    per_class = " | ".join(
        f"{c} {leaf_counts.get(c, 0)}/{stock_note[c]}" for c in QUOTA_B2
    )
    uniq_frames = len({Path(m["image"]).name for m in manifest})
    multi = len(manifest) - uniq_frames
    (out_dir / "README.md").write_text(
        f"""# T4-b2 人工 GT 参照集（装配完成，待预标注 + 人审编辑）

> **⚠️ 评测专用——禁止任何形式入训练**。本批帧源 = 批 2 种子集挖掘正本
> (p3-seedset/batch2/mining/dataset/all)——与将来训练数据同源，
> **终版合并切分(vlabel-jxl-split)时必须按本 manifest 的 event 清单排除**，
> 否则 eval 入训练 = n001 0.895 式虚高。

- manifest {len(manifest)} 行 = (帧,类) 对（同帧多类各计一行，{multi} 行为多类帧；
  唯一帧 {uniq_frames}）；负样本 0——eval 专用负池另建后补
- images = 检测层专用 960×540 实体 jpg（宽 >960 等比 LANCZOS 缩 q88；原图永不动）
- 分层: 叶类配额 × 尺寸段(small<32/medium/large, §6 small 段豁免) × 源族(挖掘池
  vehicle/garbage/vendor/phenomenon); (日×机×3h)≤2 去密, 豁免补齐 {relaxed} 帧;
  与 t4-b1 eval 帧互斥(排除 3 帧事件级巧合重合)
- 逐类(选中/存量): {per_class}; plastic_loose 存量 0 → 本批无评估源;
  garbage_pile 名义 40 实取 39(互斥排除占 1, car +1 补总量)
- 标注协议: 预标注优先(t4_preannot --batch t4-b2 四票池) + 人工编辑 = 唯一真值
- manifest sha256: {digest[:16]}… | 种子 {seed} 可复现 | 生成: projects/s4det/bin/t4_sample.py --batch t4-b2
""",
        encoding="utf-8",
    )
    logger.info(
        "T4-b2 装配 {} 帧 | 去密豁免 {} | 分布 {}",
        len(manifest),
        relaxed,
        dict(leaf_counts),
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
