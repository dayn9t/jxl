#!/usr/bin/env python3
"""s4det 批 2 负样本池装配——终版合并切分前置件（wave1）.

证据判据（质量红线：负样本须多方「无目标」证据，非「没被链选中」）:
    帧在其池的全部挖掘链上无共识框——即 det_mine 语义下「无 ≥2 模型一致的框」:
      - level==L0 且（复壮链 target 为空）→ 无共识框（孤立检出=亚共识，不成框）
      - level==L1 且链 labels/<stem>.txt 为 0 字节 → 共识框为空
      - level==review → 不合格（仲裁 GT=正样本 / 人工池=待裁，两态都不是负样本）
    排除: 已装配 dataset 帧 / 仲裁人工池(arbitration_human) / 批 1 已训帧
    （与 c_assemble 装配口径互补：装配取正样本层，本工具取全链共识空的残余。）

采样: 域配额尽量均匀（余数给合格量最少的域），域内按月尽量均匀，
      月内按孤立检出总数升序（宁缺毋毒：越少模型看见任何东西越安全）+ seed 随机 tie-break。

产物 /mnt/data/s4det/p3-seedset/negpool/v1/:
    images/<stem>.jpg (symlink→seedset) + labels/<stem>.txt (0 字节)
    + classes.txt (对齐批 2 18 类) + manifest.jsonl (选中 250 行)
    + qualified_pool.jsonl (全量合格帧证据，selected 标记——超额帧为储备不弃置)
    + README.md

典型用法:
    /home/jiang/cc/py/jxl/.venv/bin/python c_negpool.py   # 幂等（seed 固定，重跑同选）
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

app = typer.Typer(add_completion=False, help="批 2 负样本池装配（证据驱动）.")

DATA_ROOT = Path("/mnt/data/s4det")
MINING = DATA_ROOT / "p3-seedset" / "batch2" / "mining"
SEEDSET = DATA_ROOT / "p3-seedset" / "batch2" / "seedset"
BATCH1_DS = DATA_ROOT / "p3-seedset" / "batch1" / "mining" / "dataset" / "all"
POOLS: dict[str, list[str]] = {
    "vehicle_pool": ["ebike", "bicycle", "car", "shared_bike", "tricycle"],
    "garbage_pool": [
        "garbage_bag",
        "carton",
        "foam_box",
        "garbage_pile",
        "leaves_pile",
        "plastic_loose",
        "tarp_covered",
    ],
    "vendor_pool": [
        "construction_fence",
        "piled_goods",
        "scaffolding",
        "shop_display",
        "stall_cart",
        "vendor_booth",
    ],
}
QUOTA = 250


def load_chain_rows() -> dict[str, dict[str, dict]]:
    """stem → {chain: row}（18 链 validators.jsonl 全量行）."""
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for chains in POOLS.values():
        for c in chains:
            path = MINING / c / "consensus" / "validators.jsonl"
            for line in path.read_text().splitlines():
                r = json.loads(line)
                rows[r["stem"]][c] = r
    return rows


def raw_det_count(row: dict) -> int:
    """帧在该链的孤立检出总数（target+全部校验器，未过共识门）."""
    n = len(row["target"])
    for v in row["validators"].values():
        n += len(v)
    return n


def chain_negative(row: dict, chain: str) -> bool:
    """链级无共识框判据（正样本判据的补集，见模块 docstring）."""
    if row["level"] == "review":
        return False  # 仲裁 GT=正 / 人工池=待裁
    if row["level"] == "L0":
        return not row["target"]  # 复壮链 L0 target 非空=确证 GT
    lb = MINING / chain / "consensus" / "labels" / f"{row['stem']}.txt"
    return lb.is_file() and lb.stat().st_size == 0


def month_quota(sizes: dict[str, int], quota: int) -> dict[str, int]:
    """月度尽量均匀配额：每轮 1 个名额给填充率 assign/size 最低且有余量的月（确定性）."""
    months = sorted(sizes)
    assign = {m: 0 for m in months}
    remaining = quota
    while remaining > 0:
        cands = [m for m in months if assign[m] < sizes[m]]
        if not cands:
            break
        best = min(cands, key=lambda m: (assign[m] / sizes[m], months.index(m)))
        assign[best] += 1
        remaining -= 1
    return assign


@app.command()
def main(
    out_dir: Annotated[Path, typer.Option(help="输出目录")] = DATA_ROOT
    / "p3-seedset"
    / "negpool"
    / "v1",
    seed: Annotated[int, typer.Option(help="抽样种子")] = 42,
) -> None:
    """装配负样本池 v1（证据驱动 + 分层采样）."""
    rng = random.Random(seed)
    rows = load_chain_rows()

    # 排除集
    ds_stems = {p.stem for p in (MINING / "dataset").rglob("*.jpg")}
    human_stems = {
        json.loads(line)["stem"]
        for line in (MINING / "arbitration_human.jsonl").read_text().splitlines()
    }
    b1_stems = {p.stem for p in (BATCH1_DS / "images").glob("*.jpg")}
    logger.info(
        "排除集: dataset={} 人工池={} 批1已训={}",
        len(ds_stems),
        len(human_stems),
        len(b1_stems),
    )

    # 合格帧（全链无共识框）
    qual: dict[str, list[tuple[str, int]]] = {}  # pool → [(stem, raw_det_total)]
    reject: Counter = Counter()
    for pool, chains in POOLS.items():
        stems = sorted(p.stem for p in (SEEDSET / "pools" / pool).glob("*.jpg"))
        picked: list[tuple[str, int]] = []
        for s in stems:
            if s in ds_stems:
                reject[f"{pool}:in_dataset"] += 1
                continue
            if s in human_stems:
                reject[f"{pool}:human_pending"] += 1
                continue
            if s in b1_stems:
                reject[f"{pool}:batch1_trained"] += 1
                continue
            crown = rows.get(s, {})
            if any(c not in crown for c in chains):
                reject[f"{pool}:no_row"] += 1
                continue
            if all(chain_negative(crown[c], c) for c in chains):
                picked.append((s, sum(raw_det_count(crown[c]) for c in chains)))
            else:
                reject[f"{pool}:has_consensus"] += 1
        qual[pool] = picked
    total_qual = sum(len(v) for v in qual.values())
    logger.info(
        "合格负样本候选: {} {}", {k: len(v) for k, v in qual.items()}, dict(reject)
    )
    if total_qual < QUOTA:
        raise typer.BadParameter(f"合格帧 {total_qual} < 配额 {QUOTA}, 不硬凑")

    # 域配额：均分 + 余数给合格量最少的域
    names = sorted(qual, key=lambda k: len(qual[k]))
    base, rem = divmod(QUOTA, len(names))
    domain_quota = {p: base for p in names}
    for p in names[:rem]:
        domain_quota[p] += 1
    logger.info("域配额: {}", domain_quota)

    selected: list[tuple[str, str, int]] = []  # (pool, stem, raw_det_total)
    for pool, picked in qual.items():
        by_month: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for s, n in picked:
            by_month[s[:7]].append((s, n))
        mq = month_quota({m: len(v) for m, v in by_month.items()}, domain_quota[pool])
        for m in sorted(mq):
            cands = sorted(by_month[m], key=lambda t: (t[1], rng.random()))
            take = cands[: mq[m]]
            selected.extend((pool, s, n) for s, n in take)
            logger.info("[{}] {} 月配额 {} / 候选 {}", pool, m, mq[m], len(cands))

    # 落盘
    for sub in ("images", "labels"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    sel_stems = {s for _, s, _ in selected}
    qual_rows = []
    for pool, picked in qual.items():
        for s, n in picked:
            qual_rows.append(
                {
                    "stem": s,
                    "pool": pool,
                    "raw_det_total": n,
                    "selected": s in sel_stems,
                }
            )
    (out_dir / "qualified_pool.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in qual_rows) + "\n"
    )
    manifest = []
    for pool, s, n in selected:
        link = out_dir / "images" / f"{s}.jpg"
        if not link.is_symlink():
            link.symlink_to(SEEDSET / "images" / f"{s}.jpg")
        lab = out_dir / "labels" / f"{s}.txt"
        lab.touch(exist_ok=True)
        crown = rows[s]
        manifest.append(
            {
                "stem": s,
                "pool": pool,
                "month": s[:7],
                "raw_det_total": n,
                "chains": {
                    c: {"level": crown[c]["level"], "raw": raw_det_count(crown[c])}
                    for c in POOLS[pool]
                },
            }
        )
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n"
    )
    classes = (MINING / "dataset" / "classes.txt").read_text()
    (out_dir / "classes.txt").write_text(classes)
    write_readme(out_dir, seed, len(selected), total_qual, domain_quota)
    logger.info("完成: {} 帧选中 / {} 帧合格 → {}", len(selected), total_qual, out_dir)


def write_readme(
    out_dir: Path, seed: int, n_sel: int, n_qual: int, domain_quota: dict[str, int]
) -> None:
    """生成目录 README（j-dir-readme：目的/来源/产生命令/归属/状态）."""
    text = f"""# negpool v1 —— 批 2 负样本池（终版合并切分前置件）

- **目的**：为 M1 v0.1 终版合并切分提供 {n_sel} 帧负样本（无目标事件图教「没有」，annotation-guideline §10）。
- **来源判据**（多方无目标证据，非「没被链选中」）：帧在其池的**全部**挖掘链
  （vehicle 5 链 / garbage 7 链 / vendor 6 链）上均无共识框——det_mine 语义「无 ≥2 模型一致的框」
  （L0 无 target 确证 ∪ L1 空标签；孤立检出=亚共识不成框）。排除：已装配 dataset 帧、
  仲裁人工池 58、批 1 已训帧。合格候选共 {n_qual} 帧，本版采样 {n_sel}。
- **采样**：域配额尽量均匀 {domain_quota}，域内按月尽量均匀，月内按孤立检出总数升序
  （宁缺毋毒）+ 随机 tie-break，seed={seed}（幂等复现）。
- **产生命令**：`/home/jiang/cc/py/jxl/.venv/bin/python /home/jiang/cc/py/jxl/projects/s4det/bin/c_negpool.py`
- **操作归属**：jxl 会话 s4det（wave1，2026-09-27）；执行记录 `projects/s4det/tasks/wave1-negpool.md`。
- **状态**：已完成（本目录为终态产物；超额合格帧见 qualified_pool.jsonl 为储备，未弃置）。
- **消费方**：终版合并切分（consensus 合并时并入 images+labels；classes 对齐批 2 18 类）。
"""
    (out_dir / "README.md").write_text(text)


if __name__ == "__main__":
    app()
