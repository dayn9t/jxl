#!/usr/bin/env python3
"""批 2 挖掘产物 → 首过平铺数据集（cls 保真装配）.

b_multirun 的 dump 合并不可用于多类: vlabel 领域约定「dump 无类别 → cls 恒 0」,
所有 dump 框落 0 类; 且 consensus_dataset 要求 stem 级单源（多类同帧=毒源冲突）。
本工具把各链产物装配成**每帧单一合并标签文件**后走 --yolo 路径（cls 保留）:

    复壮链(weights 非空): L0 dump 行(target 非空)→框(cls=profile.output_cls_id)
                          + L1 labels(排除 review 帧混入——批 1 已知坑)
    无先验链(16 类):      L1 labels(排除 review; 空标签=负样本帧保留)

输出: <out>/all 平铺 + classes.txt + data.yaml（经 vlabel-consensus-dataset,
复用其近重复/空源校验）。review 帧不入——待 VLM 仲裁层（批 1 同流程）。

典型用法:
    python c_assemble.py   # 幂等(全清重建 all/)
"""

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from jxl.target import load_target

app = typer.Typer(add_completion=False, help="批 2 挖掘首过装配（cls 保真）.")

MINING = Path("/mnt/data/s4det/p3-seedset/batch2/mining")
POOLS = Path("/mnt/data/s4det/p3-seedset/batch2/seedset/pools")
PROFILES = Path(__file__).resolve().parents[1] / "targets"
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
# classes 表位次 = profile.output_cls_id(单一数据源), 顺序按位次排序生成


@app.command()
def run(
    out: Annotated[Path, typer.Option(help="数据集输出目录")] = MINING / "dataset",
) -> None:
    """装配 18 链挖掘产物 → cls 保真平铺数据集."""
    frames: dict[str, list[tuple[tuple, int]]] = defaultdict(list)  # stem → [(box, cls)]
    images: dict[str, Path] = {}
    profs = {c: load_target(c, PROFILES / f"{c}.toml") for c in CHAIN_SRC}
    order = sorted(profs, key=lambda c: profs[c].output_cls_id)

    for c in CHAIN_SRC:
        prof = profs[c]
        cons = MINING / c / "consensus"
        src = POOLS / CHAIN_SRC[c]
        review_stems = {p.stem for p in (cons / "review").glob("*.jpg")}
        n_l0 = n_l1 = 0
        if prof.weights:  # 复壮链: L0 dump 行(target 非空)按 profile cls 收框
            for line in (cons / "validators.jsonl").open():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("level") == "L0" and r.get("target"):
                    stem = r["stem"]
                    frames[stem].extend((tuple(b[:5]), prof.output_cls_id) for b in r["target"])
                    images.setdefault(stem, src / f"{stem}.jpg")
                    n_l0 += 1
        for lf in (cons / "labels").glob("*.txt"):
            if lf.stem in review_stems:
                continue  # review 帧待仲裁, 不入首过集(批 1 坑: labels/ 混含 review)
            for line in lf.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cx, cy, w, h = (float(v) for v in parts[1:5])
                cls = int(parts[0])
                if cls != prof.output_cls_id:
                    raise ValueError(
                        f"{lf}: 标签 cls {cls} != profile.output_cls_id {prof.output_cls_id}")
                frames[lf.stem].append(
                    ((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 1.0), cls))
            images.setdefault(lf.stem, src / f"{lf.stem}.jpg")
            n_l1 += 1
        # 仲裁 GT(review 帧 c_arbitrate 产出, 已过 vlm_only 幻觉防线)加回
        n_arb = 0
        for lf in (cons / "arbitration_labels").glob("*.txt"):
            for line in lf.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cx, cy, w, h = (float(v) for v in parts[1:5])
                cls = int(parts[0])
                if cls != prof.output_cls_id:
                    raise ValueError(
                        f"{lf}: 仲裁标签 cls {cls} != profile.output_cls_id {prof.output_cls_id}")
                frames[lf.stem].append(
                    ((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 1.0), cls))
            images.setdefault(lf.stem, src / f"{lf.stem}.jpg")
            n_arb += 1
        logger.info("[{}] L0 帧 {} | L1 帧 {} | 仲裁帧 {} (review 待裁 {})",
                    c, n_l0, n_l1, n_arb, len(review_stems) - n_arb)

    work = MINING / "assemble"
    shutil.rmtree(work, ignore_errors=True)
    (work / "images").mkdir(parents=True)
    (work / "labels").mkdir()
    for stem, bl in frames.items():
        lines = [
            f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
            for (x1, y1, x2, y2, _conf), cls in bl
        ]
        (work / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        (work / "images" / f"{stem}.jpg").symlink_to(images[stem])
    n_boxes = sum(len(bl) for bl in frames.values())
    logger.info("装配 {} 帧 / {} 框 → {}", len(frames), n_boxes, work)

    cmd = ["uv", "run", "--no-sync", "vlabel-consensus-dataset", str(out),
           "--yolo", f"{work / 'labels'}:{work / 'images'}",
           *[x for c in order for x in ("--classes", c)]]
    subprocess.run(cmd, check=True)
    logger.info("完成: {}（划分用 vlabel-jxl-split）", out)


if __name__ == "__main__":
    app()
