#!/usr/bin/env python3
"""s4det 任务书 C 步骤 1——叶类投影(preannot → manifest_leaf.jsonl + 候选帧池).

锚点(taxonomy.md v1-暂定冻结, 两族并存: EventType 码 + llm 类别串):
    ebike       = ET2 或 llm 类别串"非机动车乱停"
    garbage_bag = ET3
第一批只投影这两类, 其余行 leaf=""(后续批次)。leaf 判定优先级: llm 一致类别 >
宿主 EventType(CODEBOOK: 语义=llm 类别, 码/事件仅先验); 双族锚命中不同叶类 =
投影歧义 → 不硬投影, 入隔离台账转 s4 会话裁决。

直通候选池(任务书 C): 投影命中 ∧ agree=true ∧ conf_band≠lo ∧ 非 ambig_g12。
候选帧池 = 直通候选帧 + 该类 conf=hi 帧抽样(池为帧级去重, cap 见 POOL_CAP)。

典型用法:
    python c_project.py [--pool-cap 800]
"""

import hashlib
import random
from pathlib import Path
from typing import Annotated

import orjson
import typer
from a_common import (
    DATA_ROOT,
    LEDGER,
    QuarantineEntry,
    append_quarantine,
    read_jsonl,
    write_jsonl,
)
from loguru import logger
from pydantic import BaseModel

app = typer.Typer(add_completion=False, help="任务书 C 步骤 1: 叶类投影.")

BATCH = "2026-09-25-p3-batch1"
"""第一批批次号."""

# 批 1 投影锚(taxonomy.md v1-暂定冻结; 冻结扩展时在此登记新类)
LEAF_ANCHORS: dict[str, dict[str, object]] = {
    "ebike": {"event_types": {2}, "llm_classes": {"非机动车乱停"}},
    "garbage_bag": {"event_types": {3}, "llm_classes": set()},
}
"""叶类 → 双族锚; llm_classes 为空表示该叶类仅由 EventType 锚定."""

POOL_CAP = 800
"""每类候选帧池上限(直通优先, conf-hi 抽样补足; 机制验证批规模)."""


class LeafRow(BaseModel):
    """manifest_leaf 行 = B① 行 + leaf 投影."""

    event: str
    rid: str
    object_id: str
    box_norm: list[float]
    conf: float
    conf_band: str
    code: str
    code_verif: str
    doubao_cls: str | None = None
    qwen_cls: str | None = None
    agree: bool | None = None
    img_path: str
    crop_path: str | None = None
    leaf: str = ""
    anchor: str = ""
    direct: bool = False
    """直通候选: agree ∧ conf≠lo ∧ 非 ambig_g12."""


def project(row: dict[str, object], etype: int) -> tuple[str, str]:
    """单行 → (leaf, anchor 命中描述); 歧义返回 ("", "ambiguous")(不硬投影)."""
    llm_hits = {
        leaf
        for leaf, spec in LEAF_ANCHORS.items()
        if row.get("doubao_cls") in spec["llm_classes"]
        or row.get("qwen_cls") in spec["llm_classes"]
    }
    et_hits = {
        leaf
        for leaf, spec in LEAF_ANCHORS.items()
        if etype in spec["event_types"]  # type: ignore[operator]
    }
    if len(llm_hits) > 1 or (llm_hits and et_hits and llm_hits != et_hits):
        return "", "ambiguous"
    hits = llm_hits | et_hits
    if not hits:
        return "", ""
    leaf = sorted(hits)[0]
    src = "llm" if llm_hits else "etype"
    return leaf, f"{src}:{leaf}"


@app.command()
def main(
    index: Annotated[Path, typer.Option(help="事件索引(event_type 来源)")] = DATA_ROOT
    / "p0-cluster"
    / "event_index.jsonl",
    preannot: Annotated[Path, typer.Option(help="B① 预标注 manifest")] = DATA_ROOT
    / "preannot"
    / "manifest.jsonl",
    out_dir: Annotated[Path, typer.Option(help="输出目录")] = DATA_ROOT / "preannot",
    pool_dir: Annotated[Path, typer.Option(help="候选帧池输出目录")] = DATA_ROOT
    / "p3-seedset"
    / "batch1"
    / "pools",
    pool_cap: Annotated[int, typer.Option(help="每类帧池上限")] = POOL_CAP,
    seed: Annotated[int, typer.Option(help="抽样种子")] = 42,
) -> None:
    """投影 + 直通筛选 + 帧池构建."""
    etype_of = {
        f"{r['date']}/{r['event_id']}": r["event_type"] for r in read_jsonl(index)
    }
    rows_in = read_jsonl(preannot)
    quar: list[QuarantineEntry] = []
    leaves: list[LeafRow] = []
    n_ambig = 0
    for row in rows_in:
        key = str(row["event"])
        leaf, anchor = project(row, int(etype_of.get(key, 0)))  # type: ignore[arg-type]
        if anchor.startswith("ambig"):
            n_ambig += 1
            quar.append(
                QuarantineEntry(
                    id=f"{BATCH}:{key}#{row['object_id']!s}",
                    batch=BATCH,
                    gate="project-ambiguous",
                    verdict=f"双族锚命中不同叶类 {anchor}, 不硬投影转 s4 裁决",
                    model="rule",
                )
            )
        direct = bool(
            leaf
            and row.get("agree") is True
            and row.get("conf_band") != "lo"
            and row.get("code_verif") != "ambig_g12"
        )
        leaves.append(
            LeafRow(
                **{
                    k: row.get(k)
                    for k in LeafRow.model_fields
                    if k not in ("leaf", "anchor", "direct")
                },  # type: ignore[arg-type]
                leaf=leaf,
                anchor=anchor,
                direct=direct,
            )
        )
    out_path = out_dir / "manifest_leaf.jsonl"
    write_jsonl(out_path, leaves)
    append_quarantine(LEDGER, quar)

    # 帧池: 直通帧优先, conf-hi 帧抽样补足(帧级去重); 总量超 cap 时按种子随机
    # 取直通子集(无偏, 确定性)——批 1 为机制验证批, 池规模可控
    rng = random.Random(seed)
    pools: dict[str, dict[str, object]] = {}
    for leaf in LEAF_ANCHORS:
        direct_frames = sorted(
            {r.img_path for r in leaves if r.leaf == leaf and r.direct}
        )
        if len(direct_frames) > pool_cap:
            direct_frames = rng.sample(direct_frames, pool_cap)
        hi_rows = [
            r for r in leaves if r.leaf == leaf and r.conf_band == "hi" and not r.direct
        ]
        hi_frames = sorted({r.img_path for r in hi_rows})
        rng.shuffle(hi_frames)
        fill = max(pool_cap - len(direct_frames), 0)
        pool_frames = direct_frames + hi_frames[:fill]
        (pool_dir / leaf).mkdir(parents=True, exist_ok=True)
        for p in pool_frames:
            # 唯一扁平名: 事件图原名恒为 {rid}.jpg(跨事件重名, n001 同坑)——
            # 用 {date}_{event_id}_r{rid}.jpg 保证唯一且可回溯事件目录
            parts = Path(p).parts
            flat = f"{parts[-4]}_{parts[-3]}_r{Path(p).stem}.jpg"
            dst = pool_dir / leaf / flat
            if not dst.is_symlink():
                dst.symlink_to(p)
        pools[leaf] = {
            "n_direct_frames": len(direct_frames),
            "n_hi_fill": min(fill, len(hi_frames)),
            "n_pool": len(pool_frames),
            "cap": pool_cap,
        }
        logger.info(
            "[{}] 直通帧 {} + hi 补 {} = 池 {}",
            leaf,
            len(direct_frames),
            min(fill, len(hi_frames)),
            len(pool_frames),
        )
    stats = {
        "batch": BATCH,
        "n_rows": len(leaves),
        "n_projected": sum(1 for r in leaves if r.leaf),
        "n_ambiguous": n_ambig,
        "by_leaf": {
            leaf: {
                "rows": sum(1 for r in leaves if r.leaf == leaf),
                "direct_rows": sum(1 for r in leaves if r.leaf == leaf and r.direct),
                **pools[leaf],
            }
            for leaf in LEAF_ANCHORS
        },
    }
    (out_dir / "projection_stats.json").write_bytes(
        orjson.dumps(stats, option=orjson.OPT_INDENT_2)
    )
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (out_dir / "leaf_run_config.json").write_bytes(
        orjson.dumps(
            {
                "batch": BATCH,
                "manifest_leaf_sha256": digest,
                "anchors": {
                    k: {
                        kk: sorted(vv) if isinstance(vv, set) else vv
                        for kk, vv in v.items()
                    }
                    for k, v in LEAF_ANCHORS.items()
                },
                "seed": seed,
            },
            option=orjson.OPT_INDENT_2,
        )
    )
    logger.info("投影 {} / 歧义 {} → {}", stats["n_projected"], n_ambig, out_path)


if __name__ == "__main__":
    app()
