#!/usr/bin/env python3
"""s4det 任务书 B①——弱标签预标注抽取器(P3 种子集进料机).

event_index.jsonl(单一数据源, a_manifest 扫描产物, 平台重复 Id 已去重) →
preannot/manifest.jsonl, 每 object 一行(任务书 B① schema):

    {"event": "<date>/<id>", "rid": "...", "object_id": "...", "box_norm": [x0,y0,x1,y1],
     "conf": 82.0, "conf_band": "hi|mid|lo", "code": "11", "code_verif": "ebike_single",
     "doubao_cls": "...", "qwen_cls": "...", "agree": true, "img_path": "...", "crop_path": null}

组装规则(CODEBOOK 已定): 位置=平台 Polygon 外接矩形; 语义=llm 双模型一致粗类;
码 11 高优先 / 码 12 降权标记; conf ≥80 hi / 50-79 mid / <50 lo(lo 不入种子直通);
crop 惰性(P3 按需裁, 外延原则同任务书 A)。agree=false 行留在 manifest 供仲裁队列,
P3 消费侧按 agree 过滤种子。
叶类映射等 taxonomy.md 冻结后另跑投影步, 本抽取器不负责。

典型用法:
    python b_preannot.py [--index /mnt/data/s4det/p0-cluster/event_index.jsonl]
"""

import hashlib
from pathlib import Path
from typing import Annotated

import orjson
import typer
from a_common import (
    DATA_ROOT,
    EventRec,
    read_jsonl,
    write_jsonl,
)
from loguru import logger
from pydantic import BaseModel

app = typer.Typer(add_completion=False, help="任务书 B①: 弱标签预标注抽取.")

CONF_HI = 80.0
"""conf_band hi 下限(任务书 B①; 与任务书 A 的分档口径不同, 勿混用)."""

CONF_MID = 50.0
"""conf_band mid 下限."""

CODE_VERIF: dict[str, str] = {
    "11": "ebike_single",  # 确证: 非机动车单辆主体框
    "12": "ambig_g12",  # 待定码降权(任务书 A 网格实证: 路缘/地面区域框, CODEBOOK 已更新)
    "6": "water_region",  # 确证(mobile): 地面积水区域框
    "10": "road_damage",  # 低确信(样本极小)
    "1": "bag_garbage",
    "2": "construction_like",
    "3": "garbage_form2",
    "4": "piled_form3",
    "5": "stall_vendor",
    "8": "shop_renovation",
    "9": "road_anomaly_region",
}
"""平台码 → 语义档(CODEBOOK 结论; 码表按域分表, 仅 mobile 域)."""


def conf_band(conf: float) -> str:
    """任务书 B① 分档: ≥80 hi / 50-79 mid / <50 lo."""
    if conf >= CONF_HI:
        return "hi"
    return "mid" if conf >= CONF_MID else "lo"


def code_verif(band: str) -> str:
    """码分档 → 语义标记(CODEBOOK; 未知数字码原样挂 code_<n>)."""
    if band in CODE_VERIF:
        return CODE_VERIF[band]
    if band == "text":
        return "text_desc"
    if band == "none":
        return "none"
    return f"code_{band}"


class PreannotRow(BaseModel):
    """预标注行(任务书 B① schema, 字段名与顺序对齐任务书示例)."""

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


def extract(index_path: Path, out_dir: Path) -> dict[str, object]:
    """索引 → 预标注 manifest + 统计."""
    recs = [EventRec(**r) for r in read_jsonl(index_path)]
    rows: list[PreannotRow] = []
    n_events_with_obj = 0
    for rec in recs:
        if not rec.objects:
            continue
        n_events_with_obj += 1
        rel = f"{rec.date}/{rec.event_id}"
        for obj in rec.objects:
            pair = rec.llm.get(obj.object_id)
            rows.append(
                PreannotRow(
                    event=rel,
                    rid="1",
                    object_id=obj.object_id,
                    box_norm=obj.bbox,
                    conf=obj.conf,
                    conf_band=conf_band(obj.conf),
                    code=obj.code,
                    code_verif=code_verif(obj.band),
                    doubao_cls=pair.doubao if pair else None,
                    qwen_cls=pair.qwen if pair else None,
                    agree=pair.agree if pair else None,
                    img_path=rec.image_path or "",
                )
            )
    write_jsonl(out_dir / "manifest.jsonl", rows)
    stats: dict[str, object] = {
        "n_events_indexed": len(recs),
        "n_events_with_objects": n_events_with_obj,
        "n_rows": len(rows),
        "agree_rate": round(sum(1 for r in rows if r.agree) / len(rows), 4),
        "agree_true": sum(1 for r in rows if r.agree is True),
        "conflict_or_missing_llm": sum(1 for r in rows if r.agree is None),
        "conf_bands": _count(rows, "conf_band"),
        "code_bands": _count(rows, "code_verif"),
        "llm_cls_top": _count(rows, "doubao_cls"),
    }
    digest = hashlib.sha256((out_dir / "manifest.jsonl").read_bytes()).hexdigest()
    (out_dir / "run_config.json").write_bytes(
        orjson.dumps(
            {
                "batch": "2026-09-25-p2-preannot",
                "manifest_sha256": digest,
                "n_rows": len(rows),
                "index": str(index_path),
                "script": "b_preannot.py",
            },
            option=orjson.OPT_INDENT_2,
        )
    )
    return stats


def _count(rows: list[PreannotRow], field: str) -> dict[str, int]:
    """行集合按字段计数(降序)."""
    out: dict[str, int] = {}
    for r in rows:
        key = str(getattr(r, field))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


@app.command()
def main(
    index: Annotated[Path, typer.Option(help="事件索引(a_manifest 产物)")] = DATA_ROOT
    / "p0-cluster"
    / "event_index.jsonl",
    out_dir: Annotated[Path, typer.Option(help="输出目录")] = DATA_ROOT / "preannot",
) -> None:
    """抽取全量预标注 manifest."""
    if not index.is_file():
        raise typer.BadParameter(f"索引不存在(先跑 a_manifest.py): {index}")
    stats = extract(index, out_dir)
    logger.info("预标注 {} 行 | agree 率 {}", stats["n_rows"], stats["agree_rate"])
    logger.info("conf_bands: {}", stats["conf_bands"])
    logger.info("code_verif: {}", stats["code_bands"])
    (out_dir / "preannot_stats.json").write_bytes(
        orjson.dumps(stats, option=orjson.OPT_INDENT_2)
    )
    logger.info("落盘 {}", out_dir)


if __name__ == "__main__":
    app()
