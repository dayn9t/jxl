#!/usr/bin/env python3
"""Label-Audit: 标注质量位置审计——静态物毒标注的签名发现.

对 YOLO 标注集全量框做按相机分组的 DBSCAN 位置聚类, 输出「静态物嫌疑簇」:
嫌疑判据 N>=min_n ∧ 跨日期数>=min_d (尺寸变异判据禁用——n001 实证 cv 0.63 的
mixed 簇被它排除导致漏检; 见 docs/research/2026-09-08-共识分层阈值数据分析.md §5).

2026-09-08 n001 毒标注审计的工具化(当时为 gencheck/audit_clusters.py 内联脚本):
柜台红玩具被 L0 弱共识当人(gdino+la 一致误检, yoloe/rfdetr 反被忽略), 1,508 毒框
污染 train/val/test 各~21%. 审计方法=签名聚类 → 簇 VLM 定性 → 漏斗交叉 → 人工终审.

警告(同报告): 簇代表 n=3 抽样 VLM 定性对 mixed 簇漏检率 88-96%, 仅可用于
全一致 object 快判; person 判定必须走第三方模型支持漏斗.

用法:
    label_audit cluster <labels_dir> <out_json> [--eps 30]  # 位置聚类审计
    label_audit gate <validators.jsonl> <out.jsonl>         # 共识闸门校验
"""

from pathlib import Path
from typing import Annotated, NamedTuple

import orjson
import typer
from jcx.sys.fs import files_in
from loguru import logger
from sklearn.cluster import DBSCAN

from jxl.det.box_utils import xyxy_iou
from jxl.det.hardmine import Box, parse_yolo_label

app = typer.Typer(add_completion=False, help="标注质量位置审计: 静态物毒标注签名发现")

DATE_PART = 2
"""stem 中日期字段序(约定 cam_ch_date_time_frame → [cam, ch, date, time, frame])"""


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


class BoxRow(NamedTuple):
    """一个标注框的审计记录(像素坐标 xyxy + 来源 stem)."""

    stem: str
    date: str
    box: tuple[float, float, float, float]
    feats: tuple[float, float, float, float]


def collect_box_rows(labels_dir: Path, imgsz: float) -> dict[str, list[BoxRow]]:
    """读全量 YOLO 标签 -> 按相机分组的框记录. 相机=stem 首字段('1_xxx'/'2_xxx')."""
    by_cam: dict[str, list[BoxRow]] = {}
    n_boxes = 0
    for f in files_in(labels_dir, ".txt"):
        parts = f.stem.split("_")
        cam, date = parts[0], parts[DATE_PART] if len(parts) > DATE_PART else "?"
        for nx1, ny1, nx2, ny2, _ in parse_yolo_label(f.read_text(encoding="utf-8")):
            x1, y1, x2, y2 = nx1 * imgsz, ny1 * imgsz, nx2 * imgsz, ny2 * imgsz
            by_cam.setdefault(cam, []).append(
                BoxRow(f.stem, date, (x1, y1, x2, y2), ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            )
            n_boxes += 1
    logger.info(f"boxes={n_boxes} cams={sorted(by_cam)}")
    return by_cam


def cluster_cam(rows: list[BoxRow], eps: float, min_samples: int) -> list[dict]:
    """单相机 DBSCAN 聚类 -> 簇统计列表(按 n_boxes 降序)."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict([r.feats for r in rows])
    clusters = []
    for cid in set(labels) - {-1}:
        idxs = [i for i, lab in enumerate(labels) if lab == cid]
        members = [rows[i] for i in idxs]
        sizes = [rows[i].feats[2] * rows[i].feats[3] for i in idxs]

        def stat(k: int, _idxs: list[int] = idxs) -> float:
            return _mean([rows[i].feats[k] for i in _idxs])

        clusters.append(
            {
                "cam": members[0].stem.split("_")[0],
                "cid": int(cid),
                "n_boxes": len(members),
                "n_dates": len({m.date for m in members}),
                "center": [round(stat(0), 1), round(stat(1), 1)],
                "size_mean": round(_mean(sizes), 0),
                "size_std": round(_std(sizes), 0),
                "extent": [
                    round(min(m.box[0] for m in members), 1),
                    round(min(m.box[1] for m in members), 1),
                    round(max(m.box[2] for m in members), 1),
                    round(max(m.box[3] for m in members), 1),
                ],
            }
        )
    return sorted(clusters, key=lambda c: -c["n_boxes"])


@app.command()
def cluster(
    labels_dir: Annotated[Path, typer.Argument(help="YOLO labels 目录(stem 约定 cam_ch_date_time_frame)")],
    out_json: Annotated[Path, typer.Argument(help="簇统计 JSON 输出")],
    imgsz: Annotated[int, typer.Option(help="标注坐标基准尺寸")] = 640,
    eps: Annotated[float, typer.Option(help="DBSCAN 聚类半径(像素)")] = 30.0,
    min_samples: Annotated[int, typer.Option(help="DBSCAN min_samples")] = 20,
    min_n: Annotated[int, typer.Option(help="嫌疑簇最小框数")] = 20,
    min_d: Annotated[int, typer.Option(help="嫌疑簇最小跨日期数")] = 3,
) -> None:
    """位置聚类审计: 输出全部簇统计, 标记 suspect(送 VLM 簇定性)."""
    by_cam = collect_box_rows(labels_dir, imgsz)
    clusters = [c for rows in by_cam.values() for c in cluster_cam(rows, eps, min_samples)]
    for c in clusters:
        c["suspect"] = c["n_boxes"] >= min_n and c["n_dates"] >= min_d
    out_json.write_text(orjson.dumps(clusters, option=orjson.OPT_INDENT_2).decode())
    n_suspect = sum(c["n_boxes"] for c in clusters if c["suspect"])
    logger.info(f"clusters={len(clusters)} suspect_boxes={n_suspect} -> {out_json}")


STRONG_VALIDATORS = ("yoloe", "rfdetr")
"""强验证器(闭集 COCO 系, 误检玩具/静态物概率低); gdino/la 为开放词汇弱验证器."""


def _supporters(box: Box, validators: dict[str, list], iou_thr: float) -> set[str]:
    """给出框的支持验证器集合(IoU>=thr). box 为归一化 xyxy+conf( validators dump 口径)."""
    found = set()
    for name, boxes in validators.items():
        if any(xyxy_iou(box[:4], tuple(vb)[:4]) >= iou_thr for vb in boxes):
            found.add(name)
    return found


@app.command()
def gate(
    validators_jsonl: Annotated[Path, typer.Argument(help="det_mine --dump-validators 产物 jsonl")],
    out_jsonl: Annotated[Path, typer.Argument(help="被挡框清单输出(jsonl, 一行一框)")],
    level: Annotated[str, typer.Option(help="只校验该 level 的 target(如 L0); 空串=全部")] = "L0",
    iou_thr: Annotated[float, typer.Option(help="支持判定 IoU 阈值")] = 0.5,
    min_k: Annotated[int, typer.Option(help="闸门: 总支持验证器数下限")] = 2,
    min_strong: Annotated[int, typer.Option(help="闸门: 强验证器支持数下限")] = 1,
) -> None:
    """共识闸门校验: target 框中不满足 k>=min_k ∧ strong>=min_strong 的输出为待审.

    只校验共识直接采信层(level=L0)——L1/仲裁/人工层各有自己的产生机制, 不重复过闸.
    n001 GT 反演(阈值报告 §3-4): 现行 k=2 闸门对毒框拦截率仅 0.4%; k>=2∧strong>=1
    拦截 97.3%/误伤 3.1%; 任何共识闸门(含 4/4)残留非零 → 须配合 cluster 位置卫兵.
    """
    n_target = n_blocked = n_rows = 0
    with out_jsonl.open("w") as f:
        for line in validators_jsonl.open(encoding="utf-8"):
            row = orjson.loads(line)
            if level and row.get("level") != level:
                continue
            n_rows += 1
            for t in [tuple(b) for b in row["target"]]:
                n_target += 1
                supp = _supporters(t, row["validators"], iou_thr)
                n_strong = len(supp & set(STRONG_VALIDATORS))
                if len(supp) < min_k or n_strong < min_strong:
                    n_blocked += 1
                    f.write(
                        orjson.dumps(
                            {"stem": row["stem"], "box": list(t), "supporters": sorted(supp),
                             "n_strong": n_strong, "level": row.get("level")}
                        ).decode()
                        + "\n"
                    )
    logger.info(
        f"rows={n_rows} target_boxes={n_target} blocked={n_blocked} "
        f"({n_blocked / max(n_target, 1):.1%}) -> {out_jsonl}"
    )


if __name__ == "__main__":
    app()
