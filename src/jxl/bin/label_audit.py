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
    label_audit <labels_dir> <out_json> [--imgsz 640] [--eps 30] [--min-samples 20]
        [--min-n 20] [--min-d 3]
"""

from pathlib import Path
from typing import Annotated, NamedTuple

import orjson
import typer
from jcx.sys.fs import files_in
from loguru import logger
from sklearn.cluster import DBSCAN

from jxl.det.hardmine import parse_yolo_label

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
def main(
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


if __name__ == "__main__":
    app()
