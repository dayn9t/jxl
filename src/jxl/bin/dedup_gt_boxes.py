#!/usr/bin/env python3
"""YOLO 数据集帧内近重复 GT 清洗（person v3 重复检测归因的治本工具）。

背景：dataset_v3 train 含 73 对 IoU>=0.99 近重复 GT（共识融合层把多源同目标写成
两条），YOLO26 one-to-one 头由此学会重复发射。见
projects/sgcc/research/2026-09-12-检测器重复框归因.md。

默认只报告不写回（--report 输出被删框明细）；--apply 才写回，保留同对中第一条
（原文件序），删除后续近重复框。

用法：
  uv run --project . python -m jxl.bin.dedup_gt_boxes <dataset_dir>            # 报告
  uv run --project . python -m jxl.bin.dedup_gt_boxes <dataset_dir> --apply    # 清洗
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jxl.det.box_utils import xyxy_iou

import typer

NEAR_DUP_IOU = 0.95
"""帧内同类近重复判定阈（实证 dup 对全部 >=0.99；0.95 覆盖 iapx near 带）。"""

app = typer.Typer(add_completion=False)


@dataclass(frozen=True, slots=True)
class YoloBox:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)


def parse_label(text: str) -> list[YoloBox]:
    out: list[YoloBox] = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append(YoloBox(int(p[0]), *(float(v) for v in p[1:5])))
    return out


def boxes_to_text(boxes: list[YoloBox]) -> str:
    return "".join(f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}\n"
                   for b in boxes)


def iou(a: YoloBox, b: YoloBox) -> float:
    """YoloBox IoU——委托规范实现 jxl.det.box_utils.xyxy_iou."""
    return xyxy_iou(a.xyxy, b.xyxy)

def find_dup_pairs(boxes: list[YoloBox], threshold: float = NEAR_DUP_IOU
                   ) -> list[tuple[int, int]]:
    """同帧同类框对中 IoU>=threshold 的 (keep_idx, drop_idx) 序对（keep=序在前）。"""
    pairs: list[tuple[int, int]] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].cls == boxes[j].cls and iou(boxes[i], boxes[j]) >= threshold:
                pairs.append((i, j))
    return pairs


def dedup_boxes(boxes: list[YoloBox], threshold: float = NEAR_DUP_IOU
                ) -> tuple[list[YoloBox], list[tuple[int, int]]]:
    """去掉近重复框（保留每对中序在前的），返回 (清洗后, 被删原序号列表)。"""
    pairs = find_dup_pairs(boxes, threshold)
    drops = sorted({j for _, j in pairs})
    return [b for i, b in enumerate(boxes) if i not in set(drops)], drops


def process_split(labels_dir: Path, threshold: float, apply: bool) -> dict[str, int]:
    stats = {"files": 0, "dup_pairs": 0, "removed": 0}
    report_lines: list[str] = []
    for fp in sorted(labels_dir.glob("*.txt")):
        boxes = parse_label(fp.read_text(encoding="utf-8"))
        if len(boxes) < 2:
            continue
        pairs = find_dup_pairs(boxes, threshold)
        if not pairs:
            continue
        stats["files"] += 1
        stats["dup_pairs"] += len(pairs)
        cleaned, drops = dedup_boxes(boxes, threshold)
        stats["removed"] += len(drops)
        for i, j in pairs:
            report_lines.append(f"{fp.name}\tkeep#{i}\tdrop#{j}\tIoU_pair")
        if apply:
            fp.write_text(boxes_to_text(cleaned), encoding="utf-8")
    if report_lines:
        report = labels_dir.parent / f"dedup_report_{labels_dir.name}.txt"
        report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        typer.echo(f"report -> {report}")
    return stats


@app.command()
def main(
    dataset_dir: Path = typer.Argument(help="数据集根（含 train/val/test/*/labels）"),
    threshold: float = typer.Option(NEAR_DUP_IOU, help="近重复 IoU 阈"),
    apply: bool = typer.Option(False, "--apply", help="写回清洗（默认只报告）"),
) -> None:
    if not dataset_dir.is_dir():
        raise typer.BadParameter(f"数据集目录不存在: {dataset_dir}")
    total = {"files": 0, "dup_pairs": 0, "removed": 0}
    for labels_dir in sorted(dataset_dir.glob("*/labels")):
        st = process_split(labels_dir, threshold, apply)
        typer.echo(f"{labels_dir.parent.name}: {st}")
        for k in total:
            total[k] += st[k]
    typer.echo(f"TOTAL {total} (apply={apply})")


if __name__ == "__main__":
    app()
