#!/usr/bin/env python3
"""豆包 VLM 二级仲裁: det_mine review 分歧帧 → 确认标注 / 人工队列.

n001 共识标注管线的分歧层二级仲裁。读 <consensus_dir>/review/manifest.jsonl
(det_mine 产物, 行含 image/score/target_boxes/validators/breakdown), 对每图:

1. 豆包 vision grounding(复用 doubao_relabel.ground_one, prompt 取 --target 加载
   的 TargetProfile.vlm_prompt)得豆包框集;
2. find_consensus_positions(validators, iou_thr=--iou-consensus, k=2) 求模型共识
   位置(第一层放宽口径 IoU 0.3, 每位置支持模型数 ≥2);
3. 多数确认(严格口径 IoU≥--iou-major, 贪心一对一): 图内所有共识位置均获豆包
   重叠且豆包无无人支持的新位置 → 仲裁通过, 标注框取 pick_by_priority 固定
   优先序的代表框(la conf 恒 1.0, 不能按 conf 排序); 任一位置未获赞同或豆包
   独有位置 → 无多数 → 人工队列; 无共识位置且豆包无框 → 空标确认。

产出: <out>/confirmed/labels/<stem>.txt(YOLO) + <out>/manual/manifest.jsonl
(原行 + doubao 框 + 未赞同位置明细, 可再跑 review_pack 出网格) +
<out>/arbitrate_report.json(通过/人工/豆包空/错误计数); 错误图(缺图/API 失败)
落 <out>/_errors.jsonl 供重试。key 从配置文件读(--cfg), 绝不硬编码。

用法:
    doubao_arbitrate <consensus_dir> <images_dir> <out_dir> \
        --cfg <doubao.json> --target person --model doubao-seed-2-0-lite-260215
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, NamedTuple

import httpx
import orjson
import typer

from jxl.bin.det_mine import gather_images
from jxl.bin.doubao_relabel import ground_one
from jxl.bin.review_pack import ReviewEntry, find_manifests, parse_entry
from jxl.bin.rmb_ground import Backend, Detection, load_backend
from jxl.det.box_utils import xyxy_iou
from jxl.det.hardmine import (
    Box,
    find_consensus_positions,
    greedy_match,
    pick_by_priority,
    to_yolo_label,
)
from jxl.target import load_target

app = typer.Typer(
    add_completion=False, help="豆包 VLM 二级仲裁: review 分歧帧 → 确认/人工。"
)

# 标注框固定优先序: la conf 恒 1.0 无区分度, 不能按 conf 排序; rfdetr>gdino>yoloe
# 同 score_sample 的权重序语义
PICK_PRIORITY: tuple[str, ...] = ("rfdetr", "gdino", "yoloe", "la")
# 共识位置支持模型数下限(多数语义, 用户裁决)
_CONSENSUS_K = 2


class PositionVerdict(NamedTuple):
    """单个共识位置的仲裁明细(未获赞同时入 manual manifest)。"""

    box: Box  # 该位置将写入的标注框(pick 优先序选出)
    supporters: list[str]
    best_doubao_iou: float  # 豆包框与该位置标注框的最大 IoU(未达 iou_major 即不赞同)


class Arbitration(NamedTuple):
    """单图仲裁结果。"""

    confirmed: bool
    label_boxes: list[Box]  # confirmed 时的 YOLO 标注框(空标确认 = [])
    unapproved: list[PositionVerdict]  # 未获豆包赞同的共识位置
    doubao_only: list[Box]  # 豆包检出但无共识位置支持的框


def pick_label_box(representative: Box, supporters: dict[str, Box]) -> Box:
    """共识位置标注框: pick_by_priority 固定优先序; 支持者无已知校验器时回退代表框。"""
    return pick_by_priority(supporters, list(PICK_PRIORITY)) or representative


def arbitrate_image(
    consensus_positions: list[tuple[Box, dict[str, Box]]],
    doubao_boxes: list[Box],
    iou_major: float,
) -> Arbitration:
    """图级多数仲裁(纯函数)。

    判定: 所有共识位置均获某豆包框 IoU≥iou_major 重叠(greedy_match 贪心一对一)
    且无豆包独有框 → confirmed; 无共识位置且豆包无框 → 空标确认
    (confirmed, label_boxes=[]); 否则 → 人工队列。
    """
    picks = [pick_label_box(rep, sup) for rep, sup in consensus_positions]
    matched, _unmatched_pos, unmatched_doubao = greedy_match(
        picks, doubao_boxes, iou_major
    )
    supported = {ia for ia, _ib, _iov in matched}
    label_boxes: list[Box] = []
    unapproved: list[PositionVerdict] = []
    for i, (box, (_rep, sup)) in enumerate(
        zip(picks, consensus_positions, strict=True)
    ):
        if i in supported:
            label_boxes.append(box)
        else:
            best = max((xyxy_iou(box[:4], db[:4]) for db in doubao_boxes), default=0.0)
            unapproved.append(PositionVerdict(box, sorted(sup), best))
    doubao_only = [doubao_boxes[i] for i in unmatched_doubao]
    return Arbitration(
        not unapproved and not doubao_only, label_boxes, unapproved, doubao_only
    )


def detections_to_boxes(dets: list[Detection]) -> list[Box]:
    """豆包 Detection(归一化 bbox, 已 clamp [0,1]) → Box 五元组。"""
    return [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.conf) for d in dets]


async def ground_all(
    paths: list[Path],
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    concurrency: int,
) -> list[tuple[Path, list[Detection], str | None]]:
    """并发豆包 grounding(复用 doubao_relabel.ground_one, 进度每 50 图一行)。"""
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            ground_one(client, sem, p, base_url, api_key, model, prompt) for p in paths
        ]
        results: list[tuple[Path, list[Detection], str | None]] = []
        for done, coro in enumerate(asyncio.as_completed(tasks), 1):
            results.append(await coro)
            if done % 50 == 0 or done == len(paths):
                typer.echo(f"  进度 {done}/{len(paths)}")
        return results


def _manual_row(line: str, doubao_boxes: list[Box], arb: Arbitration) -> str:
    """manual manifest 行: 原 manifest 行(原样字段含 breakdown) + 豆包仲裁明细。"""
    row: dict[str, object] = dict(orjson.loads(line))
    row["doubao_boxes"] = [list(b) for b in doubao_boxes]
    row["unapproved_positions"] = [
        {
            "box": list(v.box),
            "supporters": v.supporters,
            "best_doubao_iou": round(v.best_doubao_iou, 4),
        }
        for v in arb.unapproved
    ]
    row["doubao_only"] = [list(b) for b in arb.doubao_only]
    return orjson.dumps(row).decode()


@app.command()
def run(
    consensus_dir: Annotated[
        Path, typer.Argument(help="det_mine 共识输出目录(含 review/)")
    ],
    images_dir: Annotated[Path, typer.Argument(help="帧图目录(按 stem 递归找图)")],
    out_dir: Annotated[Path, typer.Argument(help="仲裁输出目录")],
    target: Annotated[
        str, typer.Option("--target", help="目标 profile 名(targets/<name>.toml)")
    ] = "",
    target_profile: Annotated[
        Path,
        typer.Option(
            "--target-profile", help="显式 profile toml 路径(优先于 --target)"
        ),
    ] = Path(),
    cfg: Annotated[
        str, typer.Option("--cfg", help="豆包配置文件(base_url+api_key+model)")
    ] = "",
    model: Annotated[
        str, typer.Option("--model", help="覆盖模型名")
    ] = "doubao-seed-2-0-lite-260215",
    iou_major: Annotated[
        float, typer.Option("--iou-major", help="豆包多数确认 IoU 阈值(严格口径)")
    ] = 0.5,
    iou_consensus: Annotated[
        float,
        typer.Option("--iou-consensus", help="共识位置聚类 IoU 阈值(第一层放宽口径)"),
    ] = 0.3,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="豆包 API 并发数")
    ] = 6,
    limit: Annotated[int, typer.Option("--limit", help="只处理前 N 行(0=全部)")] = 0,
) -> None:
    """豆包二级仲裁: 多数确认 → confirmed/labels; 无多数 → manual 队列。"""
    if not target and not target_profile.name:
        msg = "需指定 --target 或 --target-profile"
        raise typer.BadParameter(msg)
    prof = load_target(target, target_profile if target_profile.name else None)
    if not 0.0 <= iou_major <= 1.0 or not 0.0 <= iou_consensus <= 1.0:
        typer.secho(
            "--iou-major/--iou-consensus 须在 [0,1]", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)

    manifests = find_manifests(consensus_dir)
    if not manifests:
        typer.secho(
            f"无 review manifest: {consensus_dir}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)
    lines = [
        ln
        for m in manifests
        for ln in m.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        typer.secho(
            f"零 review 行(豆包空输入): {consensus_dir}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)
    if limit:
        lines = lines[:limit]
    entries = [parse_entry(ln) for ln in lines]

    img_by_stem = {p.stem: p for p in gather_images(images_dir)}
    if not img_by_stem:
        typer.secho(f"候选目录无图: {images_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    base_url, api_key, use_model = load_backend(Backend.DOUBAO, model, cfg)
    labels_dir = out_dir / "confirmed" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manual").mkdir(parents=True, exist_ok=True)
    typer.secho(
        f"豆包二级仲裁 {len(entries)} review 图 @ {use_model}", fg=typer.colors.CYAN
    )

    # 缺图 → 错误行(不静默跳过); 有图 → grounding
    row_by_stem: dict[str, tuple[str, ReviewEntry]] = {}
    paths: list[Path] = []
    err_lines: list[str] = []
    for ln, entry in zip(lines, entries, strict=True):
        stem = Path(entry.image).stem
        row_by_stem[stem] = (ln, entry)
        path = img_by_stem.get(stem)
        if path is None:
            err_lines.append(
                orjson.dumps(
                    {"image": entry.image, "error": f"image_missing: {entry.image}"}
                ).decode()
            )
        else:
            paths.append(path)

    results = asyncio.run(
        ground_all(paths, base_url, api_key, use_model, prof.vlm_prompt, concurrency)
    )
    n_confirmed = n_manual = n_doubao_empty = 0
    manual_lines: list[str] = []
    for path, dets, err in sorted(results, key=lambda r: str(r[0])):
        line, entry = row_by_stem[path.stem]
        if err is not None:
            err_lines.append(
                orjson.dumps({"image": entry.image, "error": err}).decode()
            )
            continue  # 错误图: 不仲裁, 落 _errors.jsonl 供重试
        if not dets:
            n_doubao_empty += 1
        doubao_boxes = detections_to_boxes(dets)
        validators = {n: bs for n, bs in entry.boxes_by_model.items() if n != "target"}
        positions = find_consensus_positions(validators, iou_consensus, _CONSENSUS_K)
        arb = arbitrate_image(positions, doubao_boxes, iou_major)
        if arb.confirmed:
            (labels_dir / (path.stem + ".txt")).write_text(
                to_yolo_label(arb.label_boxes, cls_id=prof.output_cls_id),
                encoding="utf-8",
            )
            n_confirmed += 1
        else:
            manual_lines.append(_manual_row(line, doubao_boxes, arb))
            n_manual += 1

    (out_dir / "manual" / "manifest.jsonl").write_text(
        "\n".join(manual_lines) + ("\n" if manual_lines else ""), encoding="utf-8"
    )
    if err_lines:
        (out_dir / "_errors.jsonl").write_text(
            "\n".join(err_lines) + "\n", encoding="utf-8"
        )
    report: dict[str, object] = {
        "target": prof.name,
        "model": use_model,
        "total": len(lines),
        "confirmed": n_confirmed,
        "manual": n_manual,
        "doubao_empty": n_doubao_empty,
        "errors": len(err_lines),
        "iou_major": iou_major,
        "iou_consensus": iou_consensus,
        "consensus_k": _CONSENSUS_K,
    }
    (out_dir / "arbitrate_report.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2)
    )
    typer.secho(
        f"确认 {n_confirmed} | 人工 {n_manual} | 豆包空 {n_doubao_empty} | 错误 {len(err_lines)} → {out_dir}",
        fg=typer.colors.GREEN,
    )
    if n_manual:
        typer.echo(
            f"人工队列出网格: review_pack {out_dir}/manual <images_dir> {out_dir}/manual/pack"
        )


if __name__ == "__main__":
    app()
