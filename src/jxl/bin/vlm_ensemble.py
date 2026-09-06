#!/usr/bin/env python3
"""VLM-Ensemble: 人工队列争议帧的三 VLM 多数仲裁(豆包已票 + Qwen + MiniMax).

输入 doubao_arbitrate 的 manual/manifest.jsonl(431 帧, 含 doubao_boxes 与原模型框),
对每帧再调 Qwen-VL / MiniMax-VL grounding 各得一票 → 三 VLM 多数判定:
- 某共识位置(≥2 检测器, find_consensus_positions IoU 0.4)获 ≥2/3 VLM 赞同
  (IoU≥0.4, 贪心一对一) → 确认, 标注框取优先序代表框
- 无共识位置且已投票 VLM 全空(≥2 票) → 空标确认
- 其余 → 最终人工队列(manifest 原行 + 两 VLM 框)

弃权语义: 单 VLM 调用失败 = 该票 None(弃权), 不当空票计数(空标确认需真投出的
"无人"票); 赞同阈值仍是 ≥2 票, 弃权只缩小选民池.

提示词: 各家用官方 grounding 范式(用户裁决, 2026-09-04):
- Qwen 官方 "Detect all ... bounding boxes in JSON format with keys 'label'
  and 'bbox_2d' as [x1,y1,x2,y2]", 坐标为输入图像绝对像素(qwenlm blog).
- MiniMax 官方 Model Card 无 grounding 模板, 用同构 JSON bbox_2d 像素约定
  (M3 think 后稳定输出该格式, 2026-09-06 冒烟实测).

模型(网关 2026-09-06 models 实测; qwen-vl-max-latest/MiniMax-VL-01 均不在列):
- Qwen: qwen3-vl-plus(强) / qwen3-vl-flash(快) — bbox_2d 像素输出一致.
- MiniMax: MiniMax-M3 多模态(网关无 VL 系列), <think> 推理后输出 bbox_2d.

API key 从环境变量读(S4_QWEN_API_KEY / S4_MINMAX_API_KEY), 绝不落地会话文本.
用法:
    vlm_ensemble <manual_dir> <images_dir> <out_dir> --target person
"""

import asyncio
import base64
import os
import re
from pathlib import Path
from typing import Annotated, NamedTuple

import httpx
import orjson
import typer
from PIL import Image as PILImage

from jxl.bin.doubao_arbitrate import pick_label_box
from jxl.bin.review_pack import parse_entry
from jxl.det.box_utils import xyxy_iou
from jxl.det.hardmine import (
    Box,
    find_consensus_positions,
    greedy_match,
    to_yolo_label,
)

app = typer.Typer(add_completion=False, help="三 VLM 多数仲裁(Qwen+MiniMax 补票).")

QWEN_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3-vl-plus"
MM_BASE = "https://api.minimaxi.com/v1"
MM_MODEL = "MiniMax-M3"

QWEN_PROMPT_TMPL = (
    "Detect all {target} in the image and output their bounding boxes in JSON "
    "format with keys 'label' and 'bbox_2d' as [x1, y1, x2, y2]. "
    "Coordinates are absolute pixels of the input image. Output only the JSON array."
)
MM_PROMPT_TMPL = (
    "Detect all {target} in the image. Output ONLY a JSON array, no other text, "
    'no markdown: [{{"label":"{target}","bbox_2d":[x1,y1,x2,y2]}}] '
    "where bbox_2d coordinates are absolute pixels of the input image. "
    "List all visible targets (including partially occluded). Output [] if none."
)


class VlmVote(NamedTuple):
    """单 VLM 对一帧的框票(归一化 xyxy, conf=1.0); error 非 None 即弃权."""

    boxes: list[Box]
    error: str | None = None


_NUM = r"\d+(?:\.\d+)?"
# 键名后紧跟 4 个数字: 键名泛化任意 \w+_2d —— qwen3-vl-plus 冒烟/全量实测已见
# bbox_2d(官方)/coordinate_2d/label_2d 三种变体, 值恒为 4 数字像素组;
# 且容忍畸形收尾(如 "..., 445}}", 花括号收尾) —— 只锚定数字组本身
_BBOX_RE = re.compile(
    rf'"?\w+_2d"?\s*:\s*\[\s*({_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM})'
)


def parse_vlm_json(text: str, img_w: int, img_h: int) -> list[Box]:
    """VLM JSON 输出(官方 bbox_2d 绝对像素) → 归一化 [Box].

    先剥 MiniMax-M3 的 <think> 推理前缀(其内可能含方括号污染切片);
    正则提取全部 bbox_2d 数字组 —— qwen3-vl-plus 偶发把多框挤进一个对象的
    重复 bbox_2d 键(2026-09-06 冒烟实测), 逐对象解析会静默丢框.
    坐标 clamp [0,1](qwen 偶发越界, 如 926/1000 > 640).
    解析失败抛 ValueError(调用方转弃权票, 不中断整体).
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    captures = _BBOX_RE.findall(text)
    if not captures:
        if text.find("[") >= 0 and "[]" not in text.replace(" ", ""):
            raise ValueError(f"有数组无 bbox_2d: {text[:80]}")
        return []  # 空检出 [] (明确无人)
    boxes: list[Box] = []
    for cap in captures:
        nums = [float(v) for v in cap.split(",")]
        x1, y1, x2, y2 = nums[:4]
        boxes.append((
            min(max(min(x1, x2) / img_w, 0.0), 1.0),
            min(max(min(y1, y2) / img_h, 0.0), 1.0),
            min(max(max(x1, x2) / img_w, 0.0), 1.0),
            min(max(max(y1, y2) / img_h, 0.0), 1.0),
            1.0,
        ))
    return boxes


async def call_vlm(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    img_b64: str,
    img_w: int,
    img_h: int,
    sem: asyncio.Semaphore,
) -> VlmVote:
    """OpenAI 兼容 chat+vision 调用(Qwen/MiniMax 同构). 单帧一票, 失败弃权."""
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    async with sem:
        try:
            r = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=90,
            )
            if r.status_code >= 400:
                # 响应体入错误串(诊断用; 不含 key, 请求头不回显)
                raise ValueError(f"HTTP {r.status_code}: {r.text[:200]}")
            text = r.json()["choices"][0]["message"]["content"]
            return VlmVote(parse_vlm_json(text, img_w, img_h))
        except Exception as e:  # 单 VLM 失败弃权(错误串入 manifest), 不中断整体
            return VlmVote([], error=f"{type(e).__name__}: {e}"[:400])


def ensemble_verdict(
    positions: list[tuple[Box, dict[str, Box]]],
    votes: dict[str, list[Box] | None],
    iou: float,
) -> tuple[bool, list[Box], list[int]]:
    """三 VLM 多数判定(纯函数).

    votes 值 None=弃权. 每共识位置统计投出且含重叠框(IoU≥iou, 贪心一对一)的
    VLM 数, ≥2 → 该位置确认. 无位置时已投票全空且 ≥2 票 → 空标确认.

    VLM 多数独有检出挡(对称豆包层 doubao_only 语义): ≥2 个 VLM 在无共识位置
    支持的同一区域重叠出框 → 疑似检测器漏检 → 人工, 不确认.

    Returns: (图确认?, 标注框列表, 各位置赞同数)
    """
    cast = {k: v for k, v in votes.items() if v is not None}
    if not positions:
        return (len(cast) >= 2 and all(not b for b in cast.values()), [], [])
    reps = [rep for rep, _ in positions]
    label_boxes: list[Box] = []
    approves: list[int] = []
    for rep, supporters in positions:
        n = sum(
            1 for boxes in cast.values() if greedy_match(boxes, [rep], iou)[0]
        )
        approves.append(n)
        if n >= 2:
            label_boxes.append(pick_label_box(rep, supporters))
    # 各 VLM 不与任何共识位置匹配的剩余框 → 跨 VLM 两两重叠即"多数独有检出"
    extras: dict[str, list[Box]] = {}
    for vlm, boxes in cast.items():
        _, unmatched, _ = greedy_match(boxes, reps, iou)
        extras[vlm] = [boxes[i] for i in unmatched]
    vlm_only = any(
        xyxy_iou(a[:4], b[:4]) >= iou
        for va, bs_a in extras.items()
        for vb, bs_b in extras.items()
        if va < vb
        for a in bs_a
        for b in bs_b
    )
    confirmed = len(label_boxes) == len(positions) and not vlm_only
    return (confirmed, label_boxes, approves)


def load_manual_rows(manifest: Path) -> list[dict]:
    """doubao_arbitrate manual manifest → 原行 dict 列表(parse_entry 校验字段合法性)."""
    if not manifest.is_file():
        typer.secho(f"无 manual manifest: {manifest}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        typer.secho(f"manifest 零行: {manifest}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    rows: list[dict] = []
    for ln in lines:  # parse_entry 仅做校验(字段形状非法 fail-fast), 输出用原行 dict
        parse_entry(ln)
        rows.append(orjson.loads(ln))
    return rows


@app.command()
def run(
    manual_dir: Annotated[Path, typer.Argument(help="doubao_arbitrate 的 manual 目录")],
    images_dir: Annotated[Path, typer.Argument(help="图片目录(按 stem 递归找图)")],
    out_dir: Annotated[Path, typer.Argument(help="输出目录")],
    target: Annotated[str, typer.Option("--target", help="目标文本")] = "person",
    iou: Annotated[float, typer.Option("--iou", help="赞同/聚类 IoU(统一 0.4)")] = 0.4,
    consensus_k: Annotated[int, typer.Option("--consensus-k", help="检测器共识数")] = 2,
    concurrency: Annotated[int, typer.Option("--concurrency", help="每 VLM 并发")] = 4,
    limit: Annotated[int, typer.Option("--limit", help="只处理前 N 行(0=全部, 冒烟用)")] = 0,
) -> None:
    """431 争议帧三 VLM 多数仲裁(Qwen+MiniMax 各自官方 grounding 提示词)."""
    qwen_key = os.environ.get("S4_QWEN_API_KEY", "")
    mm_key = os.environ.get("S4_MINMAX_API_KEY", "")
    if not qwen_key or not mm_key:
        typer.secho("缺 S4_QWEN_API_KEY / S4_MINMAX_API_KEY 环境变量", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    rows = load_manual_rows(manual_dir / "manifest.jsonl")
    if limit:
        rows = rows[:limit]
    (out_dir / "confirmed" / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "manual").mkdir(parents=True, exist_ok=True)
    from jxl.bin.det_mine import gather_images

    img_by_stem = {p.stem: p for p in gather_images(images_dir)}
    typer.secho(f"三 VLM 仲裁 {len(rows)} 帧 @ {QWEN_MODEL} + {MM_MODEL}", fg=typer.colors.CYAN)

    async def one(client: httpx.AsyncClient, row: dict, sem: asyncio.Semaphore) -> dict:
        img = img_by_stem.get(Path(row["image"]).stem)
        if img is None:
            return {**row, "ensemble": {"confirmed": False, "error": "missing_image"}}
        with PILImage.open(img) as im:
            img_w, img_h = im.size
        b64 = base64.b64encode(img.read_bytes()).decode()
        qwen, mm = await asyncio.gather(
            call_vlm(client, QWEN_BASE, qwen_key, QWEN_MODEL,
                     QWEN_PROMPT_TMPL.format(target=target), b64, img_w, img_h, sem),
            call_vlm(client, MM_BASE, mm_key, MM_MODEL,
                     MM_PROMPT_TMPL.format(target=target), b64, img_w, img_h, sem),
        )
        entry = parse_entry(orjson.dumps(row).decode())
        validators = {n: bs for n, bs in entry.boxes_by_model.items() if n != "target"}
        positions = find_consensus_positions(validators, iou, consensus_k)
        votes: dict[str, list[Box] | None] = {
            "doubao": [tuple(b) for b in row.get("doubao_boxes", [])],
            "qwen": None if qwen.error else qwen.boxes,
            "minimax": None if mm.error else mm.boxes,
        }
        ok, label_boxes, approves = ensemble_verdict(positions, votes, iou)
        rec = {
            **row,
            "qwen_boxes": [list(b) for b in qwen.boxes],
            "minimax_boxes": [list(b) for b in mm.boxes],
            "ensemble": {"confirmed": ok, "approves": approves,
                         "errors": {"qwen": qwen.error, "minimax": mm.error}},
        }
        if ok:
            (out_dir / "confirmed" / "labels" / (img.stem + ".txt")).write_text(
                to_yolo_label(label_boxes), encoding="utf-8"
            )
        return rec

    async def main() -> list[dict]:
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient() as client:
            tasks = [one(client, r, sem) for r in rows]
            results: list[dict] = []
            for done, coro in enumerate(asyncio.as_completed(tasks), 1):
                results.append(await coro)
                if done % 25 == 0 or done == len(tasks):
                    typer.echo(f"  进度 {done}/{len(tasks)}")
            return results

    results = asyncio.run(main())
    confirmed = [r for r in results if r["ensemble"]["confirmed"]]
    manual = [r for r in results if not r["ensemble"]["confirmed"]]
    n_err = sum(
        1 for r in results if r["ensemble"].get("errors", {}).get("qwen")
        or r["ensemble"].get("errors", {}).get("minimax")
    )
    with (out_dir / "manual" / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in manual:
            f.write(orjson.dumps(r).decode() + "\n")
    report = {
        "total": len(results),
        "confirmed": len(confirmed),
        "manual": len(manual),
        "vlm_error_frames": n_err,
        "models": {"doubao": "doubao-seed-2-0-lite-260215", "qwen": QWEN_MODEL, "minimax": MM_MODEL},
        "iou": iou,
        "consensus_k": consensus_k,
        "rule": ">=2/3 VLM approve a >=2-detector position; abstain on API error",
    }
    (out_dir / "ensemble_report.json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    typer.secho(
        f"三 VLM 仲裁: 确认 {len(confirmed)} | 最终人工 {len(manual)} | VLM出错帧 {n_err} → {out_dir}",
        fg=typer.colors.GREEN,
    )
    if manual:
        typer.echo(
            f"人工队列出网格: review_pack {out_dir}/manual <images_dir> {out_dir}/manual/pack"
        )


if __name__ == "__main__":
    app()
