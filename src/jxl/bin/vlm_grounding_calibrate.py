"""多 VLM grounding 刻度标定（2026-09-13，共识标注体系前置）。

接入新 VLM 投票池前，用已知 GT 框图实测每家的输出刻度与框质量，
产出 per-model 报告（坐标 max 反推除数 / IoU 分布 / F1@0.5）——
投票池成员与权重以此报告为唯一依据（假设-实证循环）。

知识依据（勿在此重复，查 KB）：
- ~/.claude/kb/30-areas/vlm-vision-grounding/20260710-vlm-grounding-coordinate-protocols.md
  （坐标协议对照 + 接新模型 7 步清单；本工具是其中第 3 步「刻度标定」的固化实现）
- ~/.claude/kb/30-areas/vlm-vision-grounding/20260913-vlm-service-inventory.md
  （服务/模型/env 变量名清单；本文档只写 env 名，key 值永不落盘）
- 同目录 20260913 之后新增的 grounding 经验也须回写 KB（知识-项目双向飞轮）。

标定集格式（jsonl 行）：{"image": 绝对路径, "width", "height", "boxes": [[x1,y1,x2,y2],...]}
生成方式见 gencheck/gt_calib_set.jsonl（dataset_v4 test 抽样，crop640 域）。

预设候选（--models，用户裁决 2026-09-13：在线商用优先+本地千问3.8增补+今年模型优先）：
  qwen-flash   qwen3-vl-flash-2026-01-22   dashscope   S4_QWEN_API_KEY   JSON bbox_2d 0-1000
  doubao-vl    doubao-seed-1-6-vision-250815 方舟      S4_DOUBAO_API_KEY <bbox> 标签（官方 0-1000，封装曾实测 0-1 → 自动反推）
  glm-flash    glm-5.3-flash               bigmodel 网关 ANTHROPIC_AUTH_TOKEN vision 可达性待本工具实测
  qwen35b-local qwen3.5-35b-a3b-fp8        本地 182 vllm  无需 key  同 qwen 协议（2026-09-14 实测服务端模型已由
              qwen3.8-flash-next 换为 3.5-35b——服务端模型漂移，接入前先 /v1/models 核对）

用法：
  uv run --project . python -m jxl.bin.vlm_grounding_calibrate \
      GT_JSONL --models qwen-flash,doubao-vl,glm-flash,qwen35b-local --out report.json
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path

import httpx
import orjson
import typer
from PIL import Image

app = typer.Typer(help="多 VLM grounding 刻度标定（KB 7 步清单第 3 步的固化实现）")

# 共享协议/解析/匹配/候选层：单一数据源 = vlm_pool.py（勿在此复制）
from jxl.bin.vlm_pool import (  # noqa: E402
    CANDIDATES,
    MAX_SIDE,
    MATCH_IOU as CONF_THRESHOLD,
    greedy_iou_match,
    image_data_url as _data_url,
    infer_divisor,
    iou as _iou,
    parse_boxes,
)

PROMPT = (
    "Detect all persons in the image. For each person output its bounding box. "
    "Coordinates are normalized to 0-1000 relative to image width and height, "
    "top-left origin, format [x1,y1,x2,y2]. "
    'Respond ONLY with JSON: {"persons": [{"bbox_2d": [x1,y1,x2,y2]}]}'
)


async def call_model(client: httpx.AsyncClient, alias: str, spec: dict,
                     image_url: str) -> tuple[str, str]:
    """返回 (raw_text | '' , error | '')。调用失败上抛语义由调用方计为弃权。"""
    key = os.environ.get(spec["key_env"], "") if spec["key_env"] else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = {"model": spec["model"], "temperature": 0.0, "max_tokens": 1500,
               **spec.get("extra_payload", {}),
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": image_url}},
                   {"type": "text", "text": PROMPT}]}]}
    try:
        r = await client.post(spec["endpoint"], json=payload, headers=headers, timeout=90.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], ""
    except Exception as e:  # noqa: BLE001 - 标定场景逐家记录失败原因
        return "", f"{type(e).__name__}: {e}"[:200]


@app.command()
def run(
    gt_jsonl: Path = typer.Argument(..., help="标定集 jsonl：{image,width,height,boxes}"),
    models: str = typer.Option("qwen-flash,doubao-vl,glm-flash,qwen35b-local",
                               help="逗号分隔候选别名（见模块 docstring）"),
    out: Path = typer.Option(Path("gt_calib_report.json"), help="报告输出路径"),
    limit: int = typer.Option(0, help=">0 只跑前 N 张（冒烟）"),
) -> None:
    """跑标定：每模型×每图 grounding → 刻度反推 → IoU 匹配 → per-model F1 报告。"""
    rows = [orjson.loads(l) for l in gt_jsonl.open() if l.strip()]
    if limit > 0:
        rows = rows[:limit]
    aliases = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [a for a in aliases if a not in CANDIDATES]
    if unknown:
        raise typer.BadParameter(f"未知候选: {unknown}；可选: {list(CANDIDATES)}")
    missing = [a for a in aliases
               if CANDIDATES[a]["key_env"] and not os.environ.get(CANDIDATES[a]["key_env"])]
    if missing:
        raise typer.BadParameter(
            f"env 缺失（只看变量名不打印值）: {[CANDIDATES[a]['key_env'] for a in missing]}")

    async def _run() -> dict:
        report: dict[str, dict] = {a: {"model": CANDIDATES[a]["model"],
                                       "errors": [], "frames": []}
                                   for a in aliases}
        async with httpx.AsyncClient() as client:
            for r in rows:
                url = _data_url(Path(r["image"]))
                W, H = float(r["width"]), float(r["height"])
                gt = r["boxes"]
                res = await asyncio.gather(
                    *[call_model(client, a, CANDIDATES[a], url) for a in aliases])
                for a, (text, err) in zip(aliases, res):
                    rec: dict = {"image": r["image"], "n_gt": len(gt)}
                    if err:
                        report[a]["errors"].append({"image": r["image"], "err": err})
                        report[a]["frames"].append(rec | {"error": err})
                        continue
                    raw = parse_boxes(text, CANDIDATES[a]["protocol"])
                    div, coord_max = infer_divisor(raw, CANDIDATES[a]["divisor"])
                    scaled = [[v / div * (W if i % 2 == 0 else H) for i, v in enumerate(b)]
                              for b in raw]
                    tp, np_, ng, ious = greedy_iou_match(scaled, gt)
                    rec |= {"n_pred": np_, "coord_max": round(coord_max, 2),
                            "divisor_used": div, "tp": tp,
                            "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0}
                    report[a]["frames"].append(rec)
        return report

    report = asyncio.run(_run())
    summary = {}
    for a, d in report.items():
        frames = [f for f in d["frames"] if "error" not in f]
        tp = sum(f["tp"] for f in frames)
        np_ = sum(f["n_pred"] for f in frames)
        ng = sum(f["n_gt"] for f in frames)
        p = tp / np_ if np_ else 0.0
        rc = tp / ng if ng else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        divs = sorted({f["divisor_used"] for f in frames})
        coord_max = max((f["coord_max"] for f in frames), default=0.0)
        summary[a] = {"model": d["model"], "F1@0.5": round(f1, 4),
                      "precision": round(p, 4), "recall": round(rc, 4),
                      "n_err": len(d["errors"]), "divisors_seen": divs,
                      "coord_max": coord_max}
    out.write_bytes(orjson.dumps({"summary": summary, "detail": report},
                                 option=orjson.OPT_INDENT_2))
    typer.echo(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())
    typer.secho(f"报告 → {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
