#!/usr/bin/env python3
"""人工窗口材料装配（预标注优先准则：候选框预放 + 人裁决）.

打包两路人工待裁帧到统一 X-AnyLabeling 工程:
    A. 批 2 仲裁残量(arbitration_human.jsonl): 未获校验器支持的共识聚类 → 类名? 候选框
    B. 批 1 兜底 464 帧(batch1 review): 校验器检出框 → 类名? 候选框
产物: <out>/{images(symlink), labels(X-AnyLabeling v4 json), labels.txt, manifest.jsonl, README.md}
人编辑导出后由管线回收（确认=去 ?；删=弃；空标帧=负样本）。
manifest 为合并快照：重跑并入旧条目，人已编辑帧的记录不丢。

典型用法:
    python c_humanpack.py   # 幂等(已有 json 的帧跳过)
"""

import json
from collections import defaultdict
from pathlib import Path

import orjson
import typer
from loguru import logger
from PIL import Image
from t4_preannot import _shape, _xany_json

app = typer.Typer(add_completion=False, help="人工窗口材料装配.")

B2 = Path("/mnt/data/s4det/p3-seedset/batch2/mining")
B1 = Path("/mnt/data/s4det/p3-seedset/batch1/mining")
OUT = Path("/mnt/data/s4det/human_window/window1")


def _load_jsonl(p: Path) -> list[dict]:
    return [orjson.loads(line) for line in p.open() if line.strip()] if p.exists() else []


def _emit(img: Path, stem: str, shapes: list[dict], out: Path) -> bool:
    if not shapes or (out / "labels" / f"{stem}.json").exists():
        return False  # 无候选不打包(纯负样本帧人工翻阅无意义)或人已编辑
    with Image.open(img) as im:
        w, h = im.size
    (out / "labels" / f"{stem}.json").write_text(
        json.dumps(_xany_json(img, shapes, w, h), ensure_ascii=False))
    dst = out / "images" / f"{stem}.jpg"
    if not dst.is_symlink():
        dst.symlink_to(img.resolve())
    return True


@app.command()
def run() -> None:
    """装配仲裁残量 + 批 1 兜底 → 统一人工窗口工程."""
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(exist_ok=True)
    mp = OUT / "manifest.jsonl"
    manifest = _load_jsonl(mp)  # 合并快照: 并入旧记录, 重跑不丢已编辑帧的条目
    seen = {m["stem"] for m in manifest}

    # A. 批 2 仲裁残量: 未支持共识聚类 → 类名? 候选
    for r in _load_jsonl(B2 / "arbitration_human.jsonl"):
        chain, stem = r["chain"], r["stem"]
        img = B2 / chain / "consensus" / "review" / f"{stem}.jpg"
        if not img.exists():
            logger.warning("缺图跳过: {}", img)
            continue
        with Image.open(img) as im:
            w, h = im.size
        shapes = [
            _shape(f"{chain}?", c["box"], w, h,
                   f"{chain} 未支持聚类: {', '.join(c['members'])}")
            for c in r.get("unsupported", [])
        ]
        key = f"b2_{chain}_{stem}"
        if _emit(img, key, shapes, OUT) and key not in seen:
            seen.add(key)
            manifest.append({"stem": key, "src": str(img),
                             "kind": "arbitration", "chain": chain})

    # B. 批 1 兜底: 校验器检出框 → 类名? 候选
    for chain in ("ebike", "garbage_bag"):
        cons = B1 / chain / "consensus"
        rows = defaultdict(list)
        for line in (cons / "validators.jsonl").open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("level") == "review":
                for src, vb in r["validators"].items():
                    for b in vb:
                        rows[r["stem"]].append((tuple(b[:4]), src))
        for stem, boxes in rows.items():
            img = cons / "review" / f"{stem}.jpg"
            if not img.exists():
                continue
            with Image.open(img) as im:
                w, h = im.size
            shapes = [_shape(f"{chain}?", b, w, h, f"校验器 {src}") for b, src in boxes]
            key = f"b1_{chain}_{stem}"
            if _emit(img, key, shapes, OUT) and key not in seen:
                seen.add(key)
                manifest.append({"stem": key, "src": str(img),
                                 "kind": "batch1_review", "chain": chain})

    (OUT / "manifest.jsonl").write_bytes(
        b"\n".join(orjson.dumps(m) for m in manifest) + b"\n")
    labels: set[str] = set()  # 候选类名全集; 去 ? 的正类形式一并收录(确认=去 ? 的编辑流程)
    for p in (OUT / "labels").glob("*.json"):
        for s in json.loads(p.read_text())["shapes"]:
            labels.add(s["label"])
            if s["label"].endswith("?"):
                labels.add(s["label"][:-1])
    (OUT / "labels.txt").write_text("\n".join(sorted(labels)) + "\n")
    (OUT / "README.md").write_text(
        "# 人工窗口 window1（批 2 仲裁残量 + 批 1 兜底，2026-09-26 装配）\n\n"
        "> 候选框均为「预标注优先」产物：确认→去掉标签里的 ?；否→删框；空标帧=负样本。\n"
        "> 编辑导出后交 jxl 会话回收（X-AnyLabeling YAML/JSON 均可）。\n")
    logger.info("人工窗口 {} 帧 → {}（labels 侧已有编辑的帧自动跳过）", len(manifest), OUT)


if __name__ == "__main__":
    app()
