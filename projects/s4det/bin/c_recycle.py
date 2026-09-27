#!/usr/bin/env python3
"""人工窗口回收（window1 编辑产物 → YOLO 训练标签）.

规则（C-execution window1 节在案）:
    确认: label 不带 `?` → 真值框(类名 → 18 类表 id)
    弃置: label 带 `?` 残留 → 未裁决, 落隔离台账(gate=human-window-unresolved)
    负样本: 帧内无确认框(含删光) → 空 YOLO 标签
产物: <win>/recycle/{images(symlink), labels(YOLO 归一化), manifest.jsonl}
防呆: 确认率 <10% 视为"用户尚未编辑"拒跑(BadParameter)——预跑会把全部候选误弃置。

典型用法:
    python c_recycle.py    # 用户编辑完成后回收
"""

import json
from collections import Counter
from pathlib import Path

import orjson
import typer
from loguru import logger

app = typer.Typer(add_completion=False, help="人工窗口回收.")

WIN = Path("/mnt/data/s4det/human_window/window1")
CLASSES = (WIN / "labels.txt").read_text().split()  # 候选菜单含 ? 形式, 取正类用类表
CLASS_TABLE = Path(
    "/mnt/data/s4det/p3-seedset/batch2/mining/dataset/classes.txt"
).read_text().split()  # 18 类权威表(id=id 位次)
QUARANTINE = Path("/mnt/data/s4det/quarantine/quarantine-ledger.jsonl")
MIN_CONFIRM_RATE = 0.10


def _bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    """4 角点绝对像素 → (x1,y1,x2,y2)。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


@app.command()
def run() -> None:
    """回收 window1 编辑产物 → YOLO 标签 + 隔离台账。"""
    labels_dir = WIN / "labels"
    jsons = sorted(labels_dir.glob("*.json"))
    stats: Counter = Counter()
    unresolved: list[dict] = []
    frames: list[dict] = []  # (stem, [(cls_id, box_norm)])
    for p in jsons:
        d = json.loads(p.read_text())
        w, h = d["imageWidth"], d["imageHeight"]
        confirmed: list[tuple[int, tuple[float, float, float, float]]] = []
        for s in d["shapes"]:
            lab = s["label"]
            if lab.endswith("?"):
                stats["dropped_unresolved"] += 1
                unresolved.append({"stem": p.stem, "label": lab,
                                   "box": list(_bbox(s["points"]))})
                continue
            if lab not in CLASS_TABLE:
                raise typer.BadParameter(f"{p.stem}: 确认类 {lab!r} 不在 18 类表——菜单漂移?")
            x1, y1, x2, y2 = _bbox(s["points"])
            confirmed.append((CLASS_TABLE.index(lab),
                              (x1 / w, y1 / h, x2 / w, y2 / h)))
        stats["confirmed"] += len(confirmed)
        stats["frames_with_truth" if confirmed else "frames_negative"] += 1
        frames.append({"stem": p.stem, "boxes": confirmed})

    n_conf_frames = stats["frames_with_truth"]
    if n_conf_frames / max(1, len(jsons)) < MIN_CONFIRM_RATE:
        raise typer.BadParameter(
            f"确认帧率 {n_conf_frames}/{len(jsons)} < {MIN_CONFIRM_RATE:.0%}——用户尚未编辑? 拒跑防误弃置")

    out = WIN / "recycle"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    manifest = []
    src_idx = {json.loads(line)["stem"]: json.loads(line)
               for line in (WIN / "manifest.jsonl").read_text().splitlines() if line.strip()}
    for f in frames:
        lines = [f"{c} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}"
                 for c, (x1, y1, x2, y2) in f["boxes"]]
        (out / "labels" / f"{f['stem']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        src = src_idx.get(f["stem"], {})
        img = WIN / "images" / f"{f['stem']}.jpg"
        dst = out / "images" / f"{f['stem']}.jpg"
        if not dst.is_symlink():
            dst.symlink_to(img.resolve())
        manifest.append({"stem": f["stem"], "src": src.get("src", str(img)),
                         "kind": src.get("kind", ""), "chain": src.get("chain", ""),
                         "n_boxes": len(f["boxes"])})
    (out / "manifest.jsonl").write_text(
        "\n".join(orjson.dumps(m).decode() for m in manifest) + "\n")

    if unresolved:
        with QUARANTINE.open("a") as q:
            for u in unresolved:
                q.write(orjson.dumps({
                    "id": f"window1/{u['stem']}", "batch": "2026-09-27-window1-recycle",
                    "gate": "human-window-unresolved",
                    "verdict": f"候选框未裁决残留 label={u['label']}",
                    "model": "human", "priority": "low", "recoverable": True,
                }).decode() + "\n")

    typer.echo(orjson.dumps({
        "frames": len(jsons), "truth_frames": stats["frames_with_truth"],
        "negative_frames": stats["frames_negative"],
        "confirmed_boxes": stats["confirmed"],
        "dropped_unresolved": stats["dropped_unresolved"],
        "out": str(out),
    }, option=orjson.OPT_INDENT_2).decode())
    logger.info("回收 → {}（未裁决 {} 框已落隔离台账）", out, stats["dropped_unresolved"])


if __name__ == "__main__":
    app()
