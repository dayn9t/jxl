#!/usr/bin/env python3
"""委员会 v2 真值裁决器（③④ 产出，2026-11-21 用户 doctrine）。

三族席位：阿里=qwen38-local（182 本地）/ 字节=doubao-lite / 智谱=glm-flash。
v6 与一切检测器票**不参与真值表决**（学生不给自己打分）。

帧级规则（框级聚类+族支持，非全有全无）：
  - 跨族框聚类 IoU≥0.5 传递闭包；簇=成员逐坐标中位
  - 簇获 ≥2 族支持 → 真值框；单族簇 → contested
  - 帧：到场族全空 → 负例真值；有 contested → 人工残余；全簇≥2 族 → 正例真值；
    到场族 <2 → 人工残余（证据不足）
  - 弃权族不阻塞计数也不计入支持

stem 命名空间天然隔离三面：`pt*/…`=b2、`n5b1_…`=b1prime、`gtmin__…`=gt_min。
产出：adjudicated.jsonl / human_residual.tsv（下周人裁）/ gt_yolo/labels/（真值训练形）。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Sequence

import typer

app = typer.Typer(add_completion=False, help="委员会 v2 真值裁决器（三族票）。")

SEAT_FAMILY = {"qwen38-local": "ali", "doubao-lite": "byte", "glm-flash": "zhipu"}
B2_ROOT = Path("/home/jiang/ws/sgcc-mma/training/person/datasets/n005-b2")
B1_ROOT = Path("/home/jiang/ws/sgcc-mma/training/person/datasets/n005-b1")


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def family_clusters(boxes_by_fam: dict[str, list], th: float = 0.5) -> list[dict]:
    """跨族框聚类（IoU≥th 传递闭包）→ [{box(中位), families}]。"""
    items = [(fam, tuple(b)) for fam, bs in boxes_by_fam.items() for b in bs]
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if find(a) != find(b) and iou(items[a][1], items[b][1]) >= th:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for idx in range(len(items)):
        groups.setdefault(find(idx), []).append(idx)
    out = []
    for members in groups.values():
        bs = [items[m][1] for m in members]
        fams = sorted({items[m][0] for m in members})
        out.append({"box": [round(statistics.median([b[k] for b in bs]), 6) for k in range(4)],
                    "families": fams})
    out.sort(key=lambda c: -len(c["families"]))
    return out


def adjudicate(boxes_by_fam: dict[str, list], th: float) -> tuple[str, list[dict], int]:
    """帧三族票 → (label, gt_clusters, n_contested)。label∈{pos,neg,residual}。"""
    present = {f: bs for f, bs in boxes_by_fam.items() if f in set(SEAT_FAMILY.values())}
    if len(present) < 2:
        return "residual", [], 0
    if not any(present.values()):
        return "neg", [], 0
    cls = family_clusters(present, th)
    gt = [c for c in cls if len(c["families"]) >= 2]
    cont = [c for c in cls if len(c["families"]) < 2]
    return ("residual", gt, len(cont)) if cont else ("pos", gt, 0)


def _load(path: Path) -> dict[str, tuple[dict[str, list], dict[str, int]]]:
    """labels.jsonl → {stem: ({family: boxes}, {family: 弃权数})}；缺文件=空。

    席位=路径约定：…/prelabels/<seat>/labels.jsonl 或 …/committee_v2/<seat>.labels.jsonl。
    """
    out: dict[str, tuple[dict[str, list], dict[str, int]]] = {}
    if not path.is_file():
        return out
    seat = path.parent.name if path.parent.name in SEAT_FAMILY else path.stem.split(".")[0]
    if seat not in SEAT_FAMILY:
        return out
    fam = SEAT_FAMILY[seat]
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ent = out.setdefault(r["stem"], ({}, {}))
        if r["ok"]:
            ent[0][fam] = r["boxes"]
        else:
            ent[1][fam] = ent[1].get(fam, 0) + 1
    return out


def collect_votes() -> dict[str, tuple[dict[str, list], dict[str, int]]]:
    """三面票面汇集（stem 命名空间天然隔离，无换算）。"""
    merged: dict[str, tuple[dict[str, list], dict[str, int]]] = {}

    def absorb(src: dict) -> None:
        for stem, (boxes, errs) in src.items():
            tgt = merged.setdefault(stem, ({}, {}))
            tgt[0].update(boxes)
            for f, c in errs.items():
                tgt[1][f] = tgt[1].get(f, 0) + c

    # b2：qwen38/glm 在 batch prelabels；lite 在 committee_v2（斜杠 stem）
    for batch in ("base", "mine"):
        absorb(_load(B2_ROOT / batch / "prelabels" / "qwen38-local" / "labels.jsonl"))
        absorb(_load(B2_ROOT / batch / "prelabels" / "glm-flash" / "labels.jsonl"))
    # b1prime：同 b1 prelabels 形（n5b1_ stem）
    absorb(_load(B1_ROOT / "prelabels" / "qwen38-local" / "labels.jsonl"))
    absorb(_load(B1_ROOT / "prelabels" / "glm-flash" / "labels.jsonl"))
    # committee_v2 目录：doubao-lite（斜杠+gtmin__ 混面）+ gt_min 双免费席
    cv2 = B2_ROOT / "committee_v2"
    for seat in ("doubao-lite", "qwen38-local", "glm-flash"):
        src = _load(cv2 / f"{seat}.labels.jsonl")
        if seat != "doubao-lite":  # committee_v2 的免费席只收 gtmin__（b2/b1 已从他源取，防覆）
            src = {k: v for k, v in src.items() if k.startswith("gtmin__")}
        absorb(src)
    return merged


@app.command()
def run(
    out_dir: Path = typer.Option(B2_ROOT / "truth_v2", "--out-dir"),
    th: float = typer.Option(0.5, "--iou", help="跨族配对门（doctrine：≥2 族 IoU≥0.5=基本一致）"),
) -> None:
    """三族票裁决 → 真值标签包 + 人工残余清单 + 三堆计数。"""
    votes = collect_votes()
    lite_stems = {s for s, (b, _) in votes.items() if "byte" not in b and not s.startswith("gtmin__")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_yolo" / "labels").mkdir(parents=True, exist_ok=True)
    n = {"pos": 0, "neg": 0, "residual": 0}
    adj_rows, res_rows = [], []
    for stem, (boxes, errs) in sorted(votes.items()):
        label, gt, ncont = adjudicate(boxes, th)
        n[label] += 1
        adj_rows.append({"stem": stem, "label": label, "families_present": sorted(boxes),
                         "abstains": errs, "gt_boxes": gt, "n_contested": ncont})
        if label == "residual":
            res_rows.append((stem, ncont + max(0, 2 - len(boxes)), len(boxes)))
        else:
            lines = []
            for c in gt:
                x1, y1, x2, y2 = c["box"]
                lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            (out_dir / "gt_yolo" / "labels" / f"{stem.replace('/', '__')}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
    (out_dir / "adjudicated.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in adj_rows))
    res_rows.sort(key=lambda x: (-x[1], x[0]))
    (out_dir / "human_residual.tsv").write_text(
        "stem\tn_disagreement\tn_families_present\n"
        + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in res_rows))
    stats = {"帧": sum(n.values()), **n, "缺 lite 票帧": len(lite_stems)}
    (out_dir / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    typer.secho(json.dumps(stats, ensure_ascii=False), fg=typer.colors.GREEN)
    typer.echo(f"→ {out_dir}")


if __name__ == "__main__":
    app()
