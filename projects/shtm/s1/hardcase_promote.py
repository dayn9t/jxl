#!/usr/bin/env python3
"""S1 难例削减·晋升（隔离池 → 正式集 s1_relabel_v1，2026-09-11）。

前置（用户裁决 2026-09-10）：自动判定是未定状态，必须过两道闸才入正式集——
  1. 删框拼图人工否决（veto 文件）+ agent 复审 flag 仲裁结果，合并于
     s1/hardcase_veto.jsonl（行格式 {rel, idx, why}）
  2. 用户明确批准晋升（跑本脚本即视为已获批准，脚本不询问）
被否决对象的语义：该对象的自动判定作废 → 从备份恢复原对象回 review vlabels，
且其所在帧改为残留帧并入 r2 人工审核（帧不晋升）。
其余：纯自动 hardcase 帧 + backlog 全自动帧（接受/负样本）拷入 s1_relabel_v1。

幂等：写标记 s1/.hardcase_promoted.json，重跑需 --force；已有 backlog 帧亦拒绝。
用法：uv run --project /home/jiang/cc/py/jxl python hardcase_promote.py [--force]
"""
import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

sys_parent = Path(__file__).parent
import s1_vlabel_merge as svm  # noqa: E402  (备份读取经 src_vlabels)
from hardcase_prune import AUTO, R2, REVIEW, S1, SNAP, src_vlabels  # noqa: E402

S1R = Path("/home/jiang/ws/trash/s1_relabel_v1")

VETO = S1 / "hardcase_veto.jsonl"
MARK = S1 / ".hardcase_promoted.json"


def load_veto() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if VETO.exists():
        for line in VETO.open():
            if line.strip():
                d = json.loads(line)
                out[f"{d['rel']}|{d['idx']}"] = d
    return out


def ensure_not_promoted(force: bool) -> None:
    bl_stems = {json.loads(l)["rel"].replace("/", "_")
                for l in open(S1 / "det_track_backlog.jsonl")}
    hit = [s for s in bl_stems if (S1R / "vlabels" / f"{s}.json5").exists()]
    if hit and not force:
        raise SystemExit(f"backlog 帧已存在于正式集 {len(hit)} 个，疑似已晋升过；--force 覆盖")
    if MARK.exists() and not force:
        raise SystemExit(f"已晋升过（{MARK}）；--force 重跑")


def restore_object(stem: str, idx: int) -> dict:
    """从备份取被否决对象的原始态（备份=预填态，单一事实源）。"""
    src = src_vlabels() / f"{stem}.json5"
    lab = json.load(open(src))
    return lab["objects"][idx]


def apply_vetoes(veto: dict[str, dict]) -> tuple[set[str], set[str]]:
    """否决对象恢复回 review vlabels；返回（被否决的 hardcase 帧, backlog 帧）。"""
    hc_frames: set[str] = set()
    bl_frames: set[str] = set()
    by_frame: dict[str, list[int]] = {}
    for key in veto:
        rel, idx = key.rsplit("|", 1)
        by_frame.setdefault(rel, []).append(int(idx))
    for rel, idxs in by_frame.items():
        stem = rel.replace("/", "_")
        p = REVIEW / "vlabels" / f"{stem}.json5"
        lab = json.load(open(p))
        for idx in idxs:
            obj = restore_object(stem, idx)
            obj["id"] = len(lab["objects"])
            lab["objects"].append(obj)
            row = next((json.loads(l) for l in open(S1 / "hardcase_auto.jsonl")
                        if f'"{rel}"' in l and f'"idx": {idx},' in l), None)
            if row is not None and row["kind"] == "backlog":
                bl_frames.add(rel)
            else:
                hc_frames.add(rel)
        lab["objects"] = [{**o, "id": i} for i, o in enumerate(lab["objects"])]
        p.write_text(json.dumps(lab, ensure_ascii=False, indent=1) + "\n")
    return hc_frames, bl_frames


def promote(quarantine_frames: set[str], hc_veto: set[str], bl_veto: set[str]) -> Counter:
    st: Counter = Counter()
    q_hardcase = {p.stem for p in (AUTO / "vlabels").glob("*.json5")}
    bl_all = {json.loads(l)["rel"].replace("/", "_")
              for l in open(S1 / "det_track_backlog.jsonl")}
    bl_q = q_hardcase & bl_all
    hc_q = q_hardcase - bl_all
    hc_veto_stems = {r.replace("/", "_") for r in hc_veto}
    bl_veto_stems = {r.replace("/", "_") for r in bl_veto}
    for stem in sorted(hc_q):
        if stem in hc_veto_stems:
            st["hc_frames_to_r2"] += 1
            continue
        shutil.copy(REVIEW / "vlabels" / f"{stem}.json5", S1R / "vlabels" / f"{stem}.json5")
        st["hc_frames_promoted"] += 1
    for stem in sorted(bl_q):
        if stem in bl_veto_stems:
            st["bl_frames_to_r2"] += 1
            continue
        shutil.copy(AUTO / "vlabels" / f"{stem}.json5", S1R / "vlabels" / f"{stem}.json5")
        st["bl_frames_promoted"] += 1
    return st


def extend_r2(hc_veto: set[str], bl_veto: set[str], stem2rel: dict[str, str]) -> None:
    """被否决帧并入 r2（vlabel 已含恢复对象；backlog 帧需从零构建含接受框上下文）。"""
    ledger = [json.loads(l) for l in open(S1 / "hardcase_auto.jsonl")]
    per_bl: dict[str, list[dict]] = {}
    for row in ledger:
        if row["kind"] == "backlog":
            per_bl.setdefault(row["rel"], []).append(row)
    for rel in sorted(bl_veto):
        stem = rel.replace("/", "_")
        lab_src = AUTO / "vlabels" / f"{stem}.json5"
        if lab_src.exists():
            shutil.copy(lab_src, R2 / "vlabels" / f"{stem}.json5")
        if not (R2 / "images" / f"{stem}.jpg").exists():
            os.symlink(SNAP / f"{rel}.jpg", R2 / "images" / f"{stem}.jpg")
    for rel in sorted(hc_veto):
        stem = rel.replace("/", "_")
        shutil.copy(REVIEW / "vlabels" / f"{stem}.json5", R2 / "vlabels" / f"{stem}.json5")
        if not (R2 / "images" / f"{stem}.jpg").exists():
            os.symlink(SNAP / f"{rel}.jpg", R2 / "images" / f"{stem}.jpg")
    print(f"r2 扩充: hardcase {len(hc_veto)} + backlog {len(bl_veto)} 帧"
          f"（backlog 帧内被否决框需人工补行动，见 hardcase_veto.jsonl）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_not_promoted(args.force)
    veto = load_veto()
    hc_veto, bl_veto = apply_vetoes(veto)
    print(f"veto: {len(veto)} 条（hardcase 帧 {len(hc_veto)} / backlog 帧 {len(bl_veto)}）")
    stem2rel = {}
    for l in open(S1 / "hardcase_auto.jsonl"):
        d = json.loads(l)
        stem2rel[d["rel"].replace("/", "_")] = d["rel"]
    st = promote(set(stem2rel), hc_veto, bl_veto)
    extend_r2(hc_veto, bl_veto, stem2rel)
    MARK.write_text(json.dumps({"promoted_at": datetime.now().astimezone().isoformat(),
                                "veto_count": len(veto)}, ensure_ascii=False) + "\n")
    print("promote:", dict(st))


if __name__ == "__main__":
    main()
