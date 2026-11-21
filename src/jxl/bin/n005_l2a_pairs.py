#!/usr/bin/env python3
"""n005 L2a ReID 对库生成（模型阶梯战役 L2 初级处理，2026-11-21 iap-c1 升级令）。

模板位：j-model-ladder ①「程序+本地模型初级处理」——段聚合/128d 聚类/对初判全
零 API；委员会（②）只对本产物的分歧面+抽验复核（另线消费）。

产物（--out-root 下）：
  segments.jsonl           L3 段表（在场区间，绝对 epoch us，间隙≤gap 并段，≥min 留）
  same_candidates.jsonl    段内同人对（簇代表采样）
  diff_candidates.jsonl    段内多簇换人对
  drift_same_pairs.jsonl   姿态漂移对=簇心 cos≥thr 跨段≥gapmin（complete-linkage，
                           避 handcrafted 传递闭包全局塌缩——NOTE 实锤）
  segment_clusters.jsonl   段内簇表

⚠️ 缓存 pts_us 为文件内相对时基——一切跨文件时序运算必须经 base_us 换绝对 epoch。
"""

from __future__ import annotations

import calendar
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import orjson
import typer

app = typer.Typer(add_completion=False, help="n005 L2a ReID 对库生成（零成本初级处理）。")

VIDEO = Path("/var/howell/sgcc-mma/v0.13/iap/n005/output/video")


def _base_us(date: str, tstem: str) -> float:
    """`2026-11-06` + `08-10-00.000` → epoch us（绝对时基）。"""
    h, m, rest = tstem.split("-")
    s, ms = rest.split(".")
    sec = calendar.timegm(time.strptime(f"{date} {h}:{m}:{s}", "%Y-%m-%d %H:%M:%S"))
    return (sec + int(ms) / 1000) * 1e6


def _nrm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def _greedy_closure(sim: np.ndarray, th: float) -> np.ndarray:
    labels = np.full(len(sim), -1)
    cid = 0
    for i in range(len(sim)):
        if labels[i] >= 0:
            continue
        stack = [i]
        while stack:
            x = stack.pop()
            if labels[x] >= 0:
                continue
            labels[x] = cid
            cand = np.where(labels == -1)[0]
            stack.extend(cand[sim[x][cand] >= th].tolist())
        cid += 1
    return labels


def _frame_index(segs: list[dict], video: Path) -> dict[int, dict[tuple[str, str], dict]]:
    """按 (date, tstem) 缓存 json 索引，供段观测收集（避免逐段重扫全目录）。"""
    idx: dict[int, dict] = {}
    for src in range(1, 7):
        per: dict[tuple[str, str], dict] = {}
        for jf in sorted((video / str(src) / "0").glob("*/*.json")):
            per[(jf.parent.name, jf.stem)] = {"path": jf, "base": _base_us(jf.parent.name, jf.stem)}
        idx[src] = per
    return idx


def collect_obs(seg: dict, vidx: dict, cov_cache: dict) -> list[dict]:
    src = seg["src"]
    day = time.strftime("%Y-%m-%d", time.gmtime(seg["start_us"] / 1e6))
    out: list[dict] = []
    for (d, tstem), ent in vidx[src].items():
        if d != day:
            continue
        if ent["base"] > seg["end_us"] + 60e6 or ent["base"] + 600e6 < seg["start_us"]:
            continue
        key = (src, d, tstem)
        if key not in cov_cache:
            cov_cache[key] = orjson.loads(ent["path"].read_bytes())["frames"]
        for i, fr in enumerate(cov_cache[key]):
            t = ent["base"] + fr["pts_us"]
            if not (seg["start_us"] <= t <= seg["end_us"]):
                continue
            for j, det in enumerate(fr["detections"]):
                out.append({"uid": f"{d}/{tstem}#{i}#{j}", "t_us": t,
                            "vec": det["embedding"], "bbox": det["bbox"],
                            "conf": det["confidence"]})
    return out


@app.command()
def segments(
    out_root: Path = typer.Option(Path("~/ws/sgcc-mma/training/person/datasets/n005-session-gold-v1").expanduser()),
    gap_s: float = typer.Option(120, help="并段最大间隙秒"),
    min_s: float = typer.Option(30, help="段最短秒"),
) -> None:
    """在场区间段表（缓存自含，程序聚合）。"""
    segs: list[dict] = []
    for src in range(1, 7):
        for jf in sorted((VIDEO / str(src) / "0").glob("*/*.json")):
            d = orjson.loads(jf.read_bytes())
            b = _base_us(jf.parent.name, jf.stem)
            cur = None
            for fr in d["frames"]:
                if fr["detections"]:
                    if cur is None:
                        cur = [b + fr["pts_us"], b + fr["pts_us"], 1]
                    else:
                        cur[1] = b + fr["pts_us"]; cur[2] += 1
                elif cur is not None:
                    segs.append({"src": src, "date": jf.parent.name,
                                 "start_us": cur[0], "end_us": cur[1], "n_on": cur[2]})
                    cur = None
            if cur:
                segs.append({"src": src, "date": jf.parent.name,
                             "start_us": cur[0], "end_us": cur[1], "n_on": cur[2]})
    by = defaultdict(list)
    for s in segs:
        by[s["src"]].append(s)
    merged: list[dict] = []
    for src, ss in sorted(by.items()):
        ss.sort(key=lambda x: x["start_us"])
        cur = None
        for s in ss:
            if cur and s["start_us"] - cur["end_us"] <= gap_s * 1e6:
                cur["end_us"] = max(cur["end_us"], s["end_us"]); cur["n_on"] += s["n_on"]
            else:
                if cur:
                    merged.append(cur)
                cur = dict(s)
        if cur:
            merged.append(cur)
    final = [dict(m, dur_s=round((m["end_us"] - m["start_us"]) / 1e6, 1)) for m in merged
             if m["end_us"] - m["start_us"] >= min_s * 1e6]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "segments.jsonl").write_text(
        "".join(orjson.dumps(m).decode() + "\n" for m in final))
    typer.secho(f"段 {len(final)} → {out_root}/segments.jsonl", fg=typer.colors.GREEN)


@app.command()
def pairs(
    session_root: Path = typer.Option(Path("~/ws/sgcc-mma/training/person/datasets/n005-session-gold-v1").expanduser()),
    out_root: Path = typer.Option(Path("~/ws/sgcc-mma/training/person/datasets/n005-l2a-pairs-v0").expanduser()),
    cos_seg: float = typer.Option(0.65, help="段内聚类 cos 门"),
    cos_drift: float = typer.Option(0.70, help="漂移对簇心 cos 门"),
    gap_min: float = typer.Option(15, help="漂移对最小跨段分钟"),
    max_same_per_cluster: int = typer.Option(10),
) -> None:
    """段内簇→same/diff 对 + 跨段簇心漂移对（complete-linkage 防全局塌缩）。"""
    segs = [orjson.loads(l) for l in (session_root / "segments.jsonl").read_text().splitlines() if l.strip()]
    vidx = _frame_index(segs, VIDEO)
    cov: dict = {}
    same, diff, clusters, centroids = [], [], [], []
    for seg in segs:
        obs = collect_obs(seg, vidx, cov)
        if not obs:
            continue
        A = np.array([o["vec"] for o in obs], dtype=np.float32)
        sim = _nrm(A) @ _nrm(A).T
        labels = _greedy_closure(sim, cos_seg)
        cid = int(labels.max()) + 1
        clusters.append({"seg": {"src": seg["src"], "date": seg["date"], "start_us": seg["start_us"]},
                         "n_obs": len(obs), "n_clusters": cid,
                         "sizes": [int((labels == c).sum()) for c in range(cid)]})
        for c in range(cid):
            idx = np.where(labels == c)[0]
            mid = int(idx[len(idx) // 2])
            centroids.append({"src": seg["src"], "seg_us": seg["start_us"], "cid": c,
                              "center": A[idx].mean(0), "n": len(idx),
                              "uid": obs[mid]["uid"], "bbox": obs[mid]["bbox"]})
            if len(idx) < 2:
                continue
            reps = idx[:: max(1, len(idx) // 5)][:5]
            cnt = 0
            for ai in range(len(reps)):
                for bi in range(ai + 1, len(reps)):
                    if cnt >= max_same_per_cluster:
                        break
                    same.append({"src": seg["src"], "cluster": f"{seg['start_us']}-{c}",
                                 "a": obs[int(reps[ai])]["uid"], "b": obs[int(reps[bi])]["uid"],
                                 "cos": round(float(sim[reps[ai]][reps[bi]]), 4)})
                    cnt += 1
        if cid >= 2:
            r1 = int(np.where(labels == 0)[0][0]); r2 = int(np.where(labels == 1)[0][0])
            diff.append({"src": seg["src"], "seg": str(seg["start_us"]),
                         "a": obs[r1]["uid"], "b": obs[r2]["uid"],
                         "cos": round(float(sim[r1][r2]), 4)})
    C = _nrm(np.array([c["center"] for c in centroids], dtype=np.float32))
    S = C @ C.T
    drift = []
    for a, b in combinations(range(len(centroids)), 2):
        ca, cb = centroids[a], centroids[b]
        if ca["src"] != cb["src"] or ca["seg_us"] == cb["seg_us"]:
            continue
        gap = abs(ca["seg_us"] - cb["seg_us"])
        if S[a][b] >= cos_drift and gap >= gap_min * 60e6:
            drift.append({"src": ca["src"], "cos_center": round(float(S[a][b]), 4),
                          "a": ca["uid"], "a_bbox": ca["bbox"],
                          "b": cb["uid"], "b_bbox": cb["bbox"],
                          "gap_min": round(gap / 6e7, 1), "n_a": ca["n"], "n_b": cb["n"]})
    out_root.mkdir(parents=True, exist_ok=True)
    for name, rows in [("same_candidates", same), ("diff_candidates", diff),
                       ("drift_same_pairs", drift), ("segment_clusters", clusters)]:
        (out_root / f"{name}.jsonl").write_text(
            "".join(orjson.dumps(r).decode() + "\n" for r in rows))
    typer.secho(f"same {len(same)} / diff {len(diff)} / drift {len(drift)} / 簇 {len(centroids)}"
                f" → {out_root}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
