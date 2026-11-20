#!/usr/bin/env python3
"""n005 检测线自动化标注·批次1（2026-11-19 夜役，iap-4c 派遣/用户夜令）。

三源共识：v6 检测缓存（免费第五票，person_n_v6@0.5 半分口径）+ 免费双 VLM 票
（qwen3.8-flash-next@182 强票 + glm-5.3-flash 套餐仲裁票）→ 已确认子集（YOLO 形
传 n902 训练位）+ 难例子集（ulabel 集 + quarantine.tsv，明早人工裁决）。

席位/协议/解析/坐标刻度单一数据源 = jxl.bin.vlm_pool（本工具零重复定义）。
决策规则（跨源框聚类 IoU≥0.4 传递闭包）：
  - 簇支持源数 ≥2 → 确认框（坐标=成员逐坐标中位数）
  - 全源空 → 确认空帧（负样本）
  - 任一单源框 / 弃权过半 → 整帧难例（quarantine，禁自动收——⑧ 硬性规则）

子命令正序：select-extract → vote → aggregate → seed-ulabel。
"""

from __future__ import annotations

import asyncio
import math
import random
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx
import orjson
import typer

from jxl.bin.vlm_pool import (
    CANDIDATES,
    Vote,
    build_target_prompt,
    call_person,
    image_data_url,
)
from jxl.label.crop import xyxy_iou as iou

app = typer.Typer(add_completion=False, help="n005 检测线自动化标注·批次1（三源共识夜役）")

VIDEO_ROOT = Path("/var/howell/sgcc-mma/v0.13/iap/n005/output/video")
OUT_ROOT = Path("~/ws/sgcc-mma/training/person/datasets/n005-b1").expanduser()
SOURCES = (1, 2, 3, 4, 5, 6)
PERSON_QUOTA_DEFAULT = (180, 180, 140, 40, 30, 30)
EMPTY_QUOTA_DEFAULT = (60, 60, 45, 35, 30, 30)
GAP_US = 60_000_000  # 帧内去相关：60s
PER_FILE_MAX = 8
FRAME_W, FRAME_H = 960, 540  # 分辨率门控（1080p 源 ÷2，KB 20260927 正典）
SEATS_FREE = ("qwen38-local", "glm-flash")
CACHE_SEAT = "v6cache"
CLUSTER_IOU = 0.4


# ---------------- 域模型 ----------------

@dataclass(frozen=True, slots=True)
class FrameEntry:
    """缓存 json 内单个采样帧。"""

    pts_us: int
    wallclock: str  # "2026-11-06/09-00-00.000"
    boxes_norm: tuple[tuple[float, float, float, float], ...]
    confs: tuple[float, ...]
    uppers: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class FileCand:
    """一个 mkv+json 对的全部候选帧。"""

    src: int
    date: str
    mkv: Path
    w: int
    h: int
    entries: tuple[FrameEntry, ...]


@dataclass(frozen=True, slots=True)
class SelRow:
    """selection.jsonl 行。"""

    stem: str
    src: int
    date: str
    mkv: Path
    pts_us: int
    wallclock: str
    boxes_norm: tuple[tuple[float, float, float, float], ...]
    confs: tuple[float, ...]
    uppers: tuple[bool, ...]


def _stem(src: int, wallclock: str, pts_us: int) -> str:
    """全局唯一且无点号（ulabel tsv 盲 rsplit 砍秒段教训 → 禁 '.'）。"""
    date, clock = wallclock.split("/")
    d = date.replace("-", "")
    c = clock.replace("-", "").split(".")[0]  # 090000
    return f"n5b1_s{src}_{d}_{c}_{pts_us // 10_000:06d}"


# ---------------- select-extract ----------------

def _load_files(src: int, video_root: Path) -> tuple[list[FileCand], int]:
    """读一源全部缓存 json → FileCand；孤儿 json（无 mkv）计数跳过。"""
    out: list[FileCand] = []
    orphans = 0
    for jf in sorted((video_root / str(src) / "0").glob("*/*.json")):
        mkv = jf.with_suffix(".mkv")
        if not mkv.is_file():
            orphans += 1
            continue
        d = orjson.loads(jf.read_bytes())
        entries = tuple(
            FrameEntry(
                pts_us=int(fr["pts_us"]),
                wallclock=str(fr["time"]),
                boxes_norm=tuple(
                    (
                        float(b["bbox"][0]) / d["width"],
                        float(b["bbox"][1]) / d["height"],
                        float(b["bbox"][2]) / d["width"],
                        float(b["bbox"][3]) / d["height"],
                    )
                    for b in fr["detections"]
                ),
                confs=tuple(float(b["confidence"]) for b in fr["detections"]),
                uppers=tuple(bool(b.get("upper_body")) for b in fr["detections"]),
            )
            for fr in d["frames"]
        )
        out.append(FileCand(src, jf.parent.name, mkv, int(d["width"]), int(d["height"]), entries))
    return out, orphans


def _select_src(
    files: Sequence[FileCand], person_q: int, empty_q: int, rng: random.Random
) -> list[SelRow]:
    """确定性选帧：文件随机序 × 帧内 60s 格点采样（随机相位）→ 配额填充。"""
    sels: list[SelRow] = []
    for fc in rng.sample(list(files), len(files)):
        if person_q <= 0 and empty_q <= 0:
            break
        step = max(1, GAP_US // 2_000_000)  # 缓存采样 2s → 60s 步长
        phase = rng.randrange(0, step)
        taken = 0
        for e in fc.entries[phase::step]:
            if taken >= PER_FILE_MAX or (person_q <= 0 and empty_q <= 0):
                break
            is_person = bool(e.boxes_norm)
            if is_person and person_q <= 0:
                continue
            if not is_person and empty_q <= 0:
                continue
            row = SelRow(
                stem=_stem(fc.src, e.wallclock, e.pts_us), src=fc.src, date=fc.date,
                mkv=fc.mkv, pts_us=e.pts_us, wallclock=e.wallclock,
                boxes_norm=e.boxes_norm, confs=e.confs, uppers=e.uppers,
            )
            sels.append(row)
            taken += 1
            if is_person:
                person_q -= 1
            else:
                empty_q -= 1
    return sels


async def _extract_one(stem: str, mkv: Path, pts_us: int, frames_dir: Path) -> str:
    """ffmpeg 精确 seek 提单帧缩 960×540；返回 stem 或空串（失败）。"""
    out = frames_dir / f"{stem}.jpg"
    if out.exists() and out.stat().st_size > 4_096:
        return stem  # 幂等重跑
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{pts_us / 1e6:.6f}", "-i", str(mkv),
        "-frames:v", "1", "-vf", f"scale={FRAME_W}:{FRAME_H}",
        "-q:v", "3", str(out),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not out.exists() or out.stat().st_size <= 4_096:
        out.unlink(missing_ok=True)
        typer.secho(f"extract fail {stem}: {err.decode()[:120]}", fg=typer.colors.YELLOW, err=True)
        return ""
    return stem


@app.command()
def select_extract(
    video_root: Path = typer.Option(VIDEO_ROOT, "--video-root"),
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    person: str = typer.Option(",".join(map(str, PERSON_QUOTA_DEFAULT)), "--person", help="六源有人帧配额"),
    empty: str = typer.Option(",".join(map(str, EMPTY_QUOTA_DEFAULT)), "--empty", help="六源无人帧配额"),
    seed: int = typer.Option(20261119, "--seed"),
    concurrency: int = typer.Option(8, "--extract-concurrency"),
    smoke: int = typer.Option(0, "--smoke", help=">0 每源只选 N 帧（端到端冒烟）"),
) -> None:
    """分层选帧（缓存引导）→ 提帧 960×540 → selection.jsonl。"""
    person_q = [int(v) for v in person.split(",")]
    empty_q = [int(v) for v in empty.split(",")]
    if smoke:
        person_q = [min(smoke, v) for v in person_q]
        empty_q = [min(smoke, v) for v in empty_q]
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[SelRow] = []
    rng = random.Random(seed)
    for i, src in enumerate(SOURCES):
        files, orphans = _load_files(src, video_root)
        bad = [f for f in files if (f.w, f.h) != (1920, 1080)]
        if bad:
            raise typer.BadParameter(f"src{src} 非预期分辨率 {bad[0].w}x{bad[0].h}（{len(bad)} 文件）")
        sel = _select_src(files, person_q[i], empty_q[i], rng)
        typer.echo(
            f"src{src}: 文件 {len(files)}（孤儿 json {orphans}）选中 {len(sel)} "
            f"（有人 {sum(1 for r in sel if r.boxes_norm)} / 无人 {sum(1 for r in sel if not r.boxes_norm)}）"
        )
        rows.extend(sel)
    sem = asyncio.Semaphore(concurrency)

    async def guarded(r: SelRow) -> str:
        async with sem:
            return await _extract_one(r.stem, r.mkv, r.pts_us, frames_dir)

    async def _all() -> list[str]:
        return list(await asyncio.gather(*[guarded(r) for r in rows]))

    t0 = time.monotonic()
    results = asyncio.run(_all())
    ok = [r for r, s in zip(rows, results) if s]
    fail = [r.stem for r, s in zip(rows, results) if not s]
    dump(out_root / "selection.jsonl", [
        {"stem": r.stem, "src": r.src, "date": r.date, "mkv": str(r.mkv),
         "pts_us": r.pts_us, "wallclock": r.wallclock,
         "boxes_norm": [list(b) for b in r.boxes_norm], "confs": list(r.confs),
         "uppers": list(r.uppers)} for r in ok
    ])
    typer.secho(
        f"选中 {len(rows)} → 提帧成功 {len(ok)} 失败 {len(fail)} "
        f"（{time.monotonic() - t0:.0f}s）→ {frames_dir}",
        fg=typer.colors.GREEN,
    )
    if fail:
        (out_root / "extract_failures.json").write_text(orjson.dumps(fail).decode())


# ---------------- vote ----------------

def dump(path: Path, rows: Sequence[dict]) -> None:
    path.write_text("".join(orjson.dumps(r).decode() + "\n" for r in rows))


def vote_seat(
    seat: str, rows: Sequence[dict], out_root: Path, concurrency: int
) -> tuple[int, int]:
    """单席位逐帧投票（断点续跑：已有 stem 跳过）；原始响应逐字落盘（#16）。"""
    pre = out_root / "prelabels" / seat
    pre.mkdir(parents=True, exist_ok=True)
    path = pre / "labels.jsonl"
    done: set[str] = set()
    if path.exists():
        # 续跑只认 ok 票：弃权/失败记录重跑时重试（不留永久弃权死角）
        done = {r["stem"] for r in
                (orjson.loads(line) for line in path.read_text().splitlines() if line.strip())
                if r["ok"]}
    todo = [r for r in rows if r["stem"] not in done]
    fh = path.open("a")
    ok = err = 0

    async def one(client: httpx.AsyncClient, r: dict) -> None:
        nonlocal ok, err
        t0 = time.monotonic()

        async def _call() -> Vote:
            async with sem:
                url = image_data_url(out_root / "frames" / f"{r['stem']}.jpg")
                return await call_person(client, seat, url, build_target_prompt("person"))

        # 硬护栏：整任务（信号量+编码+POST）180s 强制超时——vLLM 偶发永不应答
        # 且 httpx 超时不触发的实证（2026-11-20 夜，42 单后六连接挂死 6 分钟）
        try:
            v: Vote | None = await asyncio.wait_for(_call(), timeout=180.0)
        except TimeoutError:
            v = None
        if v is None:
            rec = {"stem": r["stem"], "ok": False, "ms": int((time.monotonic() - t0) * 1000),
                   "boxes": [], "error": "wait_for timeout 180s", "raw": ""}
            fh.write(orjson.dumps(rec).decode() + "\n")
            fh.flush()
            err += 1
            return
        rec = {"stem": r["stem"], "ok": v.ok, "ms": int((time.monotonic() - t0) * 1000),
               "boxes": [list(b) for b in v.boxes], "error": v.error, "raw": v.raw}
        fh.write(orjson.dumps(rec).decode() + "\n")
        fh.flush()
        if v.ok:
            ok += 1
        else:
            err += 1

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            for i in range(0, len(todo), concurrency):
                await asyncio.gather(*[one(client, r) for r in todo[i:i + concurrency]])
                typer.echo(f"{seat}: {min(i + concurrency, len(todo))}/{len(todo)}", err=True)

    sem = asyncio.Semaphore(concurrency)
    if todo:
        asyncio.run(run())
    fh.close()
    return ok, err


@app.command()
def vote(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    seats: str = typer.Option(",".join(SEATS_FREE), "--seats"),
    concurrency: int = typer.Option(6, "--concurrency"),
    limit: int = typer.Option(0, "--limit", help=">0 只投前 N 帧（席位冒烟）"),
) -> None:
    """三票投票：v6 缓存票（selection 直转）+ 免费 VLM 双票（断点续跑）。"""
    rows = [orjson.loads(line) for line in (out_root / "selection.jsonl").read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    # 缓存票：确定性、免调用
    cache_dir = out_root / "prelabels" / CACHE_SEAT
    cache_dir.mkdir(parents=True, exist_ok=True)
    dump(cache_dir / "labels.jsonl", [
        {"stem": r["stem"], "ok": True, "ms": 0, "boxes": r["boxes_norm"],
         "error": "", "raw": "", "confs": r["confs"]} for r in rows
    ])
    unknown = [s for s in seats.split(",") if s not in CANDIDATES]
    if unknown:
        raise typer.BadParameter(f"未知席位: {unknown}")
    for seat in seats.split(","):
        ok, err = vote_seat(seat, rows, out_root, concurrency)
        rate = ok / max(1, ok + err)
        typer.echo(f"{seat}: 新投 {ok + err}，ok率 {rate:.1%}（累计含续跑 {len(rows)}）")
        if rate < 0.8 and ok + err > 5:
            typer.secho(f"{seat} 弃权率超 20%——席位异常，停查", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)


# ---------------- aggregate ----------------

def clusters(
    votes: dict[str, list[tuple[float, float, float, float]]], iou_th: float
) -> list[dict]:
    """跨源框聚类（IoU≥th 传递闭包）→ [{box(median), seats, members}]。"""
    items = [(seat, b) for seat, boxes in votes.items() for b in boxes]
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if find(a) != find(b) and iou(items[a][1], items[b][1]) >= iou_th:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for idx in range(len(items)):
        groups.setdefault(find(idx), []).append(idx)
    out = []
    for members in groups.values():
        boxes = [items[m][1] for m in members]
        out.append({
            "box": [round(statistics.median([b[k] for b in boxes]), 6) for k in range(4)],
            "seats": sorted({items[m][0] for m in members}),
            "n": len(members),
        })
    out.sort(key=lambda c: -len(c["seats"]))
    return out


ALL_SEATS = (CACHE_SEAT, *SEATS_FREE, "qwen-flash", "doubao-vl", "v6cache_new")


def _discover_seats(out_root: Path) -> list[str]:
    """在场席位=prelabels 下有 labels.jsonl 的已知席位（补票席位按需加入聚类）。"""
    return [s for s in ALL_SEATS if (out_root / "prelabels" / s / "labels.jsonl").is_file()]


def load_windows_file(windows_file: Path) -> dict[int, dict]:
    """外部窗表 {src: {x0,y0,w,h}}（用户手画 ROI 派生正本，b1' 起）。"""
    raw = orjson.loads(windows_file.read_bytes())
    wins = {int(k): {"x0": int(v["x0"]), "y0": int(v.get("y0", 0)),
                     "w": int(v.get("w", 640)), "h": int(v.get("h", 640)),
                     "source": str(windows_file)}
            for k, v in raw.items()}
    missing = [s for s in SOURCES if s not in wins]
    if missing:
        raise typer.BadParameter(f"窗表缺源 {missing}")
    return wins


def in_window(box: Sequence[float], win: dict, ioa_th: float = 0.5) -> bool:
    """全帧 0-1 框与窗（1920×1080 像素制）IoA≥阈值（b1' 域内判定口径）。"""
    x1, y1, x2, y2 = box[0] * 1920, box[1] * 1080, box[2] * 1920, box[3] * 1080
    inter = (max(0.0, min(x2, win["x0"] + win["w"]) - max(x1, win["x0"]))
             * max(0.0, min(y2, win["y0"] + win["h"]) - max(y1, win["y0"])))
    area = max(1e-9, (x2 - x1) * (y2 - y1))
    return inter / area >= ioa_th


@app.command()
def aggregate(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    iou_th: float = typer.Option(CLUSTER_IOU, "--iou"),
    windows_file: Path = typer.Option(None, "--windows-file", help="b1'：外部窗表，票框按 IoA≥0.5 域内过滤后再聚类"),
    cache_seat: str = typer.Option(CACHE_SEAT, "--cache-seat", help="检测器票席位（fp 判定键；b1'=v6cache_new）"),
) -> None:
    """多源聚类分拣 → consensus.jsonl（决策审计正本）+ quarantine.tsv + stats。

    训练包（裁切域 YOLO 形）由 package 子命令独立产出——投票/审计域=960×540
    全帧（用户裁定：提帧分辨率只服务 VLM 投票），入训形态=crop640 裁切域
    （j-consensus-labeling ⑭ 样本形态，2026-11-20 用户钉死）。
    空场多数覆写：≥3 票到场且唯检测器独见框 → confirmed_empty+cache_fp
    （检测器 FP 判定，v8 负样本素材+隔离台账联动，2026-11-20 夜役定规）。
    """
    sel = {r["stem"]: r for r in
           (orjson.loads(line) for line in (out_root / "selection.jsonl").read_text().splitlines() if line.strip())}
    seats = _discover_seats(out_root)
    wins = load_windows_file(windows_file) if windows_file is not None else None
    votes_by_stem: dict[str, dict[str, list]] = {s: {} for s in sel}
    stale = 0
    for seat in seats:
        for line in (out_root / "prelabels" / seat / "labels.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = orjson.loads(line)
            if rec["stem"] not in sel:
                stale += 1  # 烟雾/旧配额批遗留票（选帧确定性近乎前缀但非严格子集）
                continue
            if rec["ok"]:
                boxes = rec["boxes"]
                if wins is not None:
                    win = wins[sel[rec["stem"]]["src"]]
                    boxes = [b for b in boxes if in_window(b, win)]
                votes_by_stem[rec["stem"]][seat] = boxes
    if stale:
        typer.echo(f"跳过非本批选帧的遗留票 {stale} 条", err=True)
    consensus, quarantined, confirmed, empties, fp_stems = [], [], [], [], []
    for stem, r in sel.items():
        votes = votes_by_stem[stem]
        n_ok = len(votes)
        n_empty = sum(1 for b in votes.values() if not b)
        clusters = clusters(votes, iou_th)
        contested = [c for c in clusters if len(c["seats"]) < 2]
        keep = [c for c in clusters if len(c["seats"]) >= 2]
        cache_fp = n_ok >= 3 and n_empty == n_ok - 1 and bool(votes.get(cache_seat))
        decision = ("quarantine" if n_ok < 2 or contested
                    else "confirmed" if keep else "confirmed_empty")
        if cache_fp:
            decision, keep = "confirmed_empty", []  # 空场多数：缓存独见=FP
            fp_stems.append(stem)
        consensus.append({"stem": stem, "src": r["src"], "wallclock": r["wallclock"],
                          "decision": decision, "cache_fp": cache_fp, "n_ok_votes": n_ok,
                          "n_empty_votes": n_empty, "seats": seats,
                          "clusters": clusters, "confs": sel[stem]["confs"]})
        if decision == "quarantine":
            quarantined.append((stem, len(contested) + (max(0, 2 - n_ok)), votes))
        elif decision == "confirmed":
            confirmed.append(stem)
        else:
            empties.append(stem)
    dump(out_root / "consensus.jsonl", consensus)
    quarantined.sort(key=lambda q: (-q[1], q[0]))
    lines = ["stem\tdisagreement\t" + "\t".join(f"n_{s}" for s in seats)]
    lines += [f"{s}\t{d}\t" + "\t".join(str(len(v.get(x, []))) for x in seats)
              for s, d, v in quarantined]
    (out_root / "quarantine.tsv").write_text("\n".join(lines) + "\n")
    n_boxes = sum(len([c for c in rec["clusters"] if len(c["seats"]) >= 2])
                  for rec in consensus if rec["decision"] == "confirmed")
    stats = {
        "selected": len(sel), "seats": seats,
        "confirmed_person": len(confirmed), "confirmed_empty": len(empties),
        "cache_fp_empty": len(fp_stems), "quarantine": len(quarantined),
        "confirmed_boxes": n_boxes,
        "by_src": {str(src): {
            "confirmed": sum(1 for s in confirmed if sel[s]["src"] == src),
            "empty": sum(1 for s in empties if sel[s]["src"] == src),
            "quarantine": sum(1 for s, _, _ in quarantined if sel[s]["src"] == src),
        } for src in SOURCES},
    }
    (out_root / "stats.json").write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    typer.secho(
        f"确认有人 {len(confirmed)}（{n_boxes} 框）/ 确认空帧 {len(empties)}"
        f"（其中缓存 FP {len(fp_stems)}）/ 难例 {len(quarantined)}；席位 {seats}"
        f" → consensus.jsonl + quarantine.tsv（训练包另走 package）",
        fg=typer.colors.GREEN,
    )


# ---------------- tiebreak（定向付费补票：在场分歧两类高值帧） ----------------

# ---------------- recrop / revote-v6（b1'：用户新窗几何） ----------------

@app.command()
def recrop(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    windows_file: Path = typer.Option(..., "--windows-file", help="用户手画 ROI 派生窗表（正本）"),
    concurrency: int = typer.Option(8, "--concurrency"),
) -> None:
    """全部选帧按新窗重提裁切帧（b1' 几何转向；幂等但换窗表须先清目录）。"""
    wins = load_windows_file(windows_file)
    sel = [orjson.loads(line) for line in (out_root / "selection.jsonl").read_text().splitlines() if line.strip()]
    crops = out_root / "crops_b1prime"
    crops.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def _one(r: dict) -> str:
        async with sem:
            return await extract_crop(r["stem"], Path(r["mkv"]), r["pts_us"], wins[r["src"]], crops)

    async def _all() -> list[str]:
        return list(await asyncio.gather(*[_one(r) for r in sel]))

    results = asyncio.run(_all())
    typer.secho(f"重提 {sum(1 for r in results if r)}/{len(sel)} → {crops}", fg=typer.colors.GREEN)


@app.command()
def revote_v6(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    windows_file: Path = typer.Option(..., "--windows-file", help="与 recrop 同窗表（框映射回全帧用）"),
    weights: Path = typer.Option(
        Path("~/ws/sgcc-mma/training/person/datasets/sgcc-n001/crop640_persons/runs/"
             "person_n001_v6_neg0906_s1/weights/best.pt").expanduser(), "--weights"),
    conf: float = typer.Option(0.5, "--conf", help="生产口径 @0.5"),
    batch: int = typer.Option(64, "--batch"),
) -> None:
    """v6 在新窗裁切帧上重推理（b1' 第三票重组；旧缓存票域内无效仅留档）。

    框=crop 像素 → 全帧 0-1（+(x0,y0) 反变换，crop640.rs 契约同源）；conf 同落。
    """
    from ultralytics import YOLO  # 重依赖仅本子命令引入

    wins = load_windows_file(windows_file)
    sel = [orjson.loads(line) for line in (out_root / "selection.jsonl").read_text().splitlines() if line.strip()]
    crops = out_root / "crops_b1prime"
    seat_dir = out_root / "prelabels" / "v6cache_new"
    seat_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    rows_out: list[dict] = []
    n_boxes = 0
    for i in range(0, len(sel), batch):
        chunk = sel[i:i + batch]
        results = model.predict([str(crops / f"{r['stem']}.jpg") for r in chunk],
                                conf=conf, verbose=False)
        for r, res in zip(chunk, results):
            win = wins[r["src"]]
            boxes, confs = [], []
            for b in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                boxes.append([round((x1 + win["x0"]) / 1920, 6), round((y1 + win["y0"]) / 1080, 6),
                              round((x2 + win["x0"]) / 1920, 6), round((y2 + win["y0"]) / 1080, 6)])
                confs.append(round(float(b.conf[0]), 6))
            n_boxes += len(boxes)
            rows_out.append({"stem": r["stem"], "ok": True, "ms": 0, "boxes": boxes,
                             "error": "", "raw": "", "confs": confs})
        typer.echo(f"v6cache_new: {min(i + batch, len(sel))}/{len(sel)}（累计框 {n_boxes}）", err=True)
    dump(seat_dir / "labels.jsonl", rows_out)
    typer.secho(f"v6 新域重推理完成：{len(rows_out)} 帧 / {n_boxes} 框 → {seat_dir}",
                fg=typer.colors.GREEN)


@app.command()
def tiebreak(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    seat: str = typer.Option("qwen-flash", "--seat", help="补票席位（默认按量 qwen VL）"),
    model: str = typer.Option("", "--model", help="覆写席位模型 id（如 qwen3-vl-plus；dated 快照退役时用）"),
    concurrency: int = typer.Option(4, "--concurrency"),
    limit: int = typer.Option(0, "--limit", help=">0 只投前 N 帧（冒烟）"),
) -> None:
    """定向补票：A 类（缓存空+双 VLM 见人=v6 盲区）/ B 类（缓存独见+双 VLM 空=FP 嫌疑）。

    结构混战类（三源框碎片互不匹配）不补——付费票同样碎片化，人裁是正解。
    席位按量计费：token-plan key 在 compatible-mode 拒收（2026-11-20 实测 invalid_api_key，
    KB 记载形态）→ 按用户偏好序落 ALI_DASHSCOPE_API_KEY；dated 模型快照退役 → --model 覆写。
    """
    if model:
        CANDIDATES[seat]["model"] = model  # 运行时覆写，正典 vlm_pool 不动；provenance 落 MODEL.txt
        (out_root / "prelabels" / seat).mkdir(parents=True, exist_ok=True)
        (out_root / "prelabels" / seat / "MODEL.txt").write_text(
            f"{model}\n# 2026-11-20 夜役覆写：原 {seat} 席 dated 快照 404 退役\n")
    cons = [orjson.loads(l) for l in (out_root / "consensus.jsonl").read_text().splitlines() if l.strip()]
    votes: dict[str, dict[str, list]] = {r["stem"]: {} for r in cons}
    for s in _discover_seats(out_root):
        for line in (out_root / "prelabels" / s / "labels.jsonl").read_text().splitlines():
            if line.strip():
                rec = orjson.loads(line)
                if rec["ok"] and rec["stem"] in votes:
                    votes[rec["stem"]][s] = rec["boxes"]
    targets = []
    for r in cons:
        if r["decision"] != "quarantine":
            continue
        v = votes.get(r["stem"], {})
        cache_n = len(v.get(CACHE_SEAT, []))
        q_n = len(v.get("qwen38-local", []))
        g_n = len(v.get("glm-flash", []))
        if (cache_n == 0 and q_n > 0 and g_n > 0) or (cache_n > 0 and q_n == 0 and g_n == 0):
            targets.append(r["stem"])
    if limit:
        targets = targets[:limit]
    rows = [{"stem": s} for s in targets]
    typer.echo(f"补票目标 {len(targets)} 帧（A 缓存盲区 + B 缓存 FP 嫌疑）→ 席位 {seat}")
    ok, err = vote_seat(seat, rows, out_root, concurrency)
    typer.echo(f"{seat}: ok {ok} 弃权 {err}")


# ---------------- package（裁切域训练包；⑭ 样本形态 2026-11-20 用户钉死） ----------------

def derive_windows(video_root: Path) -> dict[int, dict]:
    """缓存框全量反推各源 TrainDomain 检测窗（硬约束：全框⊆窗+(x0,y0)）。

    x0 = (x1_min + x2_max − 640)/2——缓存 bbox 带 ~10-19px 系统性外溢
    （y2_max 640.6-660.4、跨度 640.7-658.6，2026-11-20 六源实测一致），
    两端逼近取中点，不确定度 ±9px。供 iap-4c VLM 中心对拍 + 晨门人裁。
    """
    out: dict[int, dict] = {}
    for src in SOURCES:
        min_x1, max_x2, max_y2, n = math.inf, -math.inf, -math.inf, 0
        for jf in (video_root / str(src) / "0").glob("*/*.json"):
            d = orjson.loads(jf.read_bytes())
            for fr in d["frames"]:
                for b in fr["detections"]:
                    x1, _, x2, y2 = b["bbox"]
                    n += 1
                    min_x1 = min(min_x1, x1)
                    max_x2 = max(max_x2, x2)
                    max_y2 = max(max_y2, y2)
        x0 = max(0, min(1920 - 640, round((min_x1 + max_x2 - 640.0) / 2)))
        out[src] = {"x0": x0, "y0": 0, "w": 640, "h": 640, "cx": x0 + 320,
                    "min_x1": round(min_x1, 1), "max_x2": round(max_x2, 1),
                    "max_y2": round(max_y2, 1), "n_boxes": n}
    return out


def box_to_crop(
    box_norm: Sequence[float], win: dict, ioa_th: float = 0.5
) -> tuple[float, float, float, float] | None:
    """全帧 0-1 xyxy → 裁切域像素（win={x0,y0,w,h}）；IoA≥ioa_th 保留并裁边，否则弃。"""
    x1 = box_norm[0] * 1920 - win["x0"]
    y1 = box_norm[1] * 1080 - win["y0"]
    x2 = box_norm[2] * 1920 - win["x0"]
    y2 = box_norm[3] * 1080 - win["y0"]
    w, h = float(win["w"]), float(win["h"])
    inter = max(0.0, min(x2, w) - max(x1, 0.0)) * max(0.0, min(y2, h) - max(y1, 0.0))
    area = max(1e-9, (x2 - x1) * (y2 - y1))
    if inter / area < ioa_th:
        return None
    return max(0.0, x1), max(0.0, y1), min(w, x2), min(h, y2)


async def extract_crop(stem: str, mkv: Path, pts_us: int, win: dict, out_dir: Path) -> str:
    """源 MKV 全分辨率重提裁切帧（win={x0,y0,w,h}，TrainDomain 或 ROI 派生窗）。"""
    out = out_dir / f"{stem}.jpg"
    if out.exists() and out.stat().st_size > 4_096:
        return stem
    vf = f"crop={win['w']}:{win['h']}:{win['x0']}:{win['y0']}"
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{pts_us / 1e6:.6f}", "-i", str(mkv),
        "-frames:v", "1", "-vf", vf,
        "-q:v", "3", str(out),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not out.exists() or out.stat().st_size <= 4_096:
        out.unlink(missing_ok=True)
        typer.secho(f"crop extract fail {stem}: {err.decode()[:120]}", fg=typer.colors.YELLOW, err=True)
        return ""
    return stem


@app.command()
def package(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    video_root: Path = typer.Option(VIDEO_ROOT, "--video-root"),
    windows_file: Path = typer.Option(None, "--windows-file", help="外部窗表 {src: {x0,y0,w,h}}（用户手画 ROI 派生，b1' 起）；缺省=缓存反推"),
    concurrency: int = typer.Option(8, "--concurrency"),
) -> None:
    """确认子集 → crop640 裁切域 YOLO 训练包（窗 provenance → windows.json）。

    样本形态铁律（j-consensus-labeling ⑭）：入训一律裁切域（与检测 TrainDomain
    同口径），整帧/半幅禁止直接入训；灰化 b2 随正式 ROI 定（n005 无逐源依据不盲灰）。
    窗来源二选一：--windows-file（用户手画 ROI 派生表，正本）或缓存反推（b1 暂定形）。
    投票/聚合均为全帧坐标——几何无关，换窗重打包零重跑（b1' 复用原理）。
    """
    if windows_file is not None:
        raw = orjson.loads(windows_file.read_bytes())
        windows = {int(k): {"x0": int(v["x0"]), "cx": int(v["x0"]) + int(v.get("w", 640)) // 2,
                            "y0": int(v.get("y0", 0)), "w": int(v.get("w", 640)),
                            "h": int(v.get("h", 640)), "source": str(windows_file)}
                   for k, v in raw.items()}
        missing = [s for s in SOURCES if s not in windows]
        if missing:
            raise typer.BadParameter(f"窗表缺源 {missing}")
    else:
        windows = derive_windows(video_root)
    (out_root / "windows.json").write_bytes(
        orjson.dumps({str(k): v for k, v in windows.items()}, option=orjson.OPT_INDENT_2)
    )
    cons = {r["stem"]: r for r in (orjson.loads(line) for line in
           (out_root / "consensus.jsonl").read_text().splitlines() if line.strip())}
    sel = {r["stem"]: r for r in (orjson.loads(line) for line in
           (out_root / "selection.jsonl").read_text().splitlines() if line.strip())}
    take = [s for s, r in cons.items() if r["decision"] in ("confirmed", "confirmed_empty")]
    img_dir = out_root / "confirmed" / "images"
    lbl_dir = out_root / "confirmed" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "confirmed" / "classes.txt").write_text("person\n")
    sem = asyncio.Semaphore(concurrency)

    async def _one(stem: str) -> str:
        async with sem:
            win = windows[sel[stem]["src"]]
            return await extract_crop(stem, Path(sel[stem]["mkv"]), sel[stem]["pts_us"], win, img_dir)

    async def _all() -> list[str]:
        return list(await asyncio.gather(*[_one(s) for s in take]))

    results = asyncio.run(_all())
    ok_stems = {s for s, r in zip(take, results) if r}
    manifest, n_boxes, n_drop = [], 0, 0
    for stem in take:
        if stem not in ok_stems:
            continue
        win = windows[cons[stem]["src"]]
        rec, lines = cons[stem], []
        if rec["decision"] == "confirmed":
            for cl in rec["clusters"]:
                if len(cl["seats"]) < 2:
                    continue
                b = box_to_crop(cl["box"], win)
                if b is None:
                    n_drop += 1
                    continue
                x1, y1, x2, y2 = b
                fw, fh = float(win["w"]), float(win["h"])
                lines.append(f"0 {(x1 + x2) / (2 * fw):.6f} {(y1 + y2) / (2 * fh):.6f} "
                             f"{(x2 - x1) / fw:.6f} {(y2 - y1) / fh:.6f}")
                n_boxes += 1
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        manifest.append({"stem": stem, "src": rec["src"], "decision": rec["decision"],
                         "window": {k: win[k] for k in ("x0", "y0", "w", "h")},
                         "n_label_boxes": len(lines)})
    dump(out_root / "confirmed" / "manifest.jsonl", manifest)
    typer.secho(
        f"裁切域训练包：{len(ok_stems)} 帧（框 {n_boxes}，IoA 弃 {n_drop}，提取失败 "
        f"{len(take) - len(ok_stems)}）→ {out_root / 'confirmed'}",
        fg=typer.colors.GREEN,
    )


# ---------------- export-verdicts（M1：晨裁回流并入训练包） ----------------

@app.command()
def export_verdicts(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    set_slug: str = typer.Option("n005_b1prime_hard_v1", "--set-slug"),
    user: str = typer.Option("jiang", "--user", help="人裁 annotator 名"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """ulabel 人裁层（Label 文档）→ YOLO 标签并入 b1' 训练包 train 域（v8 计划 M1）。

    读 /var/ulabel 磁盘正本（只读；服务是唯一写入者）；polygon=窗域归一四点
    （播种与人工编辑同在裁切域图上）→ bbox 外接→xywh 直转，零窗映射。
    图源=crops_b1prime（已按 v4 窗提好）拷入 dataset/train；已裁决帧从确认包
    train 侧移除（决策升级：模型票→人裁真值）。空 Label=有效负样本。
    """
    vdir = Path("/var/ulabel/sgcc_av/detection") / set_slug / "verdicts" / user
    if not vdir.is_dir():
        raise typer.BadParameter(f"人裁目录不存在（尚未开裁？）: {vdir}")
    files = sorted(vdir.glob("*.json"))
    if not files:
        raise typer.BadParameter("人裁层为空——等用户裁决后再跑")
    ds = out_root / "dataset"
    n_box = n_empty = 0
    plan: list[tuple[str, list[str]]] = []
    for f in files:
        stem = f.stem
        doc = orjson.loads(f.read_bytes())
        lines = []
        for obj in doc.get("objects", []):
            if int(obj.get("category", 1)) != 1:
                raise typer.BadParameter(f"{stem}: 非 person 类目 {obj['category']}（本批应单类）")
            xs = [p["x"] for p in obj["polygon"]]
            ys = [p["y"] for p in obj["polygon"]]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        if lines:
            n_box += 1
        else:
            n_empty += 1
        plan.append((stem, lines))
    typer.echo(f"人裁 {len(plan)} 帧：有框 {n_box} / 空标（负样本）{n_empty}，"
               f"框数 {sum(len(l) for _, l in plan)}")
    if dry_run:
        return
    for stem, lines in plan:
        src_crop = out_root / "crops_b1prime" / f"{stem}.jpg"
        if not src_crop.is_file():
            raise typer.BadParameter(f"裁切帧缺失: {src_crop}（先跑 recrop）")
        shutil.copy2(src_crop, ds / "train" / "images" / f"{stem}.jpg")
        (ds / "train" / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        # 确认包旧副本让位（若该帧曾以自动票入 train——人裁为真值正本）
        for split in ("train", "val"):
            for ext_dir, ext in ((ds / split / "images", ".jpg"), (ds / split / "labels", ".txt")):
                old = ext_dir / f"{stem}{ext}"
                if old.exists():
                    old.unlink()
    (out_root / f"verdicts_export_{set_slug}.json").write_bytes(orjson.dumps(
        {"user": user, "frames": len(plan), "with_box": n_box, "empty": n_empty},
        option=orjson.OPT_INDENT_2))
    typer.secho(f"回流完成 → {ds}/train（原自动票副本已让位；val 侧同步剔除防泄漏）",
                fg=typer.colors.GREEN)


# ---------------- seed-ulabel（难例集播种；API 形制对齐 ulabel build_detection_gt.py） ----------------

ULABEL_BASE = "http://127.0.0.1:8791"
ULABEL_PROJECT, ULABEL_GROUP = "sgcc_av", "detection"
SET_SLUG = "n005_b1_hard_v1"
SET_META_ID = 9  # sgcc_av 序列：sessions_v1=7 / detection_gt_min_v1=8
SEAT_TO_ANNOTATOR = {
    CACHE_SEAT: "yolo26n-v6",  # 缓存即现役 person_n_v6 输出（指纹 2026-09-18_person_n_v6@0.5）
    "v6cache_new": "yolo26n-v6",  # b1'：v6 新窗域重推理（同模型，同 annotator 层覆写）
    "qwen38-local": "qwen3-8-flash-next-local",
    "glm-flash": "glm-5-3-flash",  # 需 users.json5 注册行（2026-11-20 夜加）+ 服务重启
    "qwen-flash": "qwen3-vl-flash",  # 定向补票席位（2026-11-20 夜加注册）
}
DRAFT = "draft"
PERSON_CAT, PROP_CONF, PROP_VOTES, PROP_STATUS, RAW_VALUE = 1, 1, 2, 3, 1
LABEL_VERSION = "2.0"


class UlabelApiError(Exception):
    """ulabel HTTP 契约失败（点名端点与响应体）。"""


class UlabelClient:
    """ulabel-service 薄客户端（形制对齐 ulabel scripts/build_detection_gt.py；httpx 同步）。"""

    def __init__(self, base_url: str, set_slug: str) -> None:
        self.base = base_url.rstrip("/")
        self.set_slug = set_slug
        self.tokens: dict[str, str] = {}

    def _send(
        self, method: str, path: str, user: str, *, json_body: dict | None = None,
        data: bytes | None = None, params: dict[str, str] | None = None,
    ) -> httpx.Response:
        token = self.tokens.get(user)
        if token is None:
            raise UlabelApiError(f"用户 {user!r} 未登录")
        r = httpx.request(
            method, self.base + path, json=json_body, data=data, params=params,
            headers={"Authorization": f"Bearer {token}"}, timeout=120.0,
        )
        if r.status_code >= 500:
            raise UlabelApiError(f"{method} {path} → {r.status_code}: {r.text[:500]}")
        return r

    def login(self, user: str) -> None:
        r = httpx.post(self.base + "/api/login", json={"name": user}, timeout=30.0)
        if r.status_code != 200:
            raise UlabelApiError(f"login {user!r} → {r.status_code}: {r.text[:300]}")
        self.tokens[user] = r.json()["token"]

    def list_annotators(self, user: str) -> list[dict]:
        r = self._send("GET", "/api/annotators", user)
        if r.status_code != 200:
            raise UlabelApiError(f"list_annotators → {r.status_code}: {r.text[:300]}")
        return r.json()

    def ensure_set(self, meta: dict, user: str) -> None:
        r = self._send(
            "POST", f"/api/projects/{ULABEL_PROJECT}/groups/{ULABEL_GROUP}/sets", user,
            json_body={"slug": self.set_slug, "kind": "annotation", "media": "image", "meta": meta},
        )
        if r.status_code == 201:
            return
        if r.status_code == 409:
            return  # 幂等复用
        raise UlabelApiError(f"ensure_set → {r.status_code}: {r.text[:300]}")

    def post_sample(self, filename: str, payload: bytes, user: str) -> None:
        r = self._send(
            "POST", f"/api/projects/{ULABEL_PROJECT}/groups/{ULABEL_GROUP}/sets/{self.set_slug}/samples",
            user, params={"stem": filename}, data=payload,
        )
        if r.status_code not in (201, 409):
            raise UlabelApiError(f"post_sample {filename} → {r.status_code}: {r.text[:300]}")

    def post_annotation(self, stem: str, label: dict, user: str) -> None:
        r = self._send(
            "POST", f"/api/projects/{ULABEL_PROJECT}/groups/{ULABEL_GROUP}/sets/{self.set_slug}/verdicts/{stem}",
            user, json_body=label,
        )
        if r.status_code != 200:
            raise UlabelApiError(f"post_annotation {user}/{stem} → {r.status_code}: {r.text[:300]}")

    def list_samples(self, user: str) -> list[dict]:
        r = self._send(
            "GET", f"/api/projects/{ULABEL_PROJECT}/groups/{ULABEL_GROUP}/sets/{self.set_slug}/samples", user
        )
        if r.status_code != 200:
            raise UlabelApiError(f"list_samples → {r.status_code}: {r.text[:300]}")
        return r.json()


def set_meta(meta_id: int, set_slug: str) -> dict:
    """集 meta：克隆 detection_gt_min_v1 形制（person+conf/votes/status 三属性）。"""
    return {
        "id": meta_id,
        "name": f"n005 {set_slug} 难例裁决（夜役三源自动标注）",
        "description": (
            f"{set_slug}：809 帧多源共识（v6 新窗域重推理+qwen38-local+glm-flash+定向补票）"
            "分拣出的分歧帧。模型票层+draft 聚簇层（votes/status 属性）；人只在 draft 上裁决。"
            "制备 jxl.bin.n005_detect_b1 seed-ulabel；清单导航 tsv=工作区 quarantine.tsv"
        ),
        "shape": {"title_style": 0, "thickness": 2, "auto_save": True, "vertex_radius": 10.0},
        "roi": {"color": "#800080"},
        "categories": [{
            "id": PERSON_CAT, "names": {"en": "person", "zh": "人"},
            "description": "检测目标（本批单类）", "hotkey": "1", "color": "#FF6D00",
            "properties": [
                {"id": PROP_CONF, "names": {"en": "conf", "zh": "置信度"}, "type": "conf"},
                {"id": PROP_VOTES, "names": {"en": "votes", "zh": "票数"}, "type": "votes"},
                {"id": PROP_STATUS, "names": {"en": "status", "zh": "共识状态"}, "type": "status"},
            ],
            "rect_constraint": True,
        }],
        "property_types": [
            {"id": PROP_CONF, "names": {"en": "conf", "zh": "置信度"},
             "description": "缓存票 conf：confidence=原浮点、note=原数字符串（过渡编码同 detection_gt_min_v1）",
             "values": [{"id": RAW_VALUE, "names": {"en": "raw", "zh": "原值"},
                         "description": "conf 原值见 confidence/note", "hotkey": "", "color": "#9E9E9E", "sign": ""}]},
            {"id": PROP_VOTES, "names": {"en": "votes", "zh": "票数"},
             "description": "draft 框支持源数：confidence=票数、note=数字串；仅 draft 层",
             "values": [{"id": RAW_VALUE, "names": {"en": "raw", "zh": "原值"},
                         "description": "票数见 confidence/note", "hotkey": "", "color": "#9E9E9E", "sign": ""}]},
            {"id": PROP_STATUS, "names": {"en": "status", "zh": "共识状态"},
             "description": "draft 框支持源明细（triple/pair/solo+席位串）：note=原串、confidence 恒 1.0；仅 draft 层",
             "values": [{"id": RAW_VALUE, "names": {"en": "raw", "zh": "原值"},
                         "description": "状态见 note", "hotkey": "", "color": "#9E9E9E", "sign": ""}]},
        ],
        "property_special_values": [],
    }


def label_doc(objects: list[dict]) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {"version": LABEL_VERSION, "user_agent": "n005_detect_b1",
            "created_at": now, "last_modified": now, "objects": objects}


def rect_polygon(box_norm: Sequence[float]) -> list[dict]:
    """0-1 归一 xyxy → 归一四点多边形（六位小数，顺时针；经 960×540 像素制）。

    钳制到 [0,1]：VLM 0-1000 刻度偶发越值（>1000）归一后 >1，服务端校验拒绝
    coordinate_out_of_range（2026-11-20 夜役实测）；退化框（x2<=x1 等）压零面积。
    """
    x1, y1, x2, y2 = [v * FRAME_W if i % 2 == 0 else v * FRAME_H for i, v in enumerate(box_norm)]
    x1, x2 = sorted((min(max(x1 / FRAME_W, 0.0), 1.0), min(max(x2 / FRAME_W, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1 / FRAME_H, 0.0), 1.0), min(max(y2 / FRAME_H, 0.0), 1.0)))
    return [
        {"x": round(x, 6), "y": round(y, 6)}
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    ]


@app.command()
def seed_ulabel(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    base_url: str = typer.Option(ULABEL_BASE, "--base-url"),
    set_slug: str = typer.Option(SET_SLUG, "--set-slug", help="b1' 用新集 slug（旧集留档勿裁）"),
    meta_id: int = typer.Option(SET_META_ID, "--meta-id", help="集 meta id（sgcc_av 序列递增：7/8/9 已用）"),
    admin: str = typer.Option("jiang", "--admin", help="组织/上传身份"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """难例帧播种 ulabel 集（模型层+draft 层；幂等：样本 409 跳过、verdict 覆写）。"""
    cons = {orjson.loads(l)["stem"]: orjson.loads(l)
            for l in (out_root / "consensus.jsonl").read_text().splitlines() if l.strip()}
    sel = {orjson.loads(l)["stem"]: orjson.loads(l)
           for l in (out_root / "selection.jsonl").read_text().splitlines() if l.strip()}
    tsv = (out_root / "quarantine.tsv").read_text().splitlines()[1:]
    hard = [line.split("\t")[0] for line in tsv if line.strip()]
    votes: dict[str, dict[str, dict]] = {s: {} for s in hard}
    for seat in _discover_seats(out_root):
        for l in (out_root / "prelabels" / seat / "labels.jsonl").read_text().splitlines():
            if not l.strip():
                continue
            rec = orjson.loads(l)
            if rec["stem"] in votes and rec["ok"]:
                votes[rec["stem"]][seat] = rec
    typer.echo(f"难例 {len(hard)} 帧；meta id={SET_META_ID}；slug={set_slug}")
    if dry_run:
        n_boxes = {a: sum(len(votes[s].get(seat, {}).get("boxes", [])) for s in hard)
                   for seat, a in SEAT_TO_ANNOTATOR.items()}
        n_boxes[DRAFT] = sum(len(cons[s]["clusters"]) for s in hard)
        typer.echo(f"dry-run 层框数: {n_boxes}")
        return
    client = UlabelClient(base_url, set_slug)
    client.login(admin)
    registered = {a["name"] for a in client.list_annotators(admin)}
    want = [SEAT_TO_ANNOTATOR[s] for s in _discover_seats(out_root) if s in SEAT_TO_ANNOTATOR] + [DRAFT]
    missing = [a for a in want if a not in registered]
    if missing:
        raise typer.BadParameter(
            f"注册缺口 {missing}——/var/ulabel/users.json5 加行+重启服务后重跑（脚本打印见 README）"
        )
    client.ensure_set(set_meta(meta_id, set_slug), admin)
    for seat_annotator in want:
        client.login(seat_annotator)  # 种子署名=登录身份
    n_frames = n_layers = 0
    for stem in hard:
        client.post_sample(f"{stem}.jpg", (out_root / "frames" / f"{stem}.jpg").read_bytes(), admin)
        for seat, annotator in SEAT_TO_ANNOTATOR.items():
            rec = votes[stem].get(seat)
            if rec is None:
                continue  # 弃权=缺席文件（未标注语义，非零检出）
            confs = sel[stem]["confs"] if seat == CACHE_SEAT else None
            objects = []
            for i, b in enumerate(rec["boxes"], 1):
                obj = {"id": i, "category": PERSON_CAT, "confidence": 1.0,
                       "polygon": rect_polygon(b), "properties": []}
                if confs is not None and i <= len(confs):
                    obj["properties"] = [{"id": PROP_CONF, "value": RAW_VALUE,
                                          "confidence": float(confs[i - 1]), "note": str(confs[i - 1])}]
                objects.append(obj)
            client.post_annotation(stem, label_doc(objects), annotator)
            n_layers += 1
        draft_objects = []
        for i, cl in enumerate(cons[stem]["clusters"], 1):
            seats = "+".join(cl["seats"])
            status = ("triple" if len(cl["seats"]) == 3 else
                      f"pair:{seats}" if len(cl["seats"]) == 2 else f"solo:{seats}")
            draft_objects.append({
                "id": i, "category": PERSON_CAT, "confidence": 1.0,
                "polygon": rect_polygon(cl["box"]),
                "properties": [
                    {"id": PROP_VOTES, "value": RAW_VALUE, "confidence": float(len(cl["seats"])),
                     "note": str(len(cl["seats"]))},
                    {"id": PROP_STATUS, "value": RAW_VALUE, "confidence": 1.0, "note": status},
                ],
            })
        client.post_annotation(stem, label_doc(draft_objects), DRAFT)
        n_layers += 1
        n_frames += 1
        if n_frames % 20 == 0:
            typer.echo(f"{n_frames}/{len(hard)}", err=True)
    samples = client.list_samples(admin)
    ann = {s["stem"]: s.get("annotations", {}) for s in samples}
    short = [s for s in hard if len(ann.get(s, {})) < 2]
    typer.secho(
        f"播种 {n_frames} 帧 / {n_layers} 层；回读 {len(samples)} 样本，层缺帧 {len(short)} "
        f"→ /var/ulabel/{ULABEL_PROJECT}/{ULABEL_GROUP}/{set_slug}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
