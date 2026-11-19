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
    _dump(out_root / "selection.jsonl", [
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

def _dump(path: Path, rows: Sequence[dict]) -> None:
    path.write_text("".join(orjson.dumps(r).decode() + "\n" for r in rows))


def _vote_seat(
    seat: str, rows: Sequence[dict], out_root: Path, concurrency: int
) -> tuple[int, int]:
    """单席位逐帧投票（断点续跑：已有 stem 跳过）；原始响应逐字落盘（#16）。"""
    pre = out_root / "prelabels" / seat
    pre.mkdir(parents=True, exist_ok=True)
    path = pre / "labels.jsonl"
    done: set[str] = set()
    if path.exists():
        done = {orjson.loads(line)["stem"] for line in path.read_text().splitlines() if line.strip()}
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
    _dump(cache_dir / "labels.jsonl", [
        {"stem": r["stem"], "ok": True, "ms": 0, "boxes": r["boxes_norm"],
         "error": "", "raw": "", "confs": r["confs"]} for r in rows
    ])
    unknown = [s for s in seats.split(",") if s not in CANDIDATES]
    if unknown:
        raise typer.BadParameter(f"未知席位: {unknown}")
    for seat in seats.split(","):
        ok, err = _vote_seat(seat, rows, out_root, concurrency)
        rate = ok / max(1, ok + err)
        typer.echo(f"{seat}: 新投 {ok + err}，ok率 {rate:.1%}（累计含续跑 {len(rows)}）")
        if rate < 0.8 and ok + err > 5:
            typer.secho(f"{seat} 弃权率超 20%——席位异常，停查", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)


# ---------------- aggregate ----------------

def _clusters(
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


@app.command()
def aggregate(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    iou_th: float = typer.Option(CLUSTER_IOU, "--iou"),
) -> None:
    """三源聚类分拣 → confirmed/ YOLO 包 + quarantine.tsv + consensus.jsonl + stats。"""
    sel = {r["stem"]: r for r in
           (orjson.loads(line) for line in (out_root / "selection.jsonl").read_text().splitlines() if line.strip())}
    seats = [CACHE_SEAT, *SEATS_FREE]
    votes_by_stem: dict[str, dict[str, list]] = {s: {} for s in sel}
    for seat in seats:
        for line in (out_root / "prelabels" / seat / "labels.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = orjson.loads(line)
            if rec["ok"]:
                votes_by_stem[rec["stem"]][seat] = rec["boxes"]
    conf_dir = out_root / "confirmed"
    (conf_dir / "images").mkdir(parents=True, exist_ok=True)
    (conf_dir / "labels").mkdir(parents=True, exist_ok=True)
    (conf_dir / "classes.txt").write_text("person\n")
    consensus, quarantined, confirmed, empties = [], [], [], []
    for stem, r in sel.items():
        votes = votes_by_stem[stem]
        n_ok = len(votes)
        clusters = _clusters(votes, iou_th)
        contested = [c for c in clusters if len(c["seats"]) < 2]
        rec = {"stem": stem, "src": r["src"], "wallclock": r["wallclock"],
               "n_ok_votes": n_ok, "clusters": clusters, "confs": sel[stem]["confs"]}
        consensus.append(rec)
        if n_ok < 2 or contested:
            quarantined.append((stem, len(contested) + (2 - n_ok), votes))
            continue
        keep = [c for c in clusters if len(c["seats"]) >= 2]
        if keep:
            confirmed.append(stem)
        else:
            empties.append(stem)
        shutil.copy2(out_root / "frames" / f"{stem}.jpg", conf_dir / "images" / f"{stem}.jpg")
        label_lines = []
        for c in keep:
            x1, y1, x2, y2 = c["box"]
            label_lines.append(
                f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
            )
        (conf_dir / "labels" / f"{stem}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
    _dump(out_root / "consensus.jsonl", consensus)
    quarantined.sort(key=lambda q: (-q[1], q[0]))
    lines = ["stem\tdisagreement\tn_v6cache\tn_qwen38\tn_glm"]
    lines += [f"{s}\t{d}\t{len(v.get(CACHE_SEAT, []))}\t"
              f"{len(v.get('qwen38-local', []))}\t{len(v.get('glm-flash', []))}"
              for s, d, v in quarantined]
    (out_root / "quarantine.tsv").write_text("\n".join(lines) + "\n")
    n_boxes = sum(len((conf_dir / "labels" / f"{s}.txt").read_text().splitlines()) for s in confirmed)
    stats = {
        "selected": len(sel), "confirmed_person": len(confirmed), "confirmed_empty": len(empties),
        "quarantine": len(quarantined), "confirmed_boxes": n_boxes,
        "by_src": {str(src): {
            "confirmed": sum(1 for s in confirmed if sel[s]["src"] == src),
            "empty": sum(1 for s in empties if sel[s]["src"] == src),
            "quarantine": sum(1 for s, _, _ in quarantined if sel[s]["src"] == src),
        } for src in SOURCES},
    }
    (out_root / "stats.json").write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    typer.secho(
        f"确认有人 {len(confirmed)}（{n_boxes} 框）/ 确认空帧 {len(empties)} / 难例 {len(quarantined)}"
        f" → {conf_dir} + quarantine.tsv",
        fg=typer.colors.GREEN,
    )


# ---------------- seed-ulabel（难例集播种；API 形制对齐 ulabel build_detection_gt.py） ----------------

ULABEL_BASE = "http://127.0.0.1:8791"
ULABEL_PROJECT, ULABEL_GROUP = "sgcc_av", "detection"
SET_SLUG = "n005_b1_hard_v1"
SET_META_ID = 9  # sgcc_av 序列：sessions_v1=7 / detection_gt_min_v1=8
SEAT_TO_ANNOTATOR = {
    CACHE_SEAT: "yolo26n-v6",  # 缓存即现役 person_n_v6 输出（指纹 2026-09-18_person_n_v6@0.5）
    "qwen38-local": "qwen3-8-flash-next-local",
    "glm-flash": "glm-5-3-flash",  # 需 users.json5 注册行（2026-11-20 夜加）+ 服务重启
}
DRAFT = "draft"
PERSON_CAT, PROP_CONF, PROP_VOTES, PROP_STATUS, RAW_VALUE = 1, 1, 2, 3, 1
LABEL_VERSION = "2.0"


class UlabelApiError(Exception):
    """ulabel HTTP 契约失败（点名端点与响应体）。"""


class UlabelClient:
    """ulabel-service 薄客户端（形制对齐 ulabel scripts/build_detection_gt.py；httpx 同步）。"""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
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
            json_body={"slug": SET_SLUG, "kind": "annotation", "media": "image", "meta": meta},
        )
        if r.status_code == 201:
            return
        if r.status_code == 409:
            return  # 幂等复用
        raise UlabelApiError(f"ensure_set → {r.status_code}: {r.text[:300]}")

    def post_sample(self, filename: str, payload: bytes, user: str) -> None:
        r = self._send(
            "POST", f"/api/projects/{ULABEL_PROJECT}/groups/{ULABEL_GROUP}/sets/{SET_SLUG}/samples",
            user, params={"stem": filename}, data=payload,
        )
        if r.status_code not in (201, 409):
            raise UlabelApiError(f"post_sample {filename} → {r.status_code}: {r.text[:300]}")

    def post_annotation(self, stem: str, label: dict, user: str) -> None:
        r = self._send(
            "POST", f"/api/projects/{ULABEL_PROJECT}/groups/{ULABEL_GROUP}/sets/{SET_SLUG}/verdicts/{stem}",
            user, json_body=label,
        )
        if r.status_code != 200:
            raise UlabelApiError(f"post_annotation {user}/{stem} → {r.status_code}: {r.text[:300]}")

    def list_samples(self, user: str) -> list[dict]:
        r = self._send(
            "GET", f"/api/projects/{ULABEL_PROJECT}/groups/{ULABEL_GROUP}/sets/{SET_SLUG}/samples", user
        )
        if r.status_code != 200:
            raise UlabelApiError(f"list_samples → {r.status_code}: {r.text[:300]}")
        return r.json()


def _set_meta() -> dict:
    """集 meta：克隆 detection_gt_min_v1 形制（person+conf/votes/status 三属性）。"""
    return {
        "id": SET_META_ID,
        "name": "n005 批次1 难例裁决（夜役三源自动标注）",
        "description": (
            "n005-b1 夜役难例：809 帧三源共识（v6 缓存+qwen38-local+glm-flash）分拣出的分歧帧。"
            "三层模型票+draft 聚簇层（votes/status 属性）；人只在 draft 上裁决。"
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


def _label_doc(objects: list[dict]) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {"version": LABEL_VERSION, "user_agent": "n005_detect_b1",
            "created_at": now, "last_modified": now, "objects": objects}


def _rect_polygon(box_norm: Sequence[float]) -> list[dict]:
    """0-1 归一 xyxy → 归一四点多边形（六位小数，顺时针；经 960×540 像素制）。"""
    x1, y1, x2, y2 = [v * FRAME_W if i % 2 == 0 else v * FRAME_H for i, v in enumerate(box_norm)]
    return [
        {"x": round(x / FRAME_W, 6), "y": round(y / FRAME_H, 6)}
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    ]


@app.command()
def seed_ulabel(
    out_root: Path = typer.Option(OUT_ROOT, "--out-root"),
    base_url: str = typer.Option(ULABEL_BASE, "--base-url"),
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
    for seat in (CACHE_SEAT, *SEATS_FREE):
        for l in (out_root / "prelabels" / seat / "labels.jsonl").read_text().splitlines():
            if not l.strip():
                continue
            rec = orjson.loads(l)
            if rec["stem"] in votes and rec["ok"]:
                votes[rec["stem"]][seat] = rec
    typer.echo(f"难例 {len(hard)} 帧；meta id={SET_META_ID}；slug={SET_SLUG}")
    if dry_run:
        n_boxes = {a: sum(len(votes[s].get(seat, {}).get("boxes", [])) for s in hard)
                   for seat, a in SEAT_TO_ANNOTATOR.items()}
        n_boxes[DRAFT] = sum(len(cons[s]["clusters"]) for s in hard)
        typer.echo(f"dry-run 层框数: {n_boxes}")
        return
    client = UlabelClient(base_url)
    client.login(admin)
    registered = {a["name"] for a in client.list_annotators(admin)}
    want = [SEAT_TO_ANNOTATOR[s] for s in (CACHE_SEAT, *SEATS_FREE)] + [DRAFT]
    missing = [a for a in want if a not in registered]
    if missing:
        raise typer.BadParameter(
            f"注册缺口 {missing}——/var/ulabel/users.json5 加行+重启服务后重跑（脚本打印见 README）"
        )
    client.ensure_set(_set_meta(), admin)
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
                       "polygon": _rect_polygon(b), "properties": []}
                if confs is not None and i <= len(confs):
                    obj["properties"] = [{"id": PROP_CONF, "value": RAW_VALUE,
                                          "confidence": float(confs[i - 1]), "note": str(confs[i - 1])}]
                objects.append(obj)
            client.post_annotation(stem, _label_doc(objects), annotator)
            n_layers += 1
        draft_objects = []
        for i, cl in enumerate(cons[stem]["clusters"], 1):
            seats = "+".join(cl["seats"])
            status = ("triple" if len(cl["seats"]) == 3 else
                      f"pair:{seats}" if len(cl["seats"]) == 2 else f"solo:{seats}")
            draft_objects.append({
                "id": i, "category": PERSON_CAT, "confidence": 1.0,
                "polygon": _rect_polygon(cl["box"]),
                "properties": [
                    {"id": PROP_VOTES, "value": RAW_VALUE, "confidence": float(len(cl["seats"])),
                     "note": str(len(cl["seats"]))},
                    {"id": PROP_STATUS, "value": RAW_VALUE, "confidence": 1.0, "note": status},
                ],
            })
        client.post_annotation(stem, _label_doc(draft_objects), DRAFT)
        n_layers += 1
        n_frames += 1
        if n_frames % 20 == 0:
            typer.echo(f"{n_frames}/{len(hard)}", err=True)
    samples = client.list_samples(admin)
    ann = {s["stem"]: s.get("annotations", {}) for s in samples}
    short = [s for s in hard if len(ann.get(s, {})) < 2]
    typer.secho(
        f"播种 {n_frames} 帧 / {n_layers} 层；回读 {len(samples)} 样本，层缺帧 {len(short)} "
        f"→ /var/ulabel/{ULABEL_PROJECT}/{ULABEL_GROUP}/{SET_SLUG}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
