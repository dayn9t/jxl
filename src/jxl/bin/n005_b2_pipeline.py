#!/usr/bin/env python3
"""n005 b2 裁切域共识标注流水线（SillAnchor crop 直投，2026-11-20 iap-94 对接）。

与 b1 线（n005_detect_b1）的根本差异：b2 输入帧**已是 SillAnchor 窗裁切**
（iap-crop-samples 切制，窗几何与生产 detector 同源）——全帧域/窗映射/recrop
概念全部不存在，投票与标签直接在裁切域进行，零坐标变换。

席位与门（消费 iap docs/2026-11-20-dual-threshold-calibration.md）：
  v6crop（对 crop 图直接推理，窗域框+conf）+ qwen38-local（182 免费强票）
  + glm-flash（套餐仲裁）——全免费；付费补票仅定向分歧帧另跑。
  IoU 聚类门 0.4；v6 conf<0.40（置信不确定门）框禁自动收；
  v6 确认框与簇心 IoU<0.55（一致门）→ 该框降「框漂移」复核池（整帧不废）。

子命令序：v6 → vote → aggregate。产物目录=--out-root（独立于包目录，包只读）。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Sequence

import httpx
import orjson
import typer
from PIL import Image

from jxl.bin.n005_detect_b1 import _clusters, _dump  # 同源聚类纯函数（原则 8）
from jxl.bin.vlm_pool import build_target_prompt, call_person, image_data_url

app = typer.Typer(add_completion=False, help="n005 b2 裁切域共识标注流水线。")

CONF_UNCERTAIN = 0.40  # 置信不确定门（标定报告 v1）
IOU_AGREE = 0.55  # IoU 一致门：v6 框 vs 簇心 <此值=框漂移复核
VOTE_SEATS = ("qwen38-local", "glm-flash")


def _iter_frames(pkg: Path) -> list[tuple[str, Path]]:
    """包内帧清单（stem=相对路径剥扩展，图按 pt 档目录组织）。"""
    frames = [(str(p.relative_to(pkg).with_suffix("")), p)
              for p in sorted(pkg.rglob("*.jpg")) if "labels" not in p.parts]
    if not frames:
        raise typer.BadParameter(f"{pkg} 无 jpg 帧")
    return frames


@app.command()
def v6(
    pkg: Path = typer.Option(..., "--pkg", help="b2 包根（n902 上则本地路径同形）"),
    out_root: Path = typer.Option(..., "--out-root"),
    weights: Path = typer.Option(
        Path("~/ws/sgcc-mma/training/person/datasets/sgcc-n001/crop640_persons/runs/"
             "person_n001_v6_neg0906_s1/weights/best.pt").expanduser(), "--weights"),
    conf: float = typer.Option(0.25, "--conf", help="降阈 0.25=置信带全量（标定警示①）"),
    batch: int = typer.Option(64, "--batch"),
) -> None:
    """v6 对 b2 裁切帧直接推理 → prelabels/v6crop/（窗域归一框+conf）。"""
    from ultralytics import YOLO

    frames = _iter_frames(pkg)
    seat_dir = out_root / "prelabels" / "v6crop"
    seat_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    rows: list[dict] = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        results = model.predict([str(p) for _, p in chunk], conf=conf, verbose=False)
        for (stem, path), res in zip(chunk, results):
            w, h = Image.open(path).size
            boxes = [
                [round(float(v) / (w if k % 2 == 0 else h), 6)
                 for k, v in enumerate(b.xyxy[0].tolist())]
                for b in res.boxes
            ]
            confs = [round(float(b.conf[0]), 6) for b in res.boxes]
            rows.append({"stem": stem, "ok": True, "ms": 0, "boxes": boxes,
                         "error": "", "raw": "", "confs": confs})
        typer.echo(f"v6crop: {min(i + batch, len(frames))}/{len(frames)}", err=True)
    _dump(seat_dir / "labels.jsonl", rows)
    typer.secho(f"v6 crop 域票：{len(rows)} 帧 / {sum(len(r['boxes']) for r in rows)} 框 → {seat_dir}",
                fg=typer.colors.GREEN)


@app.command()
def vote(
    pkg: Path = typer.Option(..., "--pkg"),
    out_root: Path = typer.Option(..., "--out-root"),
    seats: str = typer.Option(",".join(VOTE_SEATS), "--seats"),
    concurrency: int = typer.Option(6, "--concurrency"),
    limit: int = typer.Option(0, "--limit"),
) -> None:
    """VLM 直喂 crop 图投票（断点续跑只认 ok 票；原始响应逐字落盘 #16）。"""
    from jxl.bin.n005_detect_b1 import _vote_seat  # 夜役护栏版（wait_for 180s+flush）

    frames = _iter_frames(pkg)
    rows = [{"stem": s} for s, _ in frames]
    if limit:
        rows = rows[:limit]
    path_by_stem = dict(frames)
    (out_root / "stem2img.json").write_bytes(orjson.dumps(
        {s: str(p) for s, p in frames}, option=orjson.OPT_INDENT_2))
    for seat in seats.split(","):
        ok, err = _vote_seat_frames(seat, rows, path_by_stem, out_root, concurrency)
        rate = ok / max(1, ok + err)
        typer.echo(f"{seat}: 新投 {ok + err}，ok率 {rate:.1%}")
        if rate < 0.8 and ok + err > 5:
            typer.secho(f"{seat} 弃权率超 20%——席位异常，停查", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)


def _vote_seat_frames(seat: str, rows: Sequence[dict], path_by_stem: dict[str, Path],
                      out_root: Path, concurrency: int) -> tuple[int, int]:
    """夜役 _vote_seat 的 b2 变体：图路径来自 stem2img 映射而非 frames/ 约定目录。"""
    pre = out_root / "prelabels" / seat
    pre.mkdir(parents=True, exist_ok=True)
    path = pre / "labels.jsonl"
    done: set[str] = set()
    if path.exists():
        done = {r["stem"] for r in (orjson.loads(line) for line in
                path.read_text().splitlines() if line.strip()) if r["ok"]}
    todo = [r for r in rows if r["stem"] not in done]
    fh = path.open("a")
    ok = err = 0

    async def one(client: httpx.AsyncClient, r: dict) -> None:
        nonlocal ok, err
        t0 = time.monotonic()

        async def _call():
            async with sem:
                url = image_data_url(path_by_stem[r["stem"]])
                return await call_person(client, seat, url, build_target_prompt("person"))

        try:
            v = await asyncio.wait_for(_call(), timeout=180.0)
        except TimeoutError:
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
def aggregate(
    pkg: Path = typer.Option(..., "--pkg"),
    out_root: Path = typer.Option(..., "--out-root"),
    iou_th: float = typer.Option(0.4, "--iou-cluster"),
) -> None:
    """裁切域共识分拣（b1 严格规则+双阈值门）→ confirmed/ YOLO + quarantine.tsv。

    规则：簇支持源≥2 且非「v6 低置信独撑」（conf<0.40 的 v6 框计入支持需另一源
    ——已由 ≥2 源要求天然覆盖）→ 确认；v6 确认框与簇心 IoU<0.55 → 该框移入
    drift 复核清单（整帧其余框不受累）；全源空=负样本；任一单源框/弃权<2=难例。
    """
    frames = dict(_iter_frames(pkg))
    seats = ["v6crop", *VOTE_SEATS]
    votes: dict[str, dict[str, list]] = {s: {} for s in frames}
    confs_by_stem: dict[str, list[float]] = {}
    for seat in seats:
        f = out_root / "prelabels" / seat / "labels.jsonl"
        if not f.is_file():
            raise typer.BadParameter(f"缺席位票 {f}——先跑 v6/vote")
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = orjson.loads(line)
            if rec["stem"] in votes and rec["ok"]:
                votes[rec["stem"]][seat] = rec["boxes"]
                if seat == "v6crop":
                    confs_by_stem[rec["stem"]] = rec.get("confs", [])
    conf_dir = out_root / "confirmed"
    (conf_dir / "labels").mkdir(parents=True, exist_ok=True)
    (conf_dir / "classes.txt").write_text("person\n")
    consensus, quar, confirmed, empties, drift_rows = [], [], [], [], []
    for stem in frames:
        v = votes[stem]
        n_ok, clusters = len(v), _clusters(v, iou_th)
        contested = [c for c in clusters if len(c["seats"]) < 2]
        keep = [c for c in clusters if len(c["seats"]) >= 2]
        decision = "quarantine" if n_ok < 2 or contested else ("confirmed" if keep else "confirmed_empty")
        # IoU 一致门：确认簇内 v6 框 vs 簇心
        if decision == "confirmed":
            cs = clusters
            cf = confs_by_stem.get(stem, [])
            new_keep = []
            for c in keep:
                if "v6crop" in c["seats"]:
                    # v6 在该簇的成员框（原始像素已归一）：找最近 v6 框对簇心 IoU
                    v6_best = max((iou_fn(b, c["box"]) for b in v["v6crop"]
                                   if iou_fn(b, c["box"]) > 0.01), default=1.0)
                    if v6_best < IOU_AGREE:
                        drift_rows.append({"stem": stem, "box": c["box"], "iou_to_center": round(v6_best, 3)})
                        continue
                new_keep.append(c)
            keep = new_keep
            if not keep:
                decision = "confirmed_empty" if not clusters else "quarantine"
        consensus.append({"stem": stem, "decision": decision, "n_ok_votes": n_ok, "clusters": clusters})
        if decision == "quarantine":
            quar.append(stem)
            continue
        if decision == "confirmed":
            confirmed.append(stem)
        else:
            empties.append(stem)
        lines = []
        for c in keep:
            x1, y1, x2, y2 = c["box"]
            lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        (conf_dir / "labels" / f"{stem.replace('/', '__')}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
    _dump(out_root / "consensus.jsonl", consensus)
    (out_root / "quarantine.txt").write_text("".join(s + "\n" for s in quar))
    _dump(out_root / "drift_review.jsonl", drift_rows)
    # 确认图软链清单（img 相对包根路径双下划线名，训练时 rsync -L 物化）
    manifest = []
    for stem in confirmed + empties:
        link = conf_dir / "images" / f"{stem.replace('/', '__')}.jpg"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(frames[stem].resolve())
        manifest.append({"stem": stem, "src_rel": stem})
    _dump(conf_dir / "manifest.jsonl", manifest)
    n_boxes = sum(len([c for c in r["clusters"] if len(c["seats"]) >= 2])
                  for r in consensus if r["decision"] == "confirmed")
    typer.secho(
        f"确认 {len(confirmed)}（{n_boxes} 框）/ 负样本 {len(empties)} / 难例 {len(quar)} / "
        f"框漂移复核 {len(drift_rows)} → {out_root}",
        fg=typer.colors.GREEN,
    )


def iou_fn(a: Sequence[float], b: Sequence[float]) -> float:
    from jxl.label.crop import xyxy_iou
    return xyxy_iou(tuple(a), tuple(b))  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
