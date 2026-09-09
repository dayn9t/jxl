#!/usr/bin/env python3
"""sample_prune CLI: 训练样本簇内削减(近重复优先)+隔离池. spec 见 docs/2026-09-09-训练样本量控制机制设计.md."""

import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from jcx.sys.fs import files_in

from jxl.sample_prune import decide

app = typer.Typer(help="样本簇内削减+隔离池(plan/apply/pool-review)")

# spec §2.3: conf 下降超过该值判回流; removed_round 达该值熔断(永不再回流, 只数回流周期)
_BACKFLOW_CONF_DROP = 0.2
_FUSE_ROUND = 2
_POOL_META_KEYS = ("stem", "conf", "removed_round", "reviews")
_PLAN_KEYS = ("stems", "keep", "pool", "meta")

# confs 对数据集 stems 的命中率低于该值判口径不一致(缺 stem 默认 0.0 会静默全保, 产出假 plan)
MIN_CONF_HIT_RATIO = 0.5


@app.callback()
def main() -> None:
    """样本簇内削减+隔离池. 需显式子命令(plan/apply/pool-review), 保持多命令组形态."""


@app.command()
def plan(
    ds_dir: Annotated[Path, typer.Argument(help="数据集目录(images/+labels/)")],
    embeddings: Annotated[Path, typer.Option(help="帧 DINOv2 embeddings.npy, 行序=stems 排序")],
    confs: Annotated[Path, typer.Option(help="gencheck_infer 格式 jsonl")],
    ratio: Annotated[float, typer.Option(help="削减比例 0-1")] = 0.2,
    out: Annotated[Path, typer.Option(help="plan.json 输出")] = Path("prune_plan.json"),
) -> None:
    stems = [f.stem for f in files_in(ds_dir / "images", ".jpg")]
    emb = np.load(embeddings).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    if len(emb) != len(stems):
        raise SystemExit(f"FATAL: embeddings {len(emb)} != 帧数 {len(stems)} (行序=stems 排序)")
    # 行序交叉校验: embed_dino 产物 <emb>.txt 与数据集排序逐一比对, 防同数错序静默错绑
    sidecar = embeddings.with_suffix(".txt")
    if not sidecar.is_file():
        raise SystemExit(f"FATAL: embed 侧车缺失({sidecar.name}), 无法对齐行序 — 请补齐侧车后重试")
    emb_stems = [Path(line).stem
                 for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    # strict=False: 长度不等本身即不一致, 由下方 len 比对单独报出
    diffs = [f"行{i}: 数据集 {s} / 侧车 {e}"
             for i, (s, e) in enumerate(zip(stems, emb_stems, strict=False)) if s != e][:3]
    if diffs or len(emb_stems) != len(stems):
        raise SystemExit(
            f"FATAL: {sidecar.name} 行序与数据集 stems 排序不一致 "
            f"(侧车 {len(emb_stems)} 行 / 数据集 {len(stems)} 帧): {'; '.join(diffs)}"
        )
    conf_by_stem: dict[str, float] = {}
    for line in confs.open(encoding="utf-8"):
        r = json.loads(line)
        conf_by_stem[r["stem"]] = max(r.get("confs") or [0.0])
    hit = sum(1 for s in stems if s in conf_by_stem)
    if hit < len(stems) * MIN_CONF_HIT_RATIO:  # 乘式比较, len(stems)=0 时不除零
        raise SystemExit(
            f"FATAL: confs 仅命中 {hit}/{len(stems)} stems——疑似 stem 口径不一致 "
            f"(阈值 {MIN_CONF_HIT_RATIO:.0%})"
        )
    confs_sorted = [conf_by_stem.get(s, 0.0) for s in stems]
    p = decide(emb, confs_sorted, ratio)
    plan_doc = {"stems": stems, "keep": list(p.keep), "pool": list(p.pool), "meta": list(p.meta)}
    out.write_text(json.dumps(plan_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    # stems 侧车: plan.json stems 的纯文本镜像(行序=嵌入行序), 供与 <emb>.txt diff 回核行序
    out.with_name(out.name + ".stems.txt").write_text("\n".join(stems), encoding="utf-8")
    typer.echo(f"plan: {len(stems)} 帧 -> keep {len(p.keep)} / pool {len(p.pool)} -> {out}")


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """tmp + os.replace 原子写, 防中途死留半文件(尤其承接历史的 pool_meta)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _link_frame(ds_dir: Path, dst_root: Path, stem: str) -> None:
    """单帧 images+labels symlink 到 dst_root; 源缺失或目标已存在均 FATAL."""
    for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
        src = ds_dir / sub / f"{stem}{ext}"
        if not src.is_file():
            raise SystemExit(f"FATAL: 源文件缺失: {src}")
        dst = dst_root / sub / src.name
        if dst.exists() or dst.is_symlink():
            raise SystemExit(f"FATAL: 目标已存在: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())


@app.command()
def apply(
    ds_dir: Annotated[Path, typer.Argument(help="数据集目录(images/+labels/)")],
    plan: Annotated[Path, typer.Argument(help="plan.json (plan 命令产物)")],
    out_dir: Annotated[Path, typer.Option(help="削减后训练集目录(symlink)")],
    pool_dir: Annotated[Path, typer.Option(help="隔离池目录(symlink + pool_meta.jsonl)")],
) -> None:
    doc = json.loads(plan.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not all(k in doc for k in _PLAN_KEYS):
        raise SystemExit(f"FATAL: plan.json 结构非法: 需字段 {_PLAN_KEYS}")
    stems = doc["stems"]
    if (not isinstance(stems, list) or not all(isinstance(s, str) for s in stems)
            or not isinstance(doc["keep"], list) or not isinstance(doc["pool"], list)
            or not isinstance(doc["meta"], list)):
        raise SystemExit(f"FATAL: plan.json 结构非法: {_PLAN_KEYS} 需为列表(stems 元素为 str)")
    bad_meta = [m for m in doc["meta"] if not isinstance(m, dict)
                or not isinstance(m.get("idx"), int) or not 0 <= m["idx"] < len(stems)]
    if bad_meta:
        raise SystemExit(f"FATAL: plan.json 结构非法: {len(bad_meta)} 条 meta 的 idx 非法/越界")
    # 防 plan 与数据集错配 (同数错序/删帧后复用旧 plan)
    ds_stems = [f.stem for f in files_in(ds_dir / "images", ".jpg")]
    if ds_stems != stems:
        raise SystemExit(f"FATAL: plan.stems 与数据集不一致 (plan {len(stems)} / 数据集 {len(ds_stems)} 帧)")
    if out_dir.exists() or out_dir.is_symlink():
        raise SystemExit(f"FATAL: 输出目录已存在: {out_dir}")
    # 跨周期承接: 回流后再删场景, pool_dir 已有上一周期 pool_meta → 按 stem 承接熔断计数与评测史
    old_by_stem: dict[str, dict] = {}
    if pool_dir.exists() or pool_dir.is_symlink():
        old_meta_path = pool_dir / "pool_meta.jsonl"
        if not old_meta_path.is_file():
            raise SystemExit(f"FATAL: pool_dir 已存在但缺 pool_meta.jsonl: {pool_dir}")
        for line in old_meta_path.open(encoding="utf-8"):
            row = json.loads(line)
            if not isinstance(row, dict) or not all(k in row for k in _POOL_META_KEYS):
                raise SystemExit(f"FATAL: 旧 pool_meta.jsonl 结构非法(缺 {_POOL_META_KEYS})")
            old_by_stem[row["stem"]] = row
    for i in doc["keep"]:
        _link_frame(ds_dir, out_dir, stems[i])
    meta_rows = []
    n_carry = 0
    for m in doc["meta"]:
        stem = stems[m["idx"]]
        _link_frame(ds_dir, pool_dir, stem)
        old = old_by_stem.get(stem)
        if old is None:
            meta_rows.append({**m, "stem": stem, "reviews": []})
        else:  # 新一轮删除: 熔断计数 +1, 评测史保留
            n_carry += 1
            meta_rows.append({**m, "stem": stem, "removed_round": old["removed_round"] + 1,
                              "reviews": list(old["reviews"])})
    _write_jsonl_atomic(pool_dir / "pool_meta.jsonl", meta_rows)
    typer.echo(f"apply: keep {len(doc['keep'])} -> {out_dir} / pool {len(meta_rows)}"
               f"(承接 {n_carry} 帧历史) -> {pool_dir}")


def _infer_confs(model: str, img_dir: Path) -> dict[str, float]:
    """ultralytics 批量推理 -> {stem: 帧内最大 conf}. imgsz 640 / conf 0.25 (spec §2.3)."""
    # 延迟导入: plan/apply 与 --help 不付 ultralytics 导入成本
    from ultralytics import YOLO

    weights = Path(model)
    if not weights.is_file():
        raise SystemExit(f"FATAL: 模型权重不存在: {weights}")
    out: dict[str, float] = {}
    for res in YOLO(str(weights)).predict([str(p) for p in files_in(img_dir, ".jpg")],
                                          imgsz=640, conf=0.25, verbose=False, stream=True):
        boxes = res.boxes
        out[Path(res.path).stem] = float(boxes.conf.max()) if boxes is not None and len(boxes) else 0.0
    return out


@app.command()
def pool_review(
    pool_dir: Annotated[Path, typer.Argument(help="隔离池目录(apply 产物)")],
    model: Annotated[str, typer.Option(help="YOLO .pt 权重路径")],
    out: Annotated[Path, typer.Option(help="backflow.jsonl 输出")] = Path("backflow.jsonl"),
) -> None:
    meta_path = pool_dir / "pool_meta.jsonl"
    if not meta_path.is_file():
        raise SystemExit(f"FATAL: 隔离池元数据缺失: {meta_path}")
    rows = [json.loads(line) for line in meta_path.open(encoding="utf-8")]
    bad = [str(r.get("stem", f"行{i}")) for i, r in enumerate(rows)
           if not all(k in r for k in _POOL_META_KEYS)]
    if bad:
        raise SystemExit(f"FATAL: pool_meta.jsonl 缺字段 {_POOL_META_KEYS}: {bad[:5]}")
    new_confs = _infer_confs(model, pool_dir / "images")
    missing = [r["stem"] for r in rows if r["stem"] not in new_confs]
    if missing:
        raise SystemExit(f"FATAL: 推理结果缺失 {len(missing)} 帧: {missing[:5]}")
    records = []
    for row in rows:
        prev, new = float(row["conf"]), new_confs[row["stem"]]
        if row["removed_round"] >= _FUSE_ROUND:  # 熔断: 永不再回流, 不增轮
            rec = {"stem": row["stem"], "prev_conf": prev, "new_conf": new,
                   "backflow": False, "reason": "fused"}
        else:
            if new == 0.0:  # 检出丢失
                backflow, reason = True, "lost"
            elif prev - new > _BACKFLOW_CONF_DROP:
                backflow, reason = True, "conf_drop"
            else:
                backflow, reason = False, "stable"
            if backflow:
                row["removed_round"] += 1  # 增轮只数回流周期; 稳定帧不累积, 不受熔断误伤
            rec = {"stem": row["stem"], "prev_conf": prev, "new_conf": new,
                   "backflow": backflow, "reason": reason}
        row["reviews"].append(rec)
        records.append(rec)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    _write_jsonl_atomic(meta_path, rows)
    fused = sum(1 for r in records if r["reason"] == "fused")
    typer.echo(f"pool-review: {len(rows)} 帧 -> 回流 {sum(1 for r in records if r['backflow'])} / "
               f"熔断 {fused} -> {out}")


if __name__ == "__main__":
    app()
