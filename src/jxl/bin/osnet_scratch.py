#!/usr/bin/env python3
"""OSNet x0_75 (BN 版) 全新训练（from scratch）— 低帧率跟踪阶段③ 训练线第一轮基线。

裁决链（2026-11-14 用户令，iap-37 转达）：A 案开工 + **全新训练**（非预训练增量微调，
理由：OSNet×center3 新生产口径非同一任务谱系）；训练机 n902（用户指定，4060 Ti 16G）。
配方正本：projects/sgcc/research/2026-11-14-osnet-scratch-训练配方-v1.md。

与 v2.1 微调线（数据盘 gencheck/osnet_ft.py，本脚本的拷贝源）的唯一差异 = 初始化：
  **随机初始化，不加载 MSMT17 基座**——实验中若发现预训练初始化显著优势，
  按程序呈报用户再定，不自行改判（监督条款）。
数据/采样/评估协议与 v2.1 同源（保可比）：
  数据   iapx pair list（train 1716 对 / eval 328 对 held-out）join manifest 切 crop
         identity = train 正对并查集；held-out 绝不入训练（默认再剔重叠 train crop）
  训练   P×K batch 采样 + batch-hard triplet（v2.1 终选配方）+ warmup 5 ep
  验收   held-out 余弦距离分布分离：same_p95 < diff_p5（gap 越小越分离）
  推理   N×3×256×128 / ImageNet mean+std / 512-d 输出（消费端 L2 归一）

用法：
  --stage build                 本地建 crops/{train,heldout} + crops.jsonl + 统计
  --stage train                 训练（本机 CPU 冒烟：--run-name smoke --limit-ids 4 --epochs 2）
  --stage eval                  held-out 328 对算 same_p95/diff_p95/gap
  --remote                      rsync n902 + setsid nohup train→eval（脱离会话长跑）
  --pull                        取回 best.pth + eval_report.json

环境变量（n902 上必设；本机缺省指数据盘）：
  OSNET_SCRATCH_WORK  工作区（crops/runs 落位；n902=~/staging/jxl-reid-osnet/work）
  OSNET_SCRATCH_DATA  样本根（samples/{manifest,pairs}；n902=~/staging/jxl-reid-osnet/samples）

OSNet 定义: ~/cc/py/jxl/src/jxl/vdt/vendored/osnet.py（KaiyangZhou/deep-person-reid, MIT）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import shlex
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


def _ws(p: str) -> Path:
    """路径自适应：/mnt/data/jiang/ws 不存在时替换为 /home/jiang/ws（对等路径惯例）。"""
    path = Path(p)
    return path if path.exists() else Path(str(path).replace("/mnt/data/jiang/ws", "/home/jiang/ws"))


GENCHECK = Path(__file__).resolve().parent
WORK = _ws(os.environ.get("OSNET_SCRATCH_WORK",
                          "/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/"
                          "crop640_persons/gencheck/osnet_scratch"))
OUT = WORK
EXTRA_PAIRS = OUT / "drift_pairs.jsonl"  # v2.1 姿态漂移增量（osnet_drift_mine 产，可选存在）
CROP_DIR = OUT / "crops"
INDEX = OUT / "crops.jsonl"
BUILD_STATS = OUT / "build_stats.json"
RUNS = OUT / "runs"
SEED = 20260913

VENDOR_CANDIDATES = [Path.home() / "cc/py/jxl/src/jxl/vdt/vendored/osnet.py",
                     GENCHECK / "vendor_osnet.py"]  # 后者 = --remote rsync 落位目标

IMG_W, IMG_H = 128, 256
ASPECT = IMG_W / IMG_H  # w/h，crop pad 目标长宽比
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEAT_DIM = 512
SPLIT_TRAIN, HELDOUT = "train", "heldout"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REMOTE_ROOT = "~/staging/jxl-reid-osnet/work"
REMOTE_PYTHON = "~/staging/jxl-reid-osnet/.venv/bin/python"
REMOTE_HOST = "sgcc-n902"  # 训练机（用户指定 2026-11-14；4060 Ti 16G 半栈机，avs 常驻 ~1.8G）
RUN_NAME = "osnet_x075_scratch_v1"
P_DEFAULT, K_DEFAULT, LR_DEFAULT = 16, 2, 1e-3


DATA_ROOT = _ws(os.environ.get("OSNET_SCRATCH_DATA", "/mnt/data/jiang/ws/iapx/n001/samples"))
PAIRS_TRAIN = DATA_ROOT / "pairs" / "train.jsonl"
PAIRS_EVAL = DATA_ROOT / "pairs" / "eval.jsonl"
MANIFEST = DATA_ROOT / "manifest.jsonl"
N001_ROOT = DATA_ROOT.parent  # manifest file 字段形如 samples/1/...（相对 n001 根）


# ---------------------------------------------------------------- pair / crop 解析

@dataclass(frozen=True)
class Crop:
    uid: str
    src: int
    mkv: str
    stem: str
    det_index: int
    frame: Path
    box_norm: tuple[float, float, float, float]


def crop_key(row: dict, side: dict) -> tuple[int, str, str, int]:
    """(source_id, mkv, stem, det_index)；eval 行 mkv 带 '1/0/' / '2/0/' 前缀须剥掉取 source。"""
    segs = side["mkv"].split("/")
    i = 0
    while i < len(segs) and not DATE_RE.match(segs[i]):
        i += 1
    if i >= len(segs):
        raise ValueError(f"mkv 无日期段: {side['mkv']}")
    src = int(segs[0]) if i > 0 else int(row["source_id"])
    return src, "/".join(segs[i:]), side["stem"], int(side["det_index"])


def side_uid(row: dict, side: dict, by_uid: set[str] | None = None) -> str:
    src, mkv, stem, det = crop_key(row, side)
    uid = f"s{src}_{mkv.split('/')[0]}_{stem}_d{det}"
    if by_uid is not None and uid not in by_uid:
        raise KeyError(f"crop 不在 index: {uid}")
    return uid


def load_manifest_index() -> dict[tuple[int, str, str], tuple[str, list[list[float]]]]:
    """join 键 (source_id, mkv, wallclock=stem) -> (相对帧路径, 各 det 的 box_norm)。"""
    index: dict[tuple[int, str, str], tuple[str, list[list[float]]]] = {}
    with MANIFEST.open() as fh:
        for line in fh:
            row = json.loads(line)
            boxes = [d["box_norm"] for d in row["detections"]]
            index[(int(row["source_id"]), row["mkv"], row["wallclock"])] = (row["file"], boxes)
    return index


def resolve_crop(key: tuple[int, str, str, int],
                 manifest: dict[tuple[int, str, str], tuple[str, list[list[float]]]]) -> Crop:
    src, mkv, stem, det = key
    hit = manifest.get((src, mkv, stem))
    if hit is None:
        raise KeyError(f"manifest 无帧: {key}")
    rel_file, boxes = hit
    if not 0 <= det < len(boxes):
        raise KeyError(f"det_index 越界: {key} (n_dets={len(boxes)})")
    uid = f"s{src}_{mkv.split('/')[0]}_{stem}_d{det}"
    return Crop(uid, src, mkv, stem, det, N001_ROOT / rel_file, tuple(boxes[det]))


def read_pairs(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple, tuple] = {}

    def find(self, x: tuple) -> tuple:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: tuple, b: tuple) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def components(self) -> list[list[tuple]]:
        groups: dict[tuple, list[tuple]] = defaultdict(list)
        for k in self.parent:
            groups[self.find(k)].append(k)
        return sorted(sorted(members) for members in groups.values())


def build_identities(pairs: list[dict]) -> tuple[dict[tuple, str], Counter]:
    """identity = label=same 正对并查集连通分量；返回 crop_key -> id 名（id000…按字典序确定）。"""
    uf, stats = UnionFind(), Counter()
    for row in pairs:
        ka, kb = crop_key(row, row["a"]), crop_key(row, row["b"])
        if row["label"] != "same":
            stats["neg_pairs"] += 1
            continue
        if ka == kb:
            stats["self_pairs_dropped"] += 1
            continue
        uf.union(ka, kb)
        stats["pos_pairs"] += 1
    assign: dict[tuple, str] = {}
    for i, members in enumerate(uf.components()):
        for k in members:
            assign[k] = f"id{i:03d}"
    return assign, stats


def box_to_pixels(box_norm: tuple[float, ...], w: int, h: int, aspect: float = ASPECT) -> tuple[int, int, int, int]:
    """norm bbox(x1,y1,x2,y2) -> 像素 bbox；按目标长宽比中心扩边后 clamp（纯函数）。"""
    x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in box_norm)
    px1, py1, px2, py2 = x1 * w, y1 * h, x2 * w, y2 * h
    bw, bh = px2 - px1, py2 - py1
    if bw <= 0 or bh <= 0:
        raise ValueError(f"退化 bbox: {box_norm}")
    tw, th = max(bh * aspect, bw), max(bw / aspect, bh)
    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
    return (int(round(max(0.0, cx - tw / 2))), int(round(max(0.0, cy - th / 2))),
            int(round(min(float(w), cx + tw / 2))), int(round(min(float(h), cy + th / 2))))


# ---------------------------------------------------------------- stage: build

def build(exclude_eval_crops: bool = True) -> dict:
    manifest = load_manifest_index()
    train_rows, eval_rows = read_pairs(PAIRS_TRAIN), read_pairs(PAIRS_EVAL)
    n_extra = 0
    if EXTRA_PAIRS.exists():  # v2.1 姿态漂移增量（09-16 挖掘，铁证 5 + 困难 114）
        extra = read_pairs(EXTRA_PAIRS)
        n_extra = len(extra)
        train_rows = train_rows + extra
        print(f"[extra] +{n_extra} drift pairs <- {EXTRA_PAIRS.name}")
    eval_uids = {side_uid(r, s) for r in eval_rows for s in (r["a"], r["b"])}

    kept_rows, dropped = [], Counter()
    for row in train_rows:
        if exclude_eval_crops and any(
                side_uid(row, row[s]) in eval_uids for s in ("a", "b")):
            dropped[f"train_pairs_touch_eval_crop/{row['label']}"] += 1
            continue
        kept_rows.append(row)

    identity, uf_stats = build_identities(kept_rows)
    unique: dict[tuple[str, str], dict] = {}
    for row in kept_rows:
        for s in ("a", "b"):
            key = crop_key(row, row[s])
            if key not in identity:
                dropped["train_crops_no_identity"] += 1
                continue
            crop = resolve_crop(key, manifest)
            unique[(SPLIT_TRAIN, crop.uid)] = _crop_row(crop, SPLIT_TRAIN, identity[key])
    for row in eval_rows:
        for s in ("a", "b"):
            crop = resolve_crop(crop_key(row, row[s]), manifest)
            unique[(HELDOUT, crop.uid)] = _crop_row(crop, HELDOUT, "")
    rows_out = list(unique.values())
    _write_crop_files(rows_out)

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with INDEX.open("w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    id_crops = Counter(r["identity"] for r in rows_out if r["split"] == SPLIT_TRAIN)
    train_uids = {r["uid"] for r in rows_out if r["split"] == SPLIT_TRAIN}
    stats = {
        "train_pairs_in": len(train_rows), "train_pairs_kept": len(kept_rows),
        "drift_extra_pairs": n_extra,
        "train_dropped": dict(dropped), "union_find": dict(uf_stats),
        "heldout_pairs": len(eval_rows), "heldout_crops": len(eval_uids),
        "crops_train": len(train_uids), "crops_heldout": len(rows_out) - len(train_uids),
        "identities": len(id_crops),
        "identity_size_hist": dict(sorted(Counter(id_crops.values()).items())),
        "exclude_eval_crops": exclude_eval_crops,
        "heldout_leak_crops": len(train_uids & eval_uids),
        "pk_feasible": {"p": min(P_DEFAULT, len(id_crops)),
                        "k": min(K_DEFAULT, min(id_crops.values()) if id_crops else 0)},
    }
    BUILD_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n")
    print("build:", json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


def _crop_row(crop: Crop, split: str, identity: str) -> dict:
    return {"uid": crop.uid, "split": split, "identity": identity, "src": crop.src,
            "mkv": crop.mkv, "stem": crop.stem, "det_index": crop.det_index,
            "frame": str(crop.frame), "box_norm": list(crop.box_norm),
            "path": str(CROP_DIR / split / identity / f"{crop.uid}.jpg")}


def _write_crop_files(rows: list[dict]) -> int:
    """切图（bbox 按长宽比 pad 后 resize 128×256）；已存在则跳过（断点重入）。"""
    from PIL import Image
    written = 0
    for row in rows:
        dst = Path(row["path"])
        if dst.exists():
            continue
        with Image.open(row["frame"]) as im:
            rgb = im.convert("RGB")
            crop = rgb.crop(box_to_pixels(tuple(row["box_norm"]), rgb.width, rgb.height))
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.resize((IMG_W, IMG_H), Image.BILINEAR).save(dst, quality=95)
        written += 1
    print(f"[crops] 新切 {written} 张（跳过已存在 {len(rows) - written} 张）")
    return written


# ---------------------------------------------------------------- model / loss

def load_osnet_module():
    for cand in VENDOR_CANDIDATES:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("vendor_osnet", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[osnet] vendor 定义: {cand}")
            return mod
    raise FileNotFoundError(f"缺 vendored osnet.py: {VENDOR_CANDIDATES}（--remote 会同步 vendor_osnet.py）")


def build_model(n_classes: int):
    return load_osnet_module().osnet_x0_75(num_classes=n_classes, pretrained=False, loss="triplet")


def circle_loss(emb, labels, m: float = 0.25, gamma: int = 256):
    """官方式 CircleLoss（emb L2 归一；mask 零填充 + logsumexp）。"""
    import torch
    from torch.nn import functional as F
    sim = emb @ emb.T
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    neg = labels[:, None] != labels[None, :]
    ap = (-sim + 1 + m).clamp(min=0)
    an = (sim + m).clamp(min=0)
    logit_p = torch.where(pos, -gamma * ap * (sim - (1 - m)), torch.zeros_like(sim))
    logit_n = torch.where(neg, gamma * an * (sim - m), torch.zeros_like(sim))
    if not pos.any() or not neg.any():
        raise ValueError("P×K batch 内无正/负对（检查 P、K 配置）")
    return F.softplus(logit_p.logsumexp(dim=1) + logit_n.logsumexp(dim=1)).mean()


def batch_hard_triplet(emb, labels, margin: float = 0.3):
    import torch
    from torch.nn import functional as F
    dist = torch.cdist(emb, emb)
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    same = labels[:, None] == labels[None, :]
    pos_d = torch.where(same & ~eye, dist, torch.full_like(dist, -1e9)).max(dim=1).values
    neg_d = torch.where(~same, dist, torch.full_like(dist, 1e9)).min(dim=1).values
    valid = (pos_d > -1e8) & (neg_d < 1e8)
    if not valid.any():
        raise ValueError("P×K batch 内无正/负对（检查 P、K 配置）")
    return F.relu(pos_d[valid] - neg_d[valid] + margin).mean()


# ---------------------------------------------------------------- embed / gate

def embed(model, files: list[Path], device: str, batch: int = 64):
    """推理契约：N×3×256×128 / ImageNet mean+std / 无 TTA / 512-d 未归一化输出。"""
    import torch
    from PIL import Image
    from torchvision import transforms as T
    tf = T.Compose([T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BILINEAR),
                    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(files), batch):
            x = torch.stack([tf(Image.open(str(f)).convert("RGB")) for f in files[i:i + batch]])
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out)


def heldout_gap(model, index_rows: list[dict], device: str) -> dict:
    """held-out 对余弦距离（L2 归一后）。距离口径下分布分离判据 = same_p95 < diff_p5
    （最难同对 < 最易混异对），gap = same_p95 − diff_p5（<0 = 门过）。

    2026-09-15 判据修复：原实现 gap = same_p5 − diff_p95 且判 >0——那是 iapx §7.4
    相似度口径的判据标签，套在余弦距离值上方向反了（实际惩罚 same 对变近）。v1 与
    v2 首三轮的 best 选择/早停均基于反向 gap，其数字与本修复后不可比。
    """
    import torch
    by_uid = {r["uid"]: Path(r["path"]) for r in index_rows if r["split"] == HELDOUT}
    uids = sorted(by_uid)
    uids_set = set(uids)
    feats = torch.nn.functional.normalize(embed(model, [by_uid[u] for u in uids], device), dim=1)
    pos = {u: i for i, u in enumerate(uids)}
    same_d, diff_d = [], []
    for row in read_pairs(PAIRS_EVAL):
        ua = side_uid(row, row["a"], uids_set)
        ub = side_uid(row, row["b"], uids_set)
        d = 1.0 - float(feats[pos[ua]] @ feats[pos[ub]])
        (same_d if row["label"] == "same" else diff_d).append(d)

    def pct(sorted_vals: list[float], q: float) -> float:
        return sorted_vals[int(q * (len(sorted_vals) - 1))] if sorted_vals else float("nan")

    same_d, diff_d = sorted(same_d), sorted(diff_d)
    same_p5, same_p95 = pct(same_d, 0.05), pct(same_d, 0.95)
    diff_p5, diff_p95 = pct(diff_d, 0.05), pct(diff_d, 0.95)
    return {"same_n": len(same_d), "diff_n": len(diff_d),
            "same_p5": round(same_p5, 6), "same_p95": round(same_p95, 6),
            "diff_p5": round(diff_p5, 6), "diff_p95": round(diff_p95, 6),
            "gap": round(same_p95 - diff_p5, 6),
            "same_mean": round(sum(same_d) / len(same_d), 6),
            "diff_mean": round(sum(diff_d) / len(diff_d), 6),
            "gate_pass": bool(same_p95 < diff_p5)}


# ---------------------------------------------------------------- stage: train

def train(args) -> None:
    import torch
    from PIL import Image
    from torch import nn
    from torchvision import transforms as T

    random.seed(SEED), torch.manual_seed(SEED)
    rows = [json.loads(line) for line in INDEX.open()]
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] == SPLIT_TRAIN:
            by_id[r["identity"]].append(r)
    if not by_id:
        raise RuntimeError(f"{INDEX} 无 train crops：先 --stage build")
    ids = sorted(by_id)[:args.limit_ids or None]
    by_id = {i: by_id[i] for i in ids}
    k_eff = min(args.k, min(len(v) for v in by_id.values()))
    p_eff = min(args.p, len(ids))
    if k_eff < args.k or p_eff < args.p:
        print(f"[sampler] P {args.p}->{p_eff} / K {args.k}->{k_eff}（数据约束显式降档）")
    label_of = {i: n for n, i in enumerate(ids)}
    aug = T.Compose([T.RandomHorizontalFlip(0.5), T.Pad(10), T.RandomCrop((IMG_H, IMG_W)),
                     T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(),
                     T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "best.pth").exists() and not args.resume:
        raise RuntimeError(f"{run_dir} 已有 best.pth；--resume 续训或换 --run-name")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}（GPU 训练前确认空闲：nvidia-smi）")
    if device == "cuda" and not args.gpu_guard_off:  # 启动守卫：卡须基本空闲（防撞在跑的
        # 训练）——按占用比例判，不按绝对 GiB（4090 24G / 5060 Ti 16G 混用，20GiB 旧阈值
        # 在 16G 卡上恒拒启）。豁免：--gpu-guard-off 显式放行（本机 ≤3h 轻量授权 2026-09-18：
        # 业务常驻 ~10G 时微调 <2G 仍安全，显式 override 非静默降级）
        free, total = torch.cuda.mem_get_info()
        if free < total * 0.8 or free < 4 * 1024**3:
            raise RuntimeError(f"GPU 不够空闲（free={free / 2**30:.1f}GiB / "
                               f"total={total / 2**30:.1f}GiB，须 ≥80% 空闲且 ≥4GiB）"
                               "——先确认 nvidia-smi 空闲，或评估余量后 --gpu-guard-off")
    model = build_model(n_classes=len(ids)).to(device)
    # scratch 裁决（2026-11-14）：随机初始化，不加载任何预训练基座——
    # 预训练优势显著须呈报用户再定，不自行改判
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    warmup_ep = 5  # scratch 冷启动 warmup：前 N ep 线性升到 args.lr（常数大 lr 直接冲有崩风险）

    state = {"epoch": 0, "best_gap": float("inf"), "no_improve": 0}
    if args.resume and (run_dir / "last.pth").exists():
        ck = torch.load(run_dir / "last.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state.update(ck["state"])
        print(f"[resume] epoch={state['epoch']} best_gap={state['best_gap']}")

    def evaluate(tag: str) -> dict:
        gap = heldout_gap(model, rows, device)
        line = {"tag": tag, "epoch": state["epoch"], **gap}
        with (run_dir / "train_log.jsonl").open("a") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        print("[eval]", json.dumps(line, ensure_ascii=False), flush=True)
        return gap

    def save(name: str, gap: float) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "meta": {"num_classes": len(ids), "feature_dim": FEAT_DIM,
                             "p": p_eff, "k": k_eff, "loss": args.loss, "lr": args.lr,
                             "init": "scratch", "gap": gap, "run_name": args.run_name}},
                   run_dir / name)

    print("[baseline] 随机初始化 held-out 表现（训练前对照点）")
    evaluate("baseline")
    steps = max(1, sum(len(v) for v in by_id.values()) // (p_eff * k_eff))
    print(f"[train] ids={len(ids)} crops={sum(len(v) for v in by_id.values())} "
          f"P×K={p_eff}×{k_eff} steps/epoch={steps} loss={args.loss} lr={args.lr}")
    model.train()
    while True:
        state["epoch"] += 1
        if state["epoch"] <= warmup_ep:
            warm_lr = args.lr * state["epoch"] / warmup_ep
            for g in opt.param_groups:
                g["lr"] = warm_lr
        elif state["epoch"] == warmup_ep + 1:
            for g in opt.param_groups:
                g["lr"] = args.lr
        for _ in range(steps):
            files, labels = [], []
            for i in random.sample(ids, p_eff):
                for r in random.sample(by_id[i], k_eff):
                    files.append(Path(r["path"]))
                    labels.append(label_of[i])
            x = torch.stack([aug(Image.open(str(f)).convert("RGB")) for f in files]).to(device)
            y = torch.tensor(labels, device=device)
            _, v = model(x)
            v = nn.functional.normalize(v, dim=1)
            loss = circle_loss(v, y) if args.loss == "circle" else batch_hard_triplet(v, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if state["epoch"] % args.eval_every == 0:
            gap = evaluate(f"ep{state['epoch']}")
            save("last.pth", gap["gap"])
            if gap["gap"] < state["best_gap"]:  # 距离口径 gap 越小越分离（判据修复后取 min）
                state.update(best_gap=gap["gap"], no_improve=0)
                save("best.pth", gap["gap"])
                print(f"[best] gap={gap['gap']} @ epoch {state['epoch']}")
            else:
                state["no_improve"] += 1
            if state["no_improve"] >= args.patience:
                print(f"[early-stop] {args.patience} 次评估无提升 (best_gap={state['best_gap']})")
                break
            model.train()
        if state["epoch"] >= args.epochs:
            print(f"[done] epochs={args.epochs} best_gap={state['best_gap']}")
            break


# ---------------------------------------------------------------- stage: eval / remote

def evaluate(args) -> dict:
    import torch
    run_dir = RUNS / args.run_name
    ckpt_path = Path(args.weights) if args.weights else run_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺 checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_classes=ck["meta"]["num_classes"])
    model.load_state_dict(ck["model"])
    model.to(device)
    rows = [json.loads(line) for line in INDEX.open()]
    report = heldout_gap(model, rows, device)
    report.update({"checkpoint": str(ckpt_path), "device": device, "meta": ck["meta"],
                   "gate": "PASS" if report["gate_pass"] else "FAIL"})
    (run_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("[eval-report]", json.dumps(report, ensure_ascii=False, indent=1))
    return report


def ssh_run(host: str, script: str) -> str:
    proc = subprocess.run(["ssh", host, script], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ssh {host} 失败({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout.strip()


def remote(args, host: str = REMOTE_HOST) -> None:
    for p in (INDEX, CROP_DIR, PAIRS_EVAL):
        if not p.exists():
            raise FileNotFoundError(f"缺 {p}：先 --stage build")
    local_vendor = next((c for c in VENDOR_CANDIDATES if c.exists()), None)
    if local_vendor is None:
        raise FileNotFoundError(f"缺 vendored osnet.py: {VENDOR_CANDIDATES}")
    run_cmd = lambda cmd: (print("+", " ".join(cmd), flush=True), subprocess.run(cmd, check=True))
    samples_remote = "~/staging/jxl-reid-osnet/samples"
    run_cmd(["ssh", host, f"mkdir -p {REMOTE_ROOT}/runs {samples_remote}/pairs"])
    run_cmd(["rsync", "-aL", str(local_vendor), f"{host}:{REMOTE_ROOT}/vendor_osnet.py"])
    run_cmd(["rsync", "-aL", str(Path(__file__)), f"{host}:{REMOTE_ROOT}/"])
    run_cmd(["rsync", "-aL", str(INDEX), f"{host}:{REMOTE_ROOT}/crops.jsonl.tmp"])
    run_cmd(["rsync", "-a", str(CROP_DIR) + "/", f"{host}:{REMOTE_ROOT}/crops/"])
    # pair list 增量重产后远端旧文件会致口径错（v1 实证教训）——PAIRS_EVAL 始终重同步
    run_cmd(["rsync", "-aL", str(PAIRS_EVAL), f"{host}:{samples_remote}/pairs/eval.jsonl"])
    # 远端路径契约（v1 教训固化）：INDEX 内 path 为本机绝对路径（/mnt/data/jiang/ws），
    # 同步后在远端整体替换为 staging 根
    ssh_run(host, f"sed 's|/mnt/data/jiang/ws|~/staging/jxl-reid-osnet|g' "
                  f"{REMOTE_ROOT}/crops.jsonl.tmp > {REMOTE_ROOT}/crops.jsonl && "
                  f"rm {REMOTE_ROOT}/crops.jsonl.tmp")
    log = f"{REMOTE_ROOT}/runs/{args.run_name}.nohup.log"
    env = (f"OSNET_SCRATCH_WORK={REMOTE_ROOT} OSNET_SCRATCH_DATA={samples_remote}")
    inner = (f"{env} {REMOTE_PYTHON} {Path(__file__).name} --stage train --run-name {args.run_name} "
             f"--epochs {args.epochs} --p {args.p} --k {args.k} --lr {args.lr} --loss {args.loss} "
             f"--patience {args.patience} --eval-every {args.eval_every}"
             f"{' --resume' if args.resume else ''} && {env} {REMOTE_PYTHON} {Path(__file__).name} "
             f"--stage eval --run-name {args.run_name}")
    # stdout 重定向纪律（v1 教训 2026-09-15）：train&&eval 整体 bash -c 包裹统一重定向，
    # 防 ssh 会话关闭管道破裂死进程
    cmd = (f"cd {REMOTE_ROOT} && setsid nohup bash -c {shlex.quote(inner)} "
           f"> {log} 2>&1 < /dev/null & echo launched_pid=$!")
    print(ssh_run(host, cmd), flush=True)
    print(f"已脱离启动。跟踪: ssh {host} 'tail -f {log}'")


def pull(host: str = REMOTE_HOST, run_name: str = RUN_NAME) -> None:
    dst = RUNS / run_name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("best.pth", "last.pth", "train_log.jsonl", "eval_report.json"):
        subprocess.run(["rsync", "-aL", f"{host}:{REMOTE_ROOT}/runs/{run_name}/{f}",
                        str(dst / f)], check=False)
    print(f"已取回 -> {dst}")


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="OSNet x0_75 全新训练（scratch，阶段③训练线）")
    ap.add_argument("--stage", choices=["build", "train", "eval"], default="build")
    ap.add_argument("--remote", action="store_true", help="同步 n902 并脱离启动 train→eval")
    ap.add_argument("--pull", action="store_true", help="从 n902 取回 best.pth + eval_report.json")
    ap.add_argument("--run-name", default=RUN_NAME)
    ap.add_argument("--device", default="", help="cuda/cpu；缺省自动（打印所选）")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--p", type=int, default=P_DEFAULT, help="P×K 采样 P（每 batch 身份数）")
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="P×K 采样 K（每身份 crop 数）")
    ap.add_argument("--lr", type=float, default=LR_DEFAULT)
    ap.add_argument("--loss", choices=["circle", "triplet"], default="triplet")
    ap.add_argument("--patience", type=int, default=20, help="早停：连续 N 次评估 gap 无提升")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--limit-ids", type=int, default=0, help=">0 只取前 N 个身份（CPU 冒烟用）")
    ap.add_argument("--resume", action="store_true", help="从 last.pth 续训")
    ap.add_argument("--gpu-guard-off", action="store_true",
                    help="豁免 GPU 空闲守卫（业务常驻时显式放行，微调 <2G 余量足的场景）")
    ap.add_argument("--keep-eval-crops", action="store_true",
                    help="build 不剔除与 held-out 重叠的 train crop（默认剔除，防评估污染）")
    ap.add_argument("--weights", default="", help="eval 指定 checkpoint（缺省 runs/<run>/best.pth）")
    args = ap.parse_args()

    if args.pull:
        pull(run_name=args.run_name)
        return
    if args.remote:
        remote(args)
        return
    if args.stage == "build":
        build(exclude_eval_crops=not args.keep_eval_crops)
    elif args.stage == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
