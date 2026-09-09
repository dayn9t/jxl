#!/usr/bin/env python3
"""S1 难例集合构建：六判据筛选 → hardcase_list + VLabel 审核项目 + 拼图总览。

流程缺口修补（2026-09-09 用户指出）：S1 全自动直通训练漏了人工兜底层，本脚本按
consensus-labeling skill ⑧ hardcase 模式（SGCC purge_review 先例）补筛人工难例。

判据（优先级序，(rel, obj) 去重后 reason 合并）：
  C1_pending  det 终稿框未匹配 attr 判定 → 属性标 pending(-1)（s1_relabel_v1 已预填）
  C1_lowconf  attr 任一属性 conf<=0.6（VLM 刻度 0.9/0.95 为常规，0.8 为常规存疑，
              <=0.6 为极低；仅收已落地对象——交集帧被 det 终稿弃框的判定作废）
  C2_amount   amount 新旧不一致且新 conf<0.9（一致率 48.5% 的分歧带）
  C3_veto     det 轨 VLM not_bucket 否决框（la 锚框过闸但 VLM 判非桶；重建自 preds+vlm_cache）
  C4_emptyfp  空帧轨 la∪gdino 检出被 VLM 判误检的框（含真边界），按 conf 均匀抽 300
  C5_dump     dump 类全量（族弱——VLM 定类样本最少）
  C6_form     形态词典特殊形态 bagged/round/basket 各 30（检测 conf 最低端=边界）

总量：<=2,500 框 / <=1,500 帧；超帧时按 C1_lowconf min-conf 升序保留（最不确定优先）。

产物：
  projects/shtm/s1/hardcase_list.jsonl（rel, obj_idx, cls, reasons, detail, action）
  /home/jiang/ws/trash/s1_hardcase_review/{meta.json5, vlabels/, images/(symlink)}
  projects/shtm/research/2026-09-09-S1难例拼图.jpg
审核项目 vlabels 复制自 s1_relabel_v1 对应帧（当前判定预填，人工改后即终稿），
C3/C4 否决/误检框作为新对象追加（预填检测器意见，人工删框=确认非桶/误检）。
图源 /var/ias/snapshot/shtm 只读 symlink，绝不改动 snapshot。
"""
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import s1_vlabel_merge as svm  # noqa: E402  复用 iou/poly_to_box/常量，单一数据源

S1 = Path("/home/jiang/cc/py/jxl/projects/shtm/s1")
SNAP = Path("/var/ias/snapshot/shtm")
SRC_PROJECT = Path("/home/jiang/ws/trash/s1_relabel_v1")
PROJECT = Path("/home/jiang/ws/trash/s1_hardcase_review")
MOSAIC = S1.parent / "research" / "2026-09-09-S1难例拼图.jpg"

META_ID = 1032
USER_AGENT = "shtm-s1-hardcase"
NOW = datetime.now().astimezone().isoformat(timespec="seconds")

LOWCONF = 0.6          # C1_lowconf 阈值
C2_CONF = 0.9          # C2 分歧带新值置信上限
C4_SAMPLE = 300        # 空帧误检抽样量
C6_PER_FORM = 30       # 每形态抽样量
MAX_BOXES = 2500
MAX_FRAMES = 1500
SEED = 42

PRIORITY = ["C1_pending", "C1_lowconf", "C2_amount", "C3_veto", "C4_emptyfp",
            "C5_dump", "C6_form"]
ACTION = {"C1_pending": "补属性", "C1_lowconf": "确认", "C2_amount": "确认",
          "C3_veto": "删框", "C4_emptyfp": "删框", "C5_dump": "确认", "C6_form": "确认"}
FORMS = ("bagged", "round", "basket")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def load_preds(name: str) -> dict[str, dict]:
    return {d["rel"]: d["preds"] for d in load_jsonl(S1 / "preds" / name)}


def box_key(box: list[float]) -> str:
    return ",".join(f"{v:.3f}" for v in box[:4])


def poly_box(poly: list) -> list[float]:
    xs = [p["x"] for p in poly]
    ys = [p["y"] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def poly_from_box(box: list[float]) -> list[dict]:
    return [{"x": round(float(box[0]), 6), "y": round(float(box[1]), 6)},
            {"x": round(float(box[2]), 6), "y": round(float(box[1]), 6)},
            {"x": round(float(box[2]), 6), "y": round(float(box[3]), 6)},
            {"x": round(float(box[0]), 6), "y": round(float(box[3]), 6)}]


def iou_match_det(rel: str, det_boxes: list[dict], by_idx: dict[int, dict],
                  m31_objs: list) -> dict[int, int]:
    """复刻 s1_vlabel_merge.det_label 的 IoU 贪心匹配：m31 obj_idx → 项目对象 id。"""
    cands = [(svm.poly_to_box(o["polygon"]), o["prob_class"]["value"], i)
             for i, o in enumerate(m31_objs) if i in by_idx]
    used: set[int] = set()
    out: dict[int, int] = {}
    for bi, b in enumerate(det_boxes):
        cat = svm.CLS2ID[b["cls"]]
        if cat not in svm.ATTRS_BY_CLASS:
            continue
        bb = svm.poly_to_box([{"x": p[0], "y": p[1]} for p in b["polygon"]])
        best, bj = 0.0, -1
        for i, (cb, ccls, _) in enumerate(cands):
            if i in used or ccls != cat:
                continue
            v = svm.iou(bb, cb)
            if v > best:
                best, bj = v, i
        if bj >= 0 and best >= svm.IOU_MATCH:
            used.add(bj)
            out[cands[bj][2]] = bi
    return out


class Selector:
    """六判据收集 → (rel, 项目对象 id) 去重合并。C3/C4 新框对象 id 追加期分配。"""

    def __init__(self) -> None:
        self.det = {d["rel"]: d for d in load_jsonl(S1 / "det_track.jsonl")}
        self.vlabels = {p.stem: json.load(open(p))
                        for p in sorted((SRC_PROJECT / "vlabels").glob("*.json5"))}
        self.cache = {d["key"]: d["verdict"] for d in load_jsonl(S1 / "preds/vlm_cache.jsonl")}
        self.la_det = load_preds("la_det.jsonl")
        self.gd_det = load_preds("gdino_det.jsonl")
        self.la_em = load_preds("la_empty.jsonl")
        self.gd_em = load_preds("gdino_empty.jsonl")
        self.m31: dict[str, list] = {}
        # entries[(rel, obj_id)] = {reasons: set, detail: {code: str}, lowconf: float}
        self.entries: dict[tuple[str, int], dict] = {}
        self.new_objs: dict[str, list[dict]] = defaultdict(list)  # C3/C4 追加框
        self.dropped_unlanded = 0

    def objs31(self, rel: str) -> list:
        if rel not in self.m31:
            self.m31[rel] = json.load(open(f"{SNAP}/{rel}_m31.json")).get("objects", [])
        return self.m31[rel]

    def stem(self, rel: str) -> str:
        return rel.replace("/", "_")

    def vlabel(self, rel: str) -> dict:
        return self.vlabels[self.stem(rel)]

    def add(self, rel: str, obj_id: int, code: str, detail: str,
            lowconf: float | None = None) -> None:
        e = self.entries.setdefault((rel, obj_id), {"reasons": set(), "detail": {},
                                                    "lowconf": 1.0})
        e["reasons"].add(code)
        e["detail"].setdefault(code, detail)
        if lowconf is not None:
            e["lowconf"] = min(e["lowconf"], lowconf)

    def attr_landed_map(self) -> None:
        """C1_lowconf / C2_amount：attr_track → 已落地项目对象。"""
        attr: dict[str, list] = defaultdict(list)
        for r in load_jsonl(S1 / "attr_track.jsonl"):
            attr[r["rel"]].append(r)
        for rel, rows in attr.items():
            by_idx = {r["obj_idx"]: r for r in rows}
            track = self.det.get(rel, {}).get("track")
            if track == "det":  # merged 帧：IoU 匹配 landed
                m2o = iou_match_det(rel, self.det[rel]["boxes"], by_idx, self.objs31(rel))
            elif track is None:  # attr_only 帧：m31 对象序 == 项目对象 id
                m2o = {i: i for i in by_idx}
            else:  # empty 帧 attr 判定未落地（实测为 0，防御性跳过）
                self.dropped_unlanded += len(by_idx)
                continue
            self.dropped_unlanded += len(by_idx) - len(m2o)
            for r in rows:
                oid = m2o.get(r["obj_idx"])
                if oid is None:
                    continue
                lo = min((v["conf"] or 0) for v in r["new"].values())
                if lo <= LOWCONF:
                    weak = [f"{k}={v['value']}@{v['conf']}" for k, v in r["new"].items()
                            if (v["conf"] or 0) <= LOWCONF]
                    self.add(rel, oid, "C1_lowconf", "极低conf: " + ",".join(weak), lo)
                na, oa = r["new"].get("amount"), r.get("old", {}).get("amount")
                if na and oa and na["value"] != oa["value"] and (na["conf"] or 0) < C2_CONF:
                    self.add(rel, oid, "C2_amount",
                             f"amount {oa['value']}→{na['value']}@{na['conf']}")

    def scan_pending_and_dump(self) -> None:
        """C1_pending / C5_dump：直接扫项目 vlabels。"""
        for stem, lab in self.vlabels.items():
            rel = stem.replace("_", "/")
            for o in lab["objects"]:
                if any(p["value"] == svm.PENDING for p in o["properties"]):
                    self.add(rel, o["id"], "C1_pending", "det 终稿框属性未判定(=pending)")
                if o["category"] == svm.CLS2ID["dump"]:
                    self.add(rel, o["id"], "C5_dump", "dump 族弱(VLM 定类样本最少)")

    def det_vetoes(self) -> int:
        """C3_veto：重建 det 候选 → vlm not_bucket 否决框（追加为审核新对象）。"""
        n = 0
        for rel, d in self.det.items():
            if d["track"] != "det":
                continue
            cands = svm_candidates_det(self.la_det.get(rel, {}), self.gd_det.get(rel, {}))
            for c in cands:
                v = self.cache.get(f"det|{rel}|{box_key(c['box'])}")
                if not v or (v["cls"] != "not_bucket" and v["is_bucket"]):
                    continue
                oid = self.append_new(rel, c["box"], c["la_cls"], c["gdino_conf"])
                self.add(rel, oid, "C3_veto",
                         f"la:{c['la_cls']}/gdino:{c['gdino_conf']:.2f} → VLM:not_bucket"
                         f"({v['reason']})")
                n += 1
        return n

    def empty_fps(self) -> int:
        """C4_emptyfp：空帧轨 VLM 判误检簇（含真边界），conf 均匀抽 300。"""
        fps = []
        for rel, d in self.det.items():
            if d["track"] != "empty":
                continue
            for c in svm_clusters_empty(self.la_em.get(rel, {}), self.gd_em.get(rel, {})):
                v = self.cache.get(f"empty|{rel}|{box_key(c['box'])}")
                if v and not v["is_bucket"] and v["cls"] != "crop_error":
                    fps.append((rel, c, v))
        fps.sort(key=lambda t: -t[1]["conf"])  # conf 降序均匀抽样覆盖难度谱
        step = len(fps) / C4_SAMPLE
        picked = [fps[int(i * step)] for i in range(min(C4_SAMPLE, len(fps)))]
        for rel, c, v in picked:
            oid = self.append_new(rel, c["box"], "can", c["conf"])
            self.add(rel, oid, "C4_emptyfp",
                     f"src:{'+'.join(sorted(c['sources']))}@{c['conf']:.2f} → VLM:误检"
                     f"({v['reason']})")
        return len(picked)

    def form_samples(self) -> int:
        """C6_form：项目桶类框按 vlm_cache 形态 → 每形态 conf 最低 30。"""
        name_of = {v: k for k, v in svm.CLS2ID.items()}
        pool: dict[str, list] = defaultdict(list)
        for stem, lab in self.vlabels.items():
            rel = stem.replace("_", "/")
            track = self.det.get(rel, {}).get("track")
            if track not in ("det", "empty"):
                continue
            for o in lab["objects"]:
                if name_of.get(o["category"]) not in BUCKETS:
                    continue
                key = f"{track}|{rel}|{box_key(poly_box(o['polygon']))}"
                v = self.cache.get(key)
                if v and v.get("form") in FORMS:
                    pool[v["form"]].append((o["confidence"], rel, o["id"]))
        n = 0
        for f in FORMS:
            for _, rel, oid in sorted(pool[f])[:C6_PER_FORM]:
                self.add(rel, oid, "C6_form", f"特殊形态 form={f}")
                n += 1
        return n

    def append_new(self, rel: str, box: list[float], cls: str, conf: float) -> int:
        """C3/C4 追加框（预填检测器意见）；返回最终项目对象 id。"""
        cat = svm.CLS2ID[cls]
        props = [{"id": svm.PROP_ID[k], "value": svm.PENDING, "confidence": 1.0}
                 for k in svm.ATTRS_BY_CLASS.get(cat, [])]
        self.new_objs[rel].append({
            "category": cat, "confidence": round(float(conf), 4),
            "polygon": poly_from_box(box), "properties": props,
        })
        return -len(self.new_objs[rel])  # 临时负 id（-k ↔ 索引 k-1），构建项目时重排

    def resolve_new_ids(self) -> None:
        for (rel, oid), e in self.entries.items():
            if oid < 0:
                base = len(self.vlabel(rel)["objects"])
                e["resolved"] = base + (-oid - 1)
            else:
                e["resolved"] = oid


# ---- 40_vote_vlm 候选重建（复制关键纯函数，/tmp 非持久不作 import 源） ----

def flat(preds: dict, classes: tuple) -> list:
    out = [(c, b) for c in classes for b in preds.get(c, [])]
    return sorted(out, key=lambda t: -t[1][4])


def svm_candidates_det(la: dict, gdino: dict) -> list:
    """la 锚框 ∧ gdino 任一桶类框（与 /tmp/shtm_s1/40_vote_vlm.det_bucket_candidates 等价）。"""
    anchors, gboxes = flat(la, BUCKETS), flat(gdino, BUCKETS)
    cands = []
    for a_cls, ab in anchors:
        best_j, best_v = -1, 0.0
        for j, (_, gb) in enumerate(gboxes):
            v = svm.iou(ab, gb)
            if v >= 0.4 and v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            cands.append({"box": ab[:4], "la_cls": a_cls,
                          "gdino_conf": gboxes[best_j][1][4]})
    return cands


def svm_clusters_empty(la: dict, gdino: dict) -> list:
    """la ∪ gdino IoU>=0.5 聚类（与 40_vote_vlm.empty_candidates 等价）。"""
    items = [("la", c, b) for c, b in flat(la, BUCKETS)] + \
            [("gdino", c, b) for c, b in flat(gdino, BUCKETS)]
    items.sort(key=lambda t: (0 if t[0] == "la" else 1, -t[2][4]))
    clusters = []
    for src, cls, b in items:
        for cl in clusters:
            if svm.iou(b, cl["box"]) >= 0.5:
                cl["sources"].add(src), cl["src_cls"].add(cls)
                cl["conf"] = max(cl["conf"], b[4])
                break
        else:
            clusters.append({"box": b[:4], "sources": {src}, "src_cls": {cls},
                             "conf": b[4]})
    return clusters


BUCKETS = ("opening", "lid", "dump", "can")


def cap_frames(sel: Selector) -> set[str]:
    """帧预算：非 C1_lowconf 判据全保留，C1_lowconf 专属帧按帧内最高 conf 升序裁。"""
    firm: set[str] = set()
    weak: dict[str, float] = {}
    for (rel, _), e in sel.entries.items():
        if e["reasons"] - {"C1_lowconf"}:
            firm.add(rel)
        elif rel not in firm:
            weak[rel] = max(weak.get(rel, 0.0), e["lowconf"])
    room = MAX_FRAMES - len(firm)
    if room < 0:
        raise SystemExit(f"firm 判据帧 {len(firm)} 已超预算 {MAX_FRAMES}")
    if len(weak) <= room:
        return firm | set(weak)
    keep = [rel for _, rel in sorted((c, r) for r, c in weak.items())[:room]]
    return firm | set(keep)


def build_project(sel: Selector, frames: set[str]) -> list[dict]:
    """meta + vlabels(复制+追加) + images symlink。返回 hardcase 条目终稿。"""
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    (PROJECT / "vlabels").mkdir(parents=True)
    (PROJECT / "images").mkdir()
    meta = json.load(open(SRC_PROJECT / "meta.json5"))
    meta["id"], meta["name"] = META_ID, "shtm-s1-hardcase-review"
    meta["description"] = ("SHTM S1 难例人工兜底: C1 pending/极低conf属性, C2 amount分歧带, "
                           "C3 VLM否决框, C4 空帧误检框(抽300), C5 dump全量, C6 特殊形态; "
                           "vlabels 预填当前判定, 人工改后即终稿(拷回 s1_relabel_v1)")
    (PROJECT / "meta.json5").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    name_of = {v: k for k, v in svm.CLS2ID.items()}
    hardcases = []
    for (rel, oid), e in sorted(sel.entries.items()):
        if rel not in frames:
            continue
        reasons = sorted(e["reasons"], key=PRIORITY.index)
        if oid >= 0:
            cat = sel.vlabel(rel)["objects"][oid]["category"]
        else:  # C3/C4 追加框：检测器预填类
            cat = sel.new_objs[rel][-oid - 1]["category"]
        hardcases.append({"rel": rel, "obj_idx": e["resolved"], "cls": name_of[cat],
                          "reasons": reasons, "action": ACTION[reasons[0]],
                          "detail": e["detail"]})
    n_boxes = len(hardcases)
    if n_boxes > MAX_BOXES:
        raise SystemExit(f"hardcase 框 {n_boxes} 超预算 {MAX_BOXES}")

    for rel in sorted(frames):
        stem = sel.stem(rel)
        lab = json.loads(json.dumps(sel.vlabel(rel)))  # deep copy
        lab["user_agent"], lab["last_modified"] = USER_AGENT, NOW
        base = len(lab["objects"])
        for i, o in enumerate(sel.new_objs.get(rel, [])):
            lab["objects"].append({"id": base + i, **o})
        (PROJECT / "vlabels" / f"{stem}.json5").write_text(
            json.dumps(lab, ensure_ascii=False, indent=1) + "\n")
        src = SNAP / f"{rel}.jpg"
        assert os.path.exists(src), f"missing image {src}"
        os.symlink(src, PROJECT / "images" / f"{stem}.jpg")
    return hardcases


def build_mosaic(sel: Selector, hardcases: list[dict]) -> None:
    """40 格拼图：按判据比例分层抽样，红框+标签。"""
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(SEED)
    by_code: dict[str, list[dict]] = defaultdict(list)
    for h in hardcases:
        by_code[h["reasons"][0]].append(h)
    picked: list[dict] = []
    for code in PRIORITY:
        picked += rng.sample(by_code[code], min(4, len(by_code[code])))
    picked_keys = {(h["rel"], h["obj_idx"]) for h in picked}
    rng.shuffle(hardcases)
    for h in hardcases:
        if len(picked) >= 40:
            break
        if (h["rel"], h["obj_idx"]) not in picked_keys:
            picked.append(h)
            picked_keys.add((h["rel"], h["obj_idx"]))
    picked = picked[:40]

    CELL_W, CELL_H, COLS = 320, 240, 8
    grid = Image.new("RGB", (CELL_W * COLS, CELL_H * ((len(picked) + COLS - 1) // COLS)),
                     (24, 24, 24))
    try:
        font = ImageFont.load_default(16)
    except TypeError:  # Pillow < 10.1 无 size 参数
        font = ImageFont.load_default()
    for i, h in enumerate(picked):
        lab = json.load(open(PROJECT / "vlabels" / f"{sel.stem(h['rel'])}.json5"))
        obj = lab["objects"][h["obj_idx"]]
        im = Image.open(SNAP / f"{h['rel']}.jpg").convert("RGB")
        w, hh = im.size
        bx = poly_box(obj["polygon"])
        bw, bh = (bx[2] - bx[0]) * w, (bx[3] - bx[1]) * hh
        x1 = max(0, int(bx[0] * w - bw * 0.25))
        y1 = max(0, int(bx[1] * hh - bh * 0.25))
        x2 = min(w, int(bx[2] * w + bw * 0.25))
        y2 = min(hh, int(bx[3] * hh + bh * 0.25))
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        crop = im.crop((x1, y1, x2, y2))
        crop.thumbnail((CELL_W, CELL_H - 22))
        d = ImageDraw.Draw(crop)
        lw = max(2, int(min(crop.size) * 0.01))
        d.rectangle([bx[0] * w - x1, bx[1] * hh - y1, bx[2] * w - x1, bx[3] * hh - y1],
                    outline=(255, 0, 0), width=lw)
        cell = Image.new("RGB", (CELL_W, CELL_H), (24, 24, 24))
        cell.paste(crop, ((CELL_W - crop.width) // 2, 22))
        cd = ImageDraw.Draw(cell)
        cd.text((4, 3), f"{h['reasons'][0]}|{h['cls']}|{h['action']}",
                fill=(255, 220, 80), font=font)
        grid.paste(cell, ((i % COLS) * CELL_W, (i // COLS) * CELL_H))
    grid.save(MOSAIC, quality=88)
    print(f"mosaic: {MOSAIC} ({len(picked)} cells)")


def main() -> None:
    sel = Selector()
    sel.scan_pending_and_dump()
    sel.attr_landed_map()
    n3 = sel.det_vetoes()
    n4 = sel.empty_fps()
    n6 = sel.form_samples()
    sel.resolve_new_ids()
    frames = cap_frames(sel)
    hardcases = build_project(sel, frames)
    with open(S1 / "hardcase_list.jsonl", "w") as f:
        for h in hardcases:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    build_mosaic(sel, hardcases)

    ct = Counter(h["reasons"][0] for h in hardcases)
    multi = sum(1 for h in hardcases if len(h["reasons"]) > 1)
    print(f"hardcase boxes: {len(hardcases)} / {MAX_BOXES}, frames: {len(frames)} / {MAX_FRAMES}")
    print("by primary reason:", dict(ct))
    print(f"multi-reason objs: {multi}; C3 veto={n3}, C4 sampled={n4}, C6 form={n6}; "
          f"attr unlanded dropped={sel.dropped_unlanded}")


if __name__ == "__main__":
    main()
