#!/usr/bin/env python3
"""S1 双轨产物合并 → VLabel 重标项目（s1_relabel_v1）。

合并规则（2026-09-09 用户裁决）：
  det 轨帧    = 检测重标终稿（框+类取 det_track；person=yoloe∧rfdetr 双票，桶类=la∧gdino→VLM 定类）
  attr 轨帧   = 沿用 _m31.json 原框 + attr_track 新属性（spark VLM 重判）
  交集帧      = 框取 det 终稿；opening/lid 框与 m31 原框 IoU≥0.4 贪心匹配挂 attr 新属性，
                未匹配框的绑定属性标 pending(-1)；被 det 终稿丢弃的原框属性判定随之作废
  empty 轨帧  = VLM 判真桶的框入项目（旧模型判空的负样本/漏检修复）

产物：/home/jiang/ws/trash/s1_relabel_v1/{meta.json5, vlabels/*.json5, images/*.jpg(symlink)}
格式：VLabel Rust v2.0（category+confidence 扁平、PropertyEntry{id,value,confidence}），
      schema 见 vlabel-core label.rs/label_meta.rs；属性引用 id，side 由 m31 重复 id 4 修正为 5。
图源 /var/ias/snapshot/shtm/<rel>.jpg 只读 symlink，绝不改动 snapshot。
"""
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

S1 = Path("/home/jiang/cc/py/jxl/projects/shtm/s1")
SNAP = Path("/var/ias/snapshot/shtm")
PROJECT = Path("/home/jiang/ws/trash/s1_relabel_v1")

META_ID = 1031
USER_AGENT = "shtm-s1-merge"
IOU_MATCH = 0.4
NOW = datetime.now().astimezone().isoformat(timespec="seconds")

CLS2ID = {"opening": 0, "lid": 1, "dump": 2, "person": 3, "can": 4}
PROP_ID = {"sort": 1, "amount": 2, "direction": 3, "illegal": 4, "side": 5}  # side: m31 id4 冲突→5
ATTRS_BY_CLASS = {0: ["sort", "amount", "illegal"], 1: ["sort", "side"]}
PENDING = -1


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def poly_to_box(pts: list[dict]) -> list[float]:
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def rd(v: float) -> float:
    return round(float(v), 6)


def write_meta() -> None:
    # m31 类别/属性定义 → VLabel CatDef/PropDef（hotkey=m31 keys；side id 修正）
    meta = {
        "id": META_ID,
        "name": "shtm-s1-relabel-v1",
        "description": "SHTM S1 双轨重标: det=共识检测终稿, attr=VLM 属性重判, "
                       "empty=旧判空帧VLM复检; 人工审核后作重训基准",
        "shape": {"title_style": 1, "thickness": 2, "auto_save": True, "vertex_radius": 10},
        "roi": {"color": "#800080"},
        "categories": [
            {"id": 0, "name": "opening", "description": "垃圾桶开口", "hotkey": "1",
             "color": "#FF0000", "properties": [
                {"id": PROP_ID["sort"], "name": "sort", "type": "sort"},
                {"id": PROP_ID["amount"], "name": "amount", "type": "amount"},
                {"id": PROP_ID["illegal"], "name": "illegal", "type": "illegal"}]},
            {"id": 1, "name": "lid", "description": "垃圾桶盖", "hotkey": "2",
             "color": "#FFA500", "properties": [
                {"id": PROP_ID["sort"], "name": "sort", "type": "sort"},
                {"id": PROP_ID["side"], "name": "side", "type": "side"}]},
            {"id": 2, "name": "dump", "description": "垃圾堆", "hotkey": "3", "color": "#FFFF00"},
            {"id": 3, "name": "person", "description": "人员", "hotkey": "4", "color": "#00FF00"},
            {"id": 4, "name": "can", "description": "整个垃圾桶", "hotkey": "5",
             "color": "#0000FF", "properties": [
                {"id": PROP_ID["direction"], "name": "direction", "type": "direction"}]},
        ],
        "property_types": [
            {"id": 1, "name": "sort", "description": "垃圾桶/盖类型", "values": [
                {"id": 0, "name": "residuil", "description": "干/其他垃圾/黑/黄", "hotkey": "1",
                 "color": "#201080", "sign": "D"},
                {"id": 1, "name": "food", "description": "湿/厨余/棕/绿", "hotkey": "2",
                 "color": "#A52A2A", "sign": "W"},
                {"id": 2, "name": "recyclable", "description": "蓝/可回收", "hotkey": "3",
                 "color": "#0000FF", "sign": "R"},
                {"id": 3, "name": "hazardous", "description": "红/有害", "hotkey": "4",
                 "color": "#FFA500", "sign": "H"}]},
            {"id": 2, "name": "amount", "description": "垃圾桶内垃圾量", "values": [
                {"id": 0, "name": "0/4", "description": "空桶", "hotkey": "1",
                 "color": "#FF0000", "sign": "0"},
                {"id": 1, "name": "not_full", "description": "不满", "hotkey": "2",
                 "color": "#FFA500", "sign": "1"},
                {"id": 2, "name": "soon_full", "description": "将满", "hotkey": "3",
                 "color": "#FFFF00", "sign": "2"},
                {"id": 3, "name": "full", "description": "已满", "hotkey": "4",
                 "color": "#00FF00", "sign": "3"}]},
            {"id": 3, "name": "direction", "description": "垃圾桶朝向", "values": [
                {"id": 0, "name": "front", "description": "正面", "hotkey": "1",
                 "color": "#FF0000", "sign": "FR"},
                {"id": 1, "name": "non_front", "description": "非正面", "hotkey": "2",
                 "color": "#FFA500", "sign": "NF"}]},
            {"id": 4, "name": "illegal", "description": "是否存在违规投放垃圾", "values": [
                {"id": 0, "name": "no", "description": "无违规投放垃圾", "hotkey": "1",
                 "color": "#FF0000", "sign": "n"},
                {"id": 1, "name": "yes", "description": "有违规投放垃圾", "hotkey": "2",
                 "color": "#FF0000", "sign": "y"}]},
            {"id": 5, "name": "side", "description": "垃圾箱的正/反面分类", "values": [
                {"id": 0, "name": "front", "description": "垃圾桶盖正面", "hotkey": "1",
                 "color": "#00FF00", "sign": "F"},
                {"id": 1, "name": "back", "description": "垃圾桶盖反面", "hotkey": "2",
                 "color": "#FF0000", "sign": "B"}]},
        ],
        "property_special_values": [
            {"id": -3, "name": "error", "description": "检测器错误", "hotkey": "r",
             "color": "#FFC0CB", "sign": "E"},
            {"id": -2, "name": "exclude", "description": "排除争议样本", "hotkey": "e",
             "color": "#800080", "sign": "X"},
            {"id": PENDING, "name": "pending", "description": "待定", "hotkey": "q",
             "color": "#FFFFFF", "sign": "P"},
        ],
    }
    (PROJECT / "meta.json5").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def new_label() -> dict:
    return {"version": "2.0", "user_agent": USER_AGENT, "created_at": NOW,
            "last_modified": NOW, "rois": [], "objects": []}


def m31_label(rel: str, attr_by_idx: dict[int, dict]) -> tuple[dict, int]:
    """m31 原框全量对象；被重判对象属性替换为 VLM 新值。返回 (label, 重判对象数)。"""
    d = json.load(open(f"{SNAP}/{rel}_m31.json"))
    lab = new_label()
    roi = (d.get("sensor", {}).get("params") or {}).get("roi")
    if roi:
        lab["rois"] = [[{"x": rd(p["x"]), "y": rd(p["y"])} for p in roi]]
    judged = 0
    for idx, o in enumerate(d.get("objects", [])):
        props = []
        if idx in attr_by_idx:
            judged += 1
            for k, v in attr_by_idx[idx].items():
                props.append({"id": PROP_ID[k], "value": v["value"],
                              "confidence": v["conf"] if v["conf"] is not None else 0.0})
        lab["objects"].append({
            "id": len(lab["objects"]), "category": o["prob_class"]["value"],
            "confidence": rd(o["prob_class"]["confidence"]),
            "polygon": [{"x": rd(p["x"]), "y": rd(p["y"])} for p in o["polygon"]],
            "properties": props,
        })
    return lab, judged


def det_label(rel: str, boxes: list[dict], attr_by_idx: dict[int, dict],
              d31_objs: list) -> tuple[dict, dict]:
    """det 终稿框；交集帧 opening/lid 与 m31 原框 IoU 匹配挂属性，未匹配标 pending。"""
    lab = new_label()
    d31 = json.load(open(f"{SNAP}/{rel}_m31.json"))
    roi = (d31.get("sensor", {}).get("params") or {}).get("roi")
    if roi:
        lab["rois"] = [[{"x": rd(p["x"]), "y": rd(p["y"])} for p in roi]]
    cands = []
    if attr_by_idx:
        cands = [(poly_to_box(o["polygon"]), d31["objects"][i]["prob_class"]["value"], i)
                 for i, o in enumerate(d31_objs) if i in attr_by_idx]
    st = Counter()
    used: set[int] = set()
    for b in boxes:
        props = []
        cat = CLS2ID[b["cls"]]
        if attr_by_idx and cat in ATTRS_BY_CLASS:
            bb = poly_to_box([{"x": p[0], "y": p[1]} for p in b["polygon"]])
            best, bi = 0.0, -1
            for i, (cb, ccls, orig_idx) in enumerate(cands):
                if i in used or ccls != cat:
                    continue
                v = iou(bb, cb)
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= IOU_MATCH:
                used.add(bi)
                st["attr_matched"] += 1
                for k, v in attr_by_idx[cands[bi][2]].items():
                    props.append({"id": PROP_ID[k], "value": v["value"],
                                  "confidence": v["conf"] if v["conf"] is not None else 0.0})
            else:
                st["attr_pending"] += 1
                for k in ATTRS_BY_CLASS[cat]:
                    props.append({"id": PROP_ID[k], "value": PENDING, "confidence": 1.0})
        lab["objects"].append({
            "id": len(lab["objects"]), "category": cat, "confidence": rd(b["conf"]),
            "polygon": [{"x": rd(p[0]), "y": rd(p[1])} for p in b["polygon"]],
            "properties": props,
        })
    st["attr_dropped"] = len(cands) - len(used)
    return lab, st


def main() -> None:
    attr: dict[str, list] = defaultdict(list)
    for l in open(S1 / "attr_track.jsonl"):
        r = json.loads(l)
        if "error" not in r:
            attr[r["rel"]].append(r)
    det = {}
    for l in open(S1 / "det_track.jsonl"):
        d = json.loads(l)
        det[d["rel"]] = d
    det_rels = {r for r, d in det.items() if d["track"] == "det"}
    empty_rels = {r for r, d in det.items() if d["track"] == "empty"}

    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    (PROJECT / "vlabels").mkdir(parents=True)
    (PROJECT / "images").mkdir()
    write_meta()

    stats = defaultdict(Counter)
    seen_stems: set[str] = set()
    m31_cache: dict[str, list] = {}

    def objs31(rel: str) -> list:
        if rel not in m31_cache:
            m31_cache[rel] = json.load(open(f"{SNAP}/{rel}_m31.json")).get("objects", [])
        return m31_cache[rel]

    for rel in sorted(set(det) | set(attr)):
        stem = rel.replace("/", "_")
        assert stem not in seen_stems, f"stem collision: {stem}"
        seen_stems.add(stem)
        src = f"{SNAP}/{rel}.jpg"
        assert os.path.exists(src), f"missing image {src}"
        os.symlink(src, PROJECT / "images" / f"{stem}.jpg")
        by_idx = {r["obj_idx"]: r["new"] for r in attr.get(rel, [])}

        if rel in det_rels:
            group = "merged" if by_idx else "det_only"
            lab, st = det_label(rel, det[rel]["boxes"], by_idx, objs31(rel))
        elif rel in empty_rels:
            group = "empty"
            d31p = f"{SNAP}/{rel}_m31.json"
            lab = new_label()
            if os.path.exists(d31p):
                roi = (json.load(open(d31p)).get("sensor", {}).get("params") or {}).get("roi")
                if roi:
                    lab["rois"] = [[{"x": rd(p["x"]), "y": rd(p["y"])} for p in roi]]
            for b in det[rel]["boxes"]:
                lab["objects"].append({
                    "id": len(lab["objects"]), "category": CLS2ID[b["cls"]],
                    "confidence": rd(b["conf"]),
                    "polygon": [{"x": rd(p[0]), "y": rd(p[1])} for p in b["polygon"]],
                    "properties": [],
                })
            st = Counter()
        else:
            group = "attr_only"
            lab, judged = m31_label(rel, by_idx)
            st = Counter(judged=judged)

        stats[group]["frames"] += 1
        stats[group]["boxes"] += len(lab["objects"])
        stats[group]["neg_frames"] += (len(lab["objects"]) == 0)
        stats[group]["attr_objs"] += sum(1 for o in lab["objects"] if o["properties"])
        for o in lab["objects"]:
            stats["cls"][o["category"]] += 1
            if any(p["value"] == PENDING for p in o["properties"]):
                stats[group]["pending_objs"] += 1
        for k, v in st.items():
            stats[group][k] += v
        (PROJECT / "vlabels" / f"{stem}.json5").write_text(
            json.dumps(lab, ensure_ascii=False, indent=1) + "\n")

    print(f"project: {PROJECT}")
    for g in ("merged", "det_only", "empty", "attr_only"):
        print(g, dict(stats[g]))
    print("cls dist:", dict(stats["cls"]))
    landed = stats["merged"]["attr_matched"] + stats["attr_only"]["judged"]
    print(f"attr judgments: input=8383 landed={landed} "
          f"dropped(det终稿弃框)={stats['merged']['attr_dropped']}")


if __name__ == "__main__":
    main()
