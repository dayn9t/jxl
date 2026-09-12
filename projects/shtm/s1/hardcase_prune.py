#!/usr/bin/env python3
"""S1 难例自动削减（2026-09-10）：双 VLM 一致闸门 + 隔离池，压缩人工审核面。

动机（用户）：SHTM S1 难例人工审核量太大（1,957 hardcase + 1,528 backlog 候选），
用本地 spark qwen3.5-35b + 商用 doubao-lite 双模型独立判定，逐字段一致才自动判定，
不一致/不确定的一律留给人工——只削减"无聊的"，不放过"真难的"。

隔离原则（用户裁决 2026-09-10）：自动判定部分处于"没那么确定"状态，一律进隔离池
s1_hardcase_auto/，绝不直接混入正式样本 s1_relabel_v1；晋升需用户过目自动删框拼图
（否决误删）后单独执行。本脚本对 s1_relabel_v1 零写入。

统一问题（两模型同 prompt）：对 crop 判 is_bucket/cls/form/attrs。
闸门（consensus-labeling k>=2 精神，双票全一致）：
  任一 error/crop_error        -> HUMAN(model_error)
  is_bucket 不一致             -> HUMAN(bucket_conflict)
  双判非桶: C3/C4->AUTO_DELETE  backlog->AUTO_REJECT  其余->HUMAN(both_non_target)
  双判真桶 cls 异 -> HUMAN(cls_conflict)
  双判真桶 cls 同: 该类全部适用属性均有共识 -> 与预填类同 AUTO_CONFIRM / 异 AUTO_RECLASS
                   属性有缺 -> HUMAN(attrs_unresolved)
去向：
  s1_hardcase_review/   原审核项目就地应用自动判定（先备份 vlabels_bak_<date>）——工作台
  s1_hardcase_auto/     隔离池(id 1034)：纯自动帧 + backlog 全自动帧(接受框/负样本帧)
  s1_hardcase_review_r2/(id 1033) 残留人工帧（含同帧内已自动判定的对象作上下文）
审计：s1/hardcase_auto.jsonl 全量判定留痕（双模型原始 verdict + polygon，可恢复误删）；
      s1/hardcase_residual_list.jsonl 残留清单（r2 新 obj id 对齐）；
      research/2026-09-10-SHTM自动删框拼图-*.jpg 自动删框全量拼图供否决。

用法：uv run --project /home/jiang/cc/py/jxl python hardcase_prune.py [--pilot 60]
"""
import argparse
import asyncio
import base64
import io
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

sys_parent = Path(__file__).parent
import s1_vlabel_merge as svm  # noqa: E402  CLS2ID/PROP_ID/ATTRS_BY_CLASS/poly_to_box 单一数据源

S1 = Path("/home/jiang/cc/py/jxl/projects/shtm/s1")
SNAP = Path("/var/ias/snapshot/shtm")
REVIEW = Path("/home/jiang/ws/trash/s1_hardcase_review")
R2 = Path("/home/jiang/ws/trash/s1_hardcase_review_r2")
AUTO = Path("/home/jiang/ws/trash/s1_hardcase_auto")
LEDGER = S1 / "hardcase_auto.jsonl"
RESIDUAL = S1 / "hardcase_residual_list.jsonl"
CACHE = S1 / "preds/vlm_cache_hp.jsonl"

ID2CLS = {v: k for k, v in svm.CLS2ID.items()}
SPARK_URL = "http://192.168.18.182:8000/v1/chat/completions"
SPARK_MODEL = "qwen3.5-35b-a3b-fp8"
DOUBAO_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DOUBAO_MODEL = "doubao-seed-2-0-lite-260215"
SEM_SPARK, SEM_DOUBAO = 32, 24
MARGIN = 0.15
AUTO_CONF = 0.9

PROMPT = """你是垃圾桶监控标注审核员。判断图中红框内的目标主体，仅输出一个JSON对象：
{"is_bucket":true,"cls":"can|opening|lid|dump|person|none","form":"standard|round|bagged|basket|none","attrs":{"sort":null,"amount":null,"illegal":null,"direction":null,"side":null},"reason":"20字内"}
判类规则：框内是完整桶体或大部桶身→can；仅桶口面特写(桶身大部在框外)→opening；桶盖本体(闭合/翻开/脱离桶体)→lid；脱离容器的散装垃圾堆→dump；现场真人→person；误检(路面/墙体/阴影/车辆/杂物)→is_bucket=false且cls=none。
补充：套袋桶袋口开面视同opening(form=bagged)；圆桶form=round；篮筐当桶用算bucket(form=basket)。
attrs判不准填null禁止猜：opening给sort(0干1湿2可回收3有害)+amount(0空1不满2将满3满)+illegal(0无1有)；lid给sort+side(0正1反)；can给direction(0正面1非正面)；其余null。"""

PV = "v3"  # prompt 版本，入缓存键：改 prompt 必须 bump，否则旧判定污染

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_CLS = {"opening", "lid", "can", "dump", "person", "none"}
VALID_FORM = {"standard", "round", "bagged", "basket", "none"}
VALID_ATTR = {"sort": set(range(4)), "amount": set(range(4)), "illegal": {0, 1},
              "direction": {0, 1}, "side": {0, 1}}


def parse_verdict(content: str) -> dict | None:
    cleaned = _THINK_RE.sub("", content)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    m = _OBJ_RE.search(cleaned)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    cls = str(d.get("cls", "none"))
    if cls not in VALID_CLS:
        return None
    form = str(d.get("form", "none"))
    attrs: dict[str, int | None] = {}
    raw = d.get("attrs") or {}
    for k, ok in VALID_ATTR.items():
        v = raw.get(k)
        attrs[k] = v if isinstance(v, int) and v in ok else None
    return {"is_bucket": bool(d.get("is_bucket")), "cls": cls,
            "form": form if form in VALID_FORM else "none",
            "attrs": attrs, "reason": str(d.get("reason", ""))[:40]}


def load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["key"]] = d["verdict"]
    return out


_CACHE: dict[str, dict] = {}


def latest_backup() -> Path | None:
    baks = sorted(REVIEW.glob("vlabels_bak_*"))
    return baks[-1] if baks else None


def src_vlabels() -> Path:
    """vlabel 读取源：存在备份（已 apply 过）则读最新备份，保证跨天重跑幂等。"""
    bak = latest_backup()
    return bak if bak else REVIEW / "vlabels"


def load_items() -> list[dict]:
    items: list[dict] = []
    hc = [json.loads(l) for l in (S1 / "hardcase_list.jsonl").open()]
    vlab_cache: dict[str, dict] = {}
    for r in hc:
        stem = r["rel"].replace("/", "_")
        if stem not in vlab_cache:
            vlab_cache[stem] = json.load(open(src_vlabels() / f"{stem}.json5"))
        o = vlab_cache[stem]["objects"][r["obj_idx"]]
        items.append({"kind": "hardcase", "rel": r["rel"], "idx": r["obj_idx"],
                      "reasons": r["reasons"], "action": r["action"],
                      "prefill": ID2CLS[o["category"]],
                      "poly": [[p["x"], p["y"]] for p in o["polygon"]]})
    for line in (S1 / "det_track_backlog.jsonl").open():
        d = json.loads(line)
        for bi, b in enumerate(d["boxes"]):
            items.append({"kind": "backlog", "rel": d["rel"], "idx": bi,
                          "reasons": ["BL_real"], "action": "确认",
                          "prefill": b["cls"], "conf": b["conf"],
                          "poly": b["polygon"]})
    return items


def make_crop(rel: str, poly: list) -> str | None:
    box = svm.poly_to_box([{"x": p[0], "y": p[1]} for p in poly])
    try:
        im = Image.open(SNAP / f"{rel}.jpg").convert("RGB")
    except OSError:
        return None
    w, h = im.size
    bw, bh = (box[2] - box[0]) * w, (box[3] - box[1]) * h
    mx, my = bw * MARGIN, bh * MARGIN
    x1, y1 = max(0, int(box[0] * w - mx)), max(0, int(box[1] * h - my))
    x2, y2 = min(w, int(box[2] * w + mx)), min(h, int(box[3] * h + my))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = im.crop((x1, y1, x2, y2))
    scale = min(1.0, 640 / max(crop.size))
    if scale < 1.0:
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
    d = ImageDraw.Draw(crop)
    lw = max(2, int(min(crop.size) * 0.008))
    d.rectangle([box[0] * w - x1, box[1] * h - y1, box[2] * w - x1, box[3] * h - y1],
                outline=(255, 0, 0), width=lw)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def vlm_call(client: httpx.AsyncClient, sem: asyncio.Semaphore, tag: str,
                   item: dict, url: str, model: str, headers: dict) -> tuple[str, dict]:
    key = f"hp{PV}|{tag}|{item['rel']}|{item['idx']}"
    if key in _CACHE:
        return key, _CACHE[key]
    data_url = make_crop(item["rel"], item["poly"])
    if data_url is None:
        return key, {"is_bucket": False, "cls": "crop_error", "form": "none",
                     "attrs": {}, "reason": ""}
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": PROMPT}]}],
        "temperature": 0.0 if tag == "spark" else 0.1, "max_tokens": 300,
        "response_format": {"type": "json_object"}}
    if tag == "spark":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    last = ""
    to = 45.0 if tag == "spark" else 120.0  # spark 健康时延<3s，挂死时快速失败走重试
    for attempt in range(3):
        try:
            async with sem:
                r = await client.post(url, json=payload, headers=headers, timeout=to)
            r.raise_for_status()
            v = parse_verdict(r.json()["choices"][0]["message"]["content"])
            if v is not None:
                return key, v
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            last = str(e)[:60]
            await asyncio.sleep(2 * (attempt + 1))
    return key, {"is_bucket": False, "cls": "vlm_error", "form": "none",
                 "attrs": {}, "reason": f"retries exhausted {last}"}


async def scan(items: list[dict], only: str | None = None) -> None:
    sem_s, sem_d = asyncio.Semaphore(SEM_SPARK), asyncio.Semaphore(SEM_DOUBAO)
    hdr_d = {"Authorization": f"Bearer {os.environ['S4_DOUBAO_API_KEY']}"}
    out_f = CACHE.open("a")
    done = 0
    for s in range(0, len(items), 256):
        chunk = items[s:s + 256]
        async with httpx.AsyncClient() as client:
            tasks = [vlm_call(client, sem_s, "spark", it, SPARK_URL, SPARK_MODEL, {})
                     for it in chunk if only is None or only == "spark"]
            tasks += [vlm_call(client, sem_d, "doubao", it, DOUBAO_URL, DOUBAO_MODEL, hdr_d)
                      for it in chunk if only is None or only == "doubao"]
            results = await asyncio.gather(*tasks)
        for k, v in results:
            if _CACHE.get(k) != v:
                _CACHE[k] = v
                out_f.write(json.dumps({"key": k, "verdict": v}, ensure_ascii=False) + "\n")
        out_f.flush()
        done += len(chunk)
        print(f"scan {done}/{len(items)}", flush=True)
    out_f.close()


def gate(it: dict, s: dict, d: dict) -> dict:
    errs = [t["cls"] for t in (s, d) if t["cls"] in ("vlm_error", "crop_error")]
    if errs:
        return {"disp": "HUMAN", "why": f"model_error:{','.join(errs)}"}
    if s["is_bucket"] != d["is_bucket"]:
        return {"disp": "HUMAN", "why": "bucket_conflict"}
    if not s["is_bucket"]:
        if it["kind"] == "backlog":
            return {"disp": "AUTO_REJECT", "why": f"both_fp:{s['reason']}/{d['reason']}"}
        return {"disp": "AUTO_DELETE", "why": f"both_fp:{s['reason']}/{d['reason']}"}
    if s["cls"] != d["cls"]:
        return {"disp": "HUMAN", "why": f"cls_conflict:{s['cls']}/{d['cls']}"}
    cls = s["cls"]
    need = svm.ATTRS_BY_CLASS.get(svm.CLS2ID[cls], [])
    cons: dict[str, int] = {}
    for k in need:
        sv, dv = s["attrs"].get(k), d["attrs"].get(k)
        if sv is None or dv is None or sv != dv:
            return {"disp": "HUMAN", "why": f"attrs_unresolved:{k}"}
        cons[k] = sv
    disp = "AUTO_CONFIRM" if cls == it["prefill"] else "AUTO_RECLASS"
    return {"disp": disp, "why": f"agree:{cls}", "cls": cls, "attrs": cons,
            "form": s["form"] if s["form"] != "none" else d["form"]}


def apply_review(items: list[dict], gated: list[dict]) -> dict[str, dict[int, int]]:
    """就地应用自动判定到 review vlabels；返回各帧 old_id->new_id 映射。"""
    if latest_backup() is None:
        shutil.copytree(REVIEW / "vlabels",
                        REVIEW / f"vlabels_bak_{date.today():%Y%m%d}")
    by_frame: dict[str, dict[int, dict]] = defaultdict(dict)
    for it, g in zip(items, gated):
        if it["kind"] == "hardcase" and g["disp"].startswith("AUTO"):
            by_frame[it["rel"]][it["idx"]] = g
    idmap: dict[str, dict[int, int]] = {}
    for rel, patch in by_frame.items():
        stem = rel.replace("/", "_")
        p = REVIEW / "vlabels" / f"{stem}.json5"
        lab = json.load(open(src_vlabels() / f"{stem}.json5"))
        kept = [o for o in lab["objects"]
                if not (o["id"] in patch and patch[o["id"]]["disp"] == "AUTO_DELETE")]
        idmap[rel] = {o["id"]: i for i, o in enumerate(kept)}
        for o in kept:
            g = patch.get(o["id"])
            if g is None:
                continue
            o["category"] = svm.CLS2ID[g["cls"]]
            want = svm.ATTRS_BY_CLASS.get(o["category"], [])
            drop = {svm.PROP_ID[k] for k in want}
            o["properties"] = [q for q in o["properties"] if q["id"] not in drop] + [
                {"id": svm.PROP_ID[k], "value": v, "confidence": AUTO_CONF}
                for k, v in g["attrs"].items()]
        for i, o in enumerate(kept):
            o["id"] = i
        lab["objects"] = kept
        p.write_text(json.dumps(lab, ensure_ascii=False, indent=1) + "\n")
    return idmap


def empty_vlabel(rel: str, objs: list[dict]) -> dict:
    lab = svm.new_label()
    m31 = SNAP / f"{rel}_m31.json"
    if m31.exists():
        roi = (json.load(open(m31)).get("sensor", {}).get("params") or {}).get("roi")
        if roi:
            lab["rois"] = [[{"x": svm.rd(p["x"]), "y": svm.rd(p["y"])} for p in roi]]
    lab["objects"] = objs
    return lab


def props_of(cls_id: int, attrs: dict[str, int], pending: bool) -> list[dict]:
    out = []
    for k in svm.ATTRS_BY_CLASS.get(cls_id, []):
        out.append({"id": svm.PROP_ID[k],
                    "value": svm.PENDING if pending else attrs.get(k, svm.PENDING),
                    "confidence": 1.0 if pending else AUTO_CONF})
    return out


def write_meta(dst: Path, meta_id: int, name: str, desc: str) -> None:
    meta = (REVIEW / "meta.json5").read_text()
    meta = re.sub(r"id: \d+", f"id: {meta_id}", meta, count=1)
    meta = re.sub(r'name: "[^"]*"', f'name: "{name}"', meta, count=1)
    meta = re.sub(r'description: "[^"]*"', f'description: "{desc}"', meta, count=1)
    (dst / "meta.json5").write_text(meta)


def poly_pts(poly: list) -> list[dict]:
    return [{"x": svm.rd(p[0]), "y": svm.rd(p[1])} for p in poly]


def build_quarantine(items: list[dict], gated: list[dict],
                     hc_res: set[str], bl_res: set[str],
                     stem2rel: dict[str, str]) -> Counter:
    """隔离池：纯自动 hardcase 帧 + backlog 全自动帧。对 s1_relabel_v1 零写入。"""
    st: Counter = Counter()
    bl: dict[str, dict[int, tuple[dict, dict]]] = defaultdict(dict)
    for it, g in zip(items, gated):
        if it["kind"] == "backlog":
            bl[it["rel"]][it["idx"]] = (it, g)
    if AUTO.exists():
        shutil.rmtree(AUTO)
    (AUTO / "vlabels").mkdir(parents=True)
    (AUTO / "images").mkdir()
    write_meta(AUTO, 1034, "shtm-s1-hardcase-auto-quarantine",
               "双VLM一致自动判定隔离池(未定状态):晋升需用户过目自动删框拼图否决误删后,"
               "整体拷入 s1_relabel_v1;在此之前不得入训练")
    # 纯自动 hardcase 帧：拷 post-apply review vlabel（含 HUMAN 项的混合帧不进隔离池）
    for it, g in zip(items, gated):
        if it["kind"] != "hardcase" or g["disp"] == "HUMAN" or it["rel"] in hc_res:
            continue
        stem = it["rel"].replace("/", "_")
        if not (AUTO / "vlabels" / f"{stem}.json5").exists():
            shutil.copy(REVIEW / "vlabels" / f"{stem}.json5",
                        AUTO / "vlabels" / f"{stem}.json5")
            st["hc_frames_auto"] += 1
    # backlog 全自动帧：接受框成对象；全否决=负样本帧（empty 组约定）
    for rel, patch in sorted(bl.items()):
        if rel in bl_res:
            continue
        stem = rel.replace("/", "_")
        objs = []
        for bi in sorted(patch):
            it, g = patch[bi]
            if g["disp"] == "AUTO_REJECT":
                st["bl_boxes_reject"] += 1
                continue
            cid = svm.CLS2ID[g["cls"]]
            objs.append({"id": len(objs), "category": cid, "confidence": svm.rd(0.5),
                         "polygon": poly_pts(it["poly"]),
                         "properties": props_of(cid, g.get("attrs", {}), False)})
            st["bl_boxes_accept"] += 1
        (AUTO / "vlabels" / f"{stem}.json5").write_text(
            json.dumps(empty_vlabel(rel, objs), ensure_ascii=False, indent=1) + "\n")
        st["bl_frames_auto"] += 1
        st["bl_frames_neg"] += (len(objs) == 0)
    for p in sorted((AUTO / "vlabels").glob("*.json5")):
        img = SNAP / f"{stem2rel[p.stem]}.jpg"
        if not img.exists():
            raise FileNotFoundError(img)
        os.symlink(img, AUTO / "images" / f"{p.stem}.jpg")
    return st


def build_r2(items: list[dict], gated: list[dict], idmap: dict[str, dict[int, int]],
             hc_res: set[str], bl_res: set[str], stem2rel: dict[str, str]) -> None:
    if R2.exists():
        shutil.rmtree(R2)
    (R2 / "vlabels").mkdir(parents=True)
    (R2 / "images").mkdir()
    write_meta(R2, 1033, "shtm-s1-hardcase-review-r2",
               "S1 难例双VLM削减后残留人工审核集:仅含分歧/不确定项;"
               "同帧内已自动判定对象为上下文;自动部分已隔离至 s1_hardcase_auto")
    per_rel: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for it, g in zip(items, gated):
        if it["kind"] == "backlog":
            per_rel[it["rel"]].append((it, g))
    resid_rows: list[dict] = []
    for rel in sorted(hc_res):
        stem = rel.replace("/", "_")
        shutil.copy(REVIEW / "vlabels" / f"{stem}.json5", R2 / "vlabels" / f"{stem}.json5")
    for it, g in zip(items, gated):
        if it["kind"] == "hardcase" and g["disp"] == "HUMAN" and it["rel"] in hc_res:
            new_id = idmap.get(it["rel"], {}).get(it["idx"], it["idx"])
            resid_rows.append({"rel": it["rel"], "obj_idx": new_id, "kind": "hardcase",
                               "reasons": it["reasons"], "prefill": it["prefill"],
                               "why": g["why"]})
    for rel in sorted(bl_res):
        stem = rel.replace("/", "_")
        lab = empty_vlabel(rel, [])
        for it, g in per_rel.get(rel, []):
            if g["disp"].startswith("AUTO_CONFIRM"):
                cid = svm.CLS2ID[g["cls"]]
                lab["objects"].append({"id": len(lab["objects"]), "category": cid,
                                       "confidence": svm.rd(it.get("conf", 0.5)),
                                       "polygon": poly_pts(it["poly"]),
                                       "properties": props_of(cid, g.get("attrs", {}), False)})
        for it, g in per_rel.get(rel, []):
            if g["disp"] == "HUMAN":
                cid = svm.CLS2ID[it["prefill"]]
                lab["objects"].append({"id": len(lab["objects"]), "category": cid,
                                       "confidence": svm.rd(it.get("conf", 0.5)),
                                       "polygon": poly_pts(it["poly"]),
                                       "properties": props_of(cid, {}, True)})
                resid_rows.append({"rel": rel, "obj_idx": lab["objects"][-1]["id"],
                                   "kind": "backlog", "reasons": it["reasons"],
                                   "prefill": it["prefill"], "why": g["why"]})
        (R2 / "vlabels" / f"{stem}.json5").write_text(
            json.dumps(lab, ensure_ascii=False, indent=1) + "\n")
    RESIDUAL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in resid_rows))
    for p in sorted((R2 / "vlabels").glob("*.json5")):
        img = SNAP / f"{stem2rel[p.stem]}.jpg"
        if not img.exists():
            raise FileNotFoundError(img)
        os.symlink(img, R2 / "images" / f"{p.stem}.jpg")


def write_ledger(items: list[dict], gated: list[dict], pilot: bool = False) -> None:
    path = LEDGER.with_name("hardcase_auto_pilot.jsonl") if pilot else LEDGER
    with path.open("w") as f:
        for it, g in zip(items, gated):
            f.write(json.dumps({**it, "gate": g,
                                "spark": _CACHE.get(f"hp{PV}|spark|{it['rel']}|{it['idx']}"),
                                "doubao": _CACHE.get(f"hp{PV}|doubao|{it['rel']}|{it['idx']}")},
                               ensure_ascii=False) + "\n")


def montage(items: list[dict], gated: list[dict]) -> None:
    dels = [(it, g) for it, g in zip(items, gated) if g["disp"] == "AUTO_DELETE"]
    if not dels:
        return
    cols, cell = 10, 160
    pages = [dels[i:i + 300] for i in range(0, len(dels), 300)]
    for pi, page in enumerate(pages, 1):
        rows = (len(page) + cols - 1) // cols
        im = Image.new("RGB", (cols * cell, rows * (cell + 14)), (24, 24, 24))
        dr = ImageDraw.Draw(im)
        for i, (it, g) in enumerate(page):
            data = make_crop(it["rel"], it["poly"])
            x, y = (i % cols) * cell, (i // cols) * (cell + 14)
            if data:
                crop = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1])))
                crop.thumbnail((cell, cell))
                im.paste(crop, (x, y))
            dr.text((x + 2, y + cell), f"{it['rel'].split('/')[1][-6:]}#{it['idx']}",
                    fill=(255, 255, 80))
        out = S1.parent / "research" / f"2026-09-10-SHTM自动删框拼图-{pi}.jpg"
        im.save(out, quality=85)
        print(f"montage: {out} ({len(page)} crops)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--only", choices=["spark", "doubao"], default=None,
                    help="只跑一侧（灌缓存，不出产物）；None=双模型")
    ap.add_argument("--scan-only", action="store_true", help="只跑 scan 不做闸门与应用")
    args = ap.parse_args()
    global _CACHE
    _CACHE = load_cache()
    items = load_items()
    print(f"items: {Counter(tuple(it['reasons']) for it in items)}")
    if args.pilot:
        by_key: dict[str, list[dict]] = {}
        for it in items:
            by_key.setdefault(it["reasons"][0], []).append(it)
        pick: list[dict] = []
        step = max(1, args.pilot // max(1, len(by_key)))
        for v in by_key.values():
            pick += v[::step][:args.pilot // len(by_key) + 1]
        items = pick[:args.pilot]
        print(f"pilot: {len(items)} items")
    asyncio.run(scan(items, args.only))
    if args.scan_only:
        return
    gated = [gate(it, _CACHE[f"hp{PV}|spark|{it['rel']}|{it['idx']}"],
                  _CACHE[f"hp{PV}|doubao|{it['rel']}|{it['idx']}"]) for it in items]
    st = Counter(g["disp"] for g in gated)
    why = Counter(f"{g['disp']}:{g['why'].split(':')[0]}" for g in gated)
    print("gate:", dict(st))
    print("why:", dict(why))
    write_ledger(items, gated, pilot=bool(args.pilot))
    if args.pilot:
        for it, g in zip(items, gated):
            if g["disp"] == "HUMAN" and "conflict" in g["why"]:
                print("CONFLICT sample:", it["rel"], it["idx"], g["why"])
        return
    idmap = apply_review(items, gated)
    hc_res, bl_res = set(), set()
    for it, g in zip(items, gated):
        if g["disp"] == "HUMAN":
            (hc_res if it["kind"] == "hardcase" else bl_res).add(it["rel"])
    st_q = build_quarantine(items, gated, hc_res, bl_res,
                            {it["rel"].replace("/", "_"): it["rel"] for it in items})
    build_r2(items, gated, idmap, hc_res, bl_res,
             {it["rel"].replace("/", "_"): it["rel"] for it in items})
    montage(items, gated)
    (S1 / "hardcase_prune_stats.json").write_text(json.dumps(
        {"gate": dict(st), "why": dict(why), "quarantine": dict(st_q),
         "residual_frames": {"hardcase": len(hc_res), "backlog": len(bl_res)}},
        ensure_ascii=False, indent=1))
    print("quarantine:", dict(st_q))
    print(f"residual frames: hardcase {len(hc_res)} + backlog {len(bl_res)} -> {R2}")


if __name__ == "__main__":
    main()
