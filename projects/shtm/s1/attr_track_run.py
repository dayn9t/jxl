#!/usr/bin/env python3
"""SHTM S1 attribute-track relabel: crop each attributed object -> spark VLM re-judge.

Input : projects/shtm/sampling/s1_relabel_tasks.jsonl (rows with 'attr_abs' in reasons)
Images: /var/ias/snapshot/shtm/<rel>.jpg + <rel>_m31.json (read-only, never modified)
VLM   : http://192.168.18.182:8000/v1/chat/completions  qwen3.5-35b-a3b-fp8
        chat_template_kwargs.enable_thinking=false + response_format json_object
Output: /tmp/shtm_s1/attr_track.jsonl  (resumable: non-error rows skipped on start)

Attribute binding (m31 meta /home/jiang/ws/trash/meta/m31.json):
  class 0 opening -> sort(0干/1湿/2可回收/3有害) + amount(0空/1不满/2将满/3已满) + illegal(0否/1是)
  class 1 lid     -> sort + side(0正/1反)
"""
import argparse
import asyncio
import base64
import io
import json
import os
import time
from collections import OrderedDict

import aiohttp
from PIL import Image

TASKS = "/home/jiang/cc/py/jxl/projects/shtm/sampling/s1_relabel_tasks.jsonl"
SNAP = "/var/ias/snapshot/shtm"
OUT = "/tmp/shtm_s1/attr_track.jsonl"
VLM_URL = "http://192.168.18.182:8000/v1/chat/completions"
MODEL = "qwen3.5-35b-a3b-fp8"
PAD = 0.15
JPEG_Q = 90

ATTRS_BY_CLASS = {0: ["sort", "amount", "illegal"], 1: ["sort", "side"]}
VALUE_RANGE = {"sort": (0, 3), "amount": (0, 3), "illegal": (0, 1), "side": (0, 1)}
CLASS_NAME = {0: "垃圾桶开口(opening)", 1: "垃圾桶盖(lid)"}

# m31 value semantics + form dictionary (projects/shtm/垃圾桶形态词典.md) gist
ATTR_DEF = {
    "sort": (
        "sort 垃圾桶类型(按桶体/桶盖颜色与标识判定): "
        "0=干垃圾(其他垃圾, 黑色或黄色桶) 1=湿垃圾(厨余垃圾, 棕色或绿色桶) "
        "2=可回收物(蓝色桶) 3=有害垃圾(红色桶)"
    ),
    "amount": (
        "amount 桶内垃圾量(按桶口可见垃圾高度判定): "
        "0=空桶(口内看不到垃圾) 1=不满(垃圾少量) 2=将满(垃圾接近桶口) "
        "3=已满(垃圾到达桶口或满溢出桶)"
    ),
    "illegal": (
        "illegal 违规投放: 0=无违规(垃圾均已入桶) "
        "1=有违规(垃圾袋/散垃圾堆放在桶口外、桶盖上, 或垃圾满溢出桶未入袋)"
    ),
    "side": (
        "side 桶盖正反面: 0=正面(盖顶面朝上, 可见盖面/分类标识) "
        "1=反面(盖被翻开, 可见盖底/内侧)"
    ),
}
FORM_DICT = (
    "垃圾桶形态多样: 标准方桶、圆柱形圆桶、桶口套塑料袋(袋沿外翻可见)的套袋桶、"
    "筐/篮状(镂空或编织感)都算垃圾桶。"
)

SYSTEM = "你是上海垃圾房监控图像的垃圾桶属性标注专家, 只输出 JSON。"


def build_prompt(cls: int) -> str:
    attrs = ATTRS_BY_CLASS[cls]
    defs = "\n".join(f"- {ATTR_DEF[a]}" for a in attrs)
    fmt_hint = json.dumps(
        {a: {"value": "<整数>", "conf": "<0-1小数, 你对该判定的置信度>"} for a in attrs},
        ensure_ascii=False,
    )
    fmt = json.dumps({a: {"value": 0, "conf": 0.9} for a in attrs}, ensure_ascii=False)
    return (
        f"这是固定监控摄像头画面中「{CLASS_NAME[cls]}」区域的裁切图(含少量周围背景)。\n"
        f"{FORM_DICT}\n"
        f"请仅依据图中可见内容判定以下属性:\n{defs}\n"
        f"看不清/不确定的属性照常给最可能值但把 conf 给低。禁止臆造图中没有的东西。\n"
        f"输出 JSON(仅此一个对象, 不要多余字段): {fmt_hint}\n"
        f"示例: {fmt}"
    )


def parse_attr_payload(raw: str, cls: int) -> dict:
    """Validate VLM JSON against class-bound attr set. Raise ValueError on bad shape."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"top-level not object: {type(data).__name__}")
    out = {}
    for a in ATTRS_BY_CLASS[cls]:
        if a not in data:
            raise ValueError(f"missing key: {a}")
        v = data[a]
        if not isinstance(v, dict):
            raise ValueError(f"{a} not an object: {v!r}")
        val, conf = v.get("value"), v.get("conf")
        if isinstance(val, str) and val.strip().lstrip("-").isdigit():
            val = int(val.strip())
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{a}.value not int: {val!r}")
        lo, hi = VALUE_RANGE[a]
        if not lo <= val <= hi:
            raise ValueError(f"{a}.value out of range [{lo},{hi}]: {val}")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
            conf = None
        else:
            conf = round(float(conf), 4)
        out[a] = {"value": val, "conf": conf}
    return out


def crop_bytes(img: Image.Image, polygon: list) -> bytes:
    W, H = img.size
    xs = [p["x"] for p in polygon]
    ys = [p["y"] for p in polygon]
    x0, y0, x1, y1 = min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H
    pw, ph = (x1 - x0) * PAD, (y1 - y0) * PAD
    x0, y0 = max(0.0, x0 - pw), max(0.0, y0 - ph)
    x1, y1 = min(float(W), x1 + pw), min(float(H), y1 + ph)
    buf = io.BytesIO()
    img.crop((int(x0), int(y0), int(x1), int(y1))).save(buf, format="JPEG", quality=JPEG_Q)
    return buf.getvalue()


def load_jobs() -> list:
    """One job per attributed object (has non-empty properties in m31)."""
    jobs = []
    with open(TASKS) as f:
        rels = [json.loads(l)["rel"] for l in f if "attr_abs" in json.loads(l)["reasons"]]
    for rel in rels:
        d = json.load(open(f"{SNAP}/{rel}_m31.json"))
        for idx, o in enumerate(d.get("objects", [])):
            props = o.get("properties") or []
            if not props:
                continue
            cls = o["prob_class"]["value"]
            old = {
                p["key"]: {"value": p["value"]["value"], "conf": round(p["value"]["confidence"], 4)}
                for p in props
            }
            jobs.append({"rel": rel, "obj_idx": idx, "cls": cls, "old": old, "polygon": o["polygon"]})
    return jobs


async def call_vlm(session, b64: str, cls: int, extra_msg: str | None) -> str:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": build_prompt(cls)},
    ]
    if extra_msg:
        content.append({"type": "text", "text": extra_msg})
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(VLM_URL, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        body = await resp.json()
    return body["choices"][0]["message"]["content"]


async def judge(session, img: Image.Image, job: dict) -> dict:
    """Caller must hold the semaphore (bounds concurrently live images)."""
    b64 = await asyncio.to_thread(lambda: base64.b64encode(crop_bytes(img, job["polygon"])).decode())
    extra = None
    last_err = None
    raw = ""
    for attempt in range(4):  # transport: 3 retries; validation: 1 self-correct retry
        try:
            raw = await call_vlm(session, b64, job["cls"], extra)
            new = parse_attr_payload(raw, job["cls"])
            break
        except ValueError as e:
            last_err = f"invalid: {e}"
            if extra is None:
                extra = (
                    f"你上次的输出不符合要求({e})。必须输出恰好含 "
                    f"{','.join(ATTRS_BY_CLASS[job['cls']])} 字段的嵌套 JSON 对象, "
                    f'每个字段形如 {{"value":0,"conf":0.9}}, 不要输出其他任何内容。'
                )
            else:
                return {"rel": job["rel"], "obj_idx": job["obj_idx"], "error": last_err, "raw": raw[:200]}
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as e:
            last_err = f"transport/parse: {type(e).__name__}: {str(e)[:200]}"
            if attempt >= 2:
                return {"rel": job["rel"], "obj_idx": job["obj_idx"], "error": last_err}
            await asyncio.sleep(2 * (attempt + 1))
    else:
        return {"rel": job["rel"], "obj_idx": job["obj_idx"], "error": last_err}

    old = job["old"]
    common = [k for k in new if k in old]
    agree = all(new[k]["value"] == old[k]["value"] for k in common) if common else None
    return {"rel": job["rel"], "obj_idx": job["obj_idx"], "cls": job["cls"], "new": new, "old": old, "agree": agree}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="pilot: only first N jobs")
    ap.add_argument("--conc", type=int, default=32)
    args = ap.parse_args()

    jobs = load_jobs()
    if args.limit:
        jobs = jobs[: args.limit]

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                try:
                    d = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if "error" not in d:  # error rows retried on rerun
                    done.add(f"{d['rel']}|{d['obj_idx']}")
    todo = [j for j in jobs if f"{j['rel']}|{j['obj_idx']}" not in done]
    print(f"jobs={len(jobs)} done_prev={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    sem = asyncio.Semaphore(args.conc)
    timeout = aiohttp.ClientTimeout(total=180)
    conn = aiohttp.TCPConnector(limit=args.conc)
    lru: OrderedDict[str, Image.Image] = OrderedDict()
    lock = asyncio.Lock()
    ok = fail = 0
    t0 = time.time()

    async def get_img(rel: str) -> Image.Image:
        if rel not in lru:
            lru[rel] = await asyncio.to_thread(lambda: Image.open(f"{SNAP}/{rel}.jpg").convert("RGB"))
        lru.move_to_end(rel)
        return lru[rel]

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:

        async def run_one(job: dict) -> None:
            nonlocal ok, fail
            async with sem:  # acquire BEFORE image load: bounds live images to conc (+lru)
                img = await get_img(job["rel"])
                row = await judge(session, img, job)
                while len(lru) > args.conc * 3:
                    lru.popitem(last=False)
            async with lock:
                with open(OUT, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if "error" in row:
                    fail += 1
                else:
                    ok += 1
                n = ok + fail
                if n % 100 == 0 or n == len(todo):
                    rate = n / (time.time() - t0)
                    print(
                        f"{n}/{len(todo)} ok={ok} fail={fail} {rate:.1f}/s "
                        f"eta={((len(todo) - n) / rate) / 60:.1f}min",
                        flush=True,
                    )

        await asyncio.gather(*(run_one(j) for j in todo))

    print(f"FINISHED ok={ok} fail={fail} in {(time.time() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
