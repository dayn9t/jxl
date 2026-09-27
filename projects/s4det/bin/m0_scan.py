#!/usr/bin/env python3
"""s4det M0 烟火独立线 phase 1——三域事件库烟火弱标签全量扫描.

目的: D2 烟火类(smoke/open_flame)无在役先验、mobile 域无 ET10 突发情况事件
(p0-cluster/event_index.jsonl 实测 0 条), 候选池只能从事件文本侧弱标签投影:
平台对象 Description + scene prompt + llm_entries 双模型语句中命中烟火词族的事件,
其资源图即候选图(phase 2 挖掘的进料底池)。

词族分类(命中即记, 互不排斥):
    fire        明火系——D2 open_flame 直接信号
    smoke       烟雾系——D2 smoke 直接信号
    scene_adj   野炊/烧烤/篝火/露营——证据链场景邻接(design §2c, 烟火线兼野炊证据源)
    cig         吸烟/烟头——邻接但非 D2(垃圾/行为域), 仅计数不进候选
    fire_ext    消防栓/消防通道等——设施域误命中, 仅计数不进候选

扫描范围: /var/howell/s4/v0.4/event/{mobile,road,scene} 全部日期目录(只读);
emergency2* 为 ET2 再标注实验目录(实测 emergency_type=2, 与烟火无关)不入扫。

产出 /mnt/data/s4det/m0-fire/scan/:
    matches.jsonl    命中事件逐条(域/日期/事件类型/词族/证据片段/候选图路径)
    scan_stats.json  三域 event_type 直方图(补 road/scene 索引空白) + 词族计数 + 计时

典型用法:
    python m0_scan.py /var/howell/s4/v0.4/event /mnt/data/s4det/m0-fire/scan
"""

import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import orjson
import typer
from loguru import logger
from pydantic import BaseModel

app = typer.Typer(add_completion=False, help="M0 phase 1: 三域烟火弱标签扫描.")

WORD_FAMILIES: dict[str, list[str]] = {
    "fire": ["明火", "火焰", "火灾", "起火", "着火", "火苗", "火情", "燃烧", "焚烧", "烧毁", "篝火", "焚毁", "烧纸"],
    "smoke": ["冒烟", "烟雾", "烟尘", "黑烟", "白烟", "浓烟", "轻烟", "炊烟", "烟柱", "起烟"],
    "scene_adj": ["野炊", "烧烤", "露营", "野营", "帐篷", "钓鱼", "垂钓"],
    "cig": ["吸烟", "香烟", "烟头", "烟蒂", "禁烟"],
    "fire_ext": ["消防栓", "消防通道", "消防车", "灭火器", "消防设施", "消防器材"],
}
FAMILY_RES: dict[str, re.Pattern[str]] = {k: re.compile("|".join(v)) for k, v in WORD_FAMILIES.items()}
ENG_RES = re.compile(r"(?i)\b(fire|flame|smoke|smoking)\b")

CANDIDATE_FAMILIES = {"fire", "smoke"}
"""进候选池的词族; scene_adj/cig/fire_ext 仅记录计数与 matches 行(供扩源分析)."""

MOBILE_ET: dict[int, str] = {
    1: "机动车乱停", 2: "非机动车乱停", 3: "丢弃垃圾", 4: "道路设施损坏", 5: "店招损坏",
    6: "占道施工", 7: "店铺装修", 8: "店招消失", 9: "店招新增", 10: "突发情况",
    11: "占道堆物", 12: "占道设摊", 13: "地面油污", 14: "地面积水", 15: "违规充电",
    16: "占道经营", 17: "井盖破损", 18: "道路破损", 19: "散落垃圾",
}
SCENE_TT: dict[int, str] = {201: "抓图分析", 202: "视频任务", 203: "事件自动闭环"}


class EventScan(BaseModel):
    """单事件扫描结果(matches.jsonl 行模型)."""

    domain: str
    date: str
    event_id: str
    event_type: int
    type_name: str
    families: list[str]
    evidence: list[str]
    image_paths: list[str]


def _scan_texts(texts: list[str]) -> tuple[list[str], list[str]]:
    """文本列表 → (命中词族, 证据片段). 英语词并入对应族(fire/smoke)."""
    fams: set[str] = set()
    ev: list[str] = []
    for t in texts:
        if not t:
            continue
        for fam, pat in FAMILY_RES.items():
            m = pat.search(t)
            if m:
                fams.add(fam)
                s = max(0, m.start() - 18)
                ev.append(f"[{fam}]…{t[s : m.end() + 18]}…")
        m = ENG_RES.search(t)
        if m:
            eng = "smoke" if "smok" in m.group(0).lower() else "fire"
            fams.add(eng)
            s = max(0, m.start() - 18)
            ev.append(f"[{eng}:en]…{t[s : m.end() + 18]}…")
    return sorted(fams), ev[:6]


def _collect_images(ev_dir: Path) -> list[str]:
    """事件目录资源图: resources/*.jpg + scene images/*.jpg(去重排序)."""
    out: set[str] = set()
    for sub in ("resources", "images"):
        d = ev_dir / sub
        if d.is_dir():
            out.update(str(p) for p in d.glob("*.jpg"))
    return sorted(out)


def scan_event(ev_dir: Path, domain: str, date: str) -> tuple[EventScan | None, tuple[str, int, str]]:
    """扫单事件: event.json(Description/prompt) + llm_entries.json(class/desc/reason).

    返回 (matches 行或 None, 域×类型直方图键)——单遍完成词族扫描与类型计数.
    """
    ej = ev_dir / "event.json"
    unk = (domain, -1, "?")
    if not ej.is_file():
        return None, unk
    try:
        data = orjson.loads(ej.read_bytes())
    except orjson.JSONDecodeError:
        logger.warning("event.json 解析失败: {}", ev_dir)
        return None, unk
    texts: list[str] = []
    for res in data.get("Resources") or []:
        if isinstance(res, dict):
            for obj in res.get("Objects") or []:
                if isinstance(obj, dict) and obj.get("Description"):
                    texts.append(str(obj["Description"]))
    if data.get("Prompt"):
        texts.append(str(data["Prompt"]))
    cj = ev_dir / "confirm.json"
    if cj.is_file():
        try:
            cdata = orjson.loads(cj.read_bytes())
            for conf in cdata.get("Confirmations") or []:
                if isinstance(conf, dict) and conf.get("Description"):
                    texts.append(str(conf["Description"]))
        except orjson.JSONDecodeError:
            logger.warning("confirm.json 解析失败: {}", ev_dir)
    etype = int(data.get("EventType") or data.get("TaskType") or -1)
    tname = (MOBILE_ET if domain == "mobile" else SCENE_TT if domain == "scene" else {}).get(etype, "?")
    hist_key = (domain, etype, tname)
    le = ev_dir / "llm_entries.json"
    if le.is_file():
        try:
            for ent in orjson.loads(le.read_bytes()) or []:
                if not isinstance(ent, dict):
                    continue
                for key in ("class", "description", "reason"):
                    for model in ("doubao", "qwen"):
                        v = ent.get(model, {}) if isinstance(ent.get(model), dict) else {}
                        if isinstance(v, dict) and v.get(key):
                            texts.append(str(v[key]))
        except orjson.JSONDecodeError:
            logger.warning("llm_entries.json 解析失败: {}", ev_dir)
    fams, ev = _scan_texts(texts)
    if not fams:
        return None, hist_key
    return EventScan(
        domain=domain, date=date, event_id=ev_dir.name, event_type=etype, type_name=tname,
        families=families_sorted(fams), evidence=ev, image_paths=_collect_images(ev_dir),
    ), hist_key


def families_sorted(fams: set[str]) -> list[str]:
    """词族排序输出(确定序)."""
    order = ["fire", "smoke", "scene_adj", "cig", "fire_ext"]
    return [f for f in order if f in fams]


def _scan_date(args: tuple[str, str, Path]) -> tuple[list[EventScan], list[tuple[str, int, str]]]:
    domain, date, ddir = args
    out: list[EventScan] = []
    hist: list[tuple[str, int, str]] = []
    for ev_dir in sorted(ddir.iterdir()):
        if ev_dir.is_dir() and len(ev_dir.name) == 24:
            r, key = scan_event(ev_dir, domain, date)
            hist.append(key)
            if r:
                out.append(r)
    return out, hist


@app.command()
def main(
    event_root: Annotated[Path, typer.Option(help="事件库根(只读)")],
    out_dir: Annotated[Path, typer.Option(help="扫描产物目录")],
    workers: Annotated[int, typer.Option(help="并行线程数")] = 16,
) -> None:
    """全库扫描 → matches.jsonl + scan_stats.json."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, Path]] = []
    et_hist: Counter[tuple[str, int, str]] = Counter()
    for domain in ("mobile", "road", "scene"):
        for ddir in sorted((event_root / domain).iterdir()):
            if ddir.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", ddir.name):
                jobs.append((domain, ddir.name, ddir))
    logger.info("待扫日期目录 {} 个(域 × 日期)", len(jobs))
    fam_hist: Counter[str] = Counter()
    rows: list[EventScan] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res, hist in ex.map(_scan_date, jobs):
            rows.extend(res)
            et_hist.update(hist)
    rows.sort(key=lambda r: (r.domain, r.date, r.event_id))
    with (out_dir / "matches.jsonl").open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r.model_dump()) + b"\n")
            for fam in r.families:
                fam_hist[fam] += 1
    n_cand = sum(1 for r in rows if CANDIDATE_FAMILIES & set(r.families))
    stats = {
        "batch": "2026-09-27-m0-phase1-scan",
        "event_root": str(event_root),
        "n_date_dirs": len(jobs),
        "n_events_matched": len(rows),
        "n_events_candidate": n_cand,
        "family_hist": dict(fam_hist),
        "event_type_hist": {f"{d}/{t}:{n}": c for (d, t, n), c in sorted(et_hist.items())},
        "elapsed_sec": round(time.time() - t0, 1),
        "script": "m0_scan.py",
    }
    (out_dir / "scan_stats.json").write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    logger.info("命中 {} 事件(候选词族 {}), 耗时 {}s → {}", len(rows), n_cand, stats["elapsed_sec"], out_dir)


if __name__ == "__main__":
    app()
