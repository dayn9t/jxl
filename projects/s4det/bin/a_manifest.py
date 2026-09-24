#!/usr/bin/env python3
"""s4det 任务书 A ①——mobile 五粗类抽样 manifest 生成.

扫描事件根(只读) → event_index.jsonl 全量索引 → 按任务书 A 抽样表产
{family}/manifest.jsonl + sampling_stats.json + {family}/run_config.json(批指纹).

抽样纪律(任务书 A): 跨日期均匀(逐月轮转); 同摄像头同时段去过密(每 (DeviceId,3h) 上限);
Conf<50 保留入聚类但记录分档; 小类全量; 弃置(解析失败/缺图)入隔离台账.

典型用法:
    python a_manifest.py /var/howell/s4/v0.4/event/mobile /mnt/data/s4det/p0-cluster
"""

import hashlib
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import orjson
import typer
from a_common import (
    BATCH_ID,
    ET_NAMES,
    FAMILY_SPECS,
    LEDGER,
    EventRec,
    FamilySpec,
    LlmPair,
    ManifestRow,
    ObjectRec,
    QuarantineEntry,
    append_quarantine,
    code_band,
    conf_band,
    read_jsonl,
    write_jsonl,
)
from loguru import logger

app = typer.Typer(add_completion=False, help="任务书 A: mobile 五粗类抽样 manifest.")


def _parse_objects(res: dict[str, object]) -> tuple[list[ObjectRec], int, int]:
    """Resources[i].Objects[] → (ObjectRec 表, 内容全等重复 Id 数, 内容冲突重复 Id 数).

    平台存在同事件重复下发同 Id 对象(2026-09-25 实测 965 事件/超额 1,079 个, 抽验
    内容逐字节雷同): 全等 → 去重; 内容不等 → 双保留加 #n 后缀(不丢真检测).
    """
    out: list[ObjectRec] = []
    seen: dict[str, tuple] = {}
    n_ident = 0
    n_conflict = 0
    occur: dict[str, int] = {}
    for obj in res.get("Objects") or []:
        if not isinstance(obj, dict):
            continue
        poly = obj.get("Polygon") or []
        xs = [float(p["X"]) for p in poly if isinstance(p, dict) and "X" in p]
        ys = [float(p["Y"]) for p in poly if isinstance(p, dict) and "Y" in p]
        if len(xs) < 2 or len(ys) < 2:
            continue
        desc = obj.get("Description")
        code = str(desc) if desc is not None else ""
        conf = float(obj.get("Confidence") or 0.0)
        oid = str(obj.get("Id", ""))
        content = (code, conf, min(xs), min(ys), max(xs), max(ys))
        if oid in seen:
            if seen[oid] == content:
                n_ident += 1
                continue
            n_conflict += 1
            occur[oid] = occur.get(oid, 1) + 1
            oid = f"{oid}#{occur[oid]}"
        else:
            seen[oid] = content
        out.append(
            ObjectRec(
                object_id=oid,
                code=code,
                band=code_band(code),
                conf=conf,
                conf_band=conf_band(conf),
                bbox=[min(xs), min(ys), max(xs), max(ys)],
            )
        )
    return out, n_ident, n_conflict


def _parse_llm(ev_dir: Path) -> tuple[dict[str, LlmPair], bool]:
    """llm_entries.json → object_id → 双模型类别; 返回 (表, 解析是否成功)."""
    path = ev_dir / "llm_entries.json"
    if not path.is_file():
        return {}, True
    try:
        rows = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        return {}, False
    if not isinstance(rows, list):
        return {}, False
    pairs: dict[str, LlmPair] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        doubao = row.get("doubao")
        qwen = row.get("qwen")
        d_cls = (
            str(doubao.get("class"))
            if isinstance(doubao, dict) and doubao.get("class")
            else None
        )
        q_cls = (
            str(qwen.get("class"))
            if isinstance(qwen, dict) and qwen.get("class")
            else None
        )
        pairs[str(row.get("object_id", ""))] = LlmPair(
            doubao=d_cls,
            qwen=q_cls,
            agree=(d_cls == q_cls if d_cls and q_cls else None),
        )
    return pairs, True


def scan_event(
    ev_dir: Path,
) -> tuple[EventRec | None, QuarantineEntry | None, bool, tuple[int, int]]:
    """单事件目录 → (索引行, 弃置行, llm 解析成功, (全等重复, 冲突重复))."""
    ej = ev_dir / "event.json"
    qid = f"{BATCH_ID}:{ev_dir.parent.name}/{ev_dir.name}"
    dups = (0, 0)
    try:
        data = orjson.loads(ej.read_bytes())
        ress = data.get("Resources") or []
        res = ress[0] if ress else {}
        rid = str(res.get("ResourceId", "1"))
        img = ev_dir / "resources" / f"{rid}.jpg"
        etype = int(data.get("EventType") or 0)
        event_time = str(data.get("EventTime") or "")
        objects, n_ident, n_conflict = _parse_objects(res)
        dups = (n_ident, n_conflict)
        rec = EventRec(
            dir=str(ev_dir),
            date=ev_dir.parent.name,
            event_id=ev_dir.name,
            event_type=etype,
            et_name=ET_NAMES.get(etype, f"ET{etype}"),
            device_id=str(data.get("DeviceId") or ""),
            hour=int(event_time[11:13]) if len(event_time) >= 13 else -1,
            image_path=str(img) if img.is_file() else None,
            objects=objects,
            llm=_parse_llm(ev_dir)[0],
        )
    except (
        orjson.JSONDecodeError,
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        OSError,
    ) as e:
        return (
            None,
            QuarantineEntry(
                id=qid,
                batch=BATCH_ID,
                gate="manifest-parse",
                verdict=f"event.json 解析失败: {type(e).__name__}: {e}",
                model="rule",
            ),
            True,
            dups,
        )
    if rec.image_path is None:
        return (
            None,
            QuarantineEntry(
                id=qid,
                batch=BATCH_ID,
                gate="image-missing",
                verdict=f"原图缺失: {ev_dir / 'resources'} 无 {rid}.jpg",
                model="rule",
            ),
            True,
            dups,
        )
    return rec, None, True, dups


def scan_root(
    root: Path,
) -> tuple[list[EventRec], list[QuarantineEntry], int, tuple[int, int]]:
    """全量扫描事件根; 返回 (索引, 弃置行, llm 解析失败数, (全等重复, 冲突重复) 总计)."""
    recs: list[EventRec] = []
    quar: list[QuarantineEntry] = []
    llm_fail = 0
    dup_ident = 0
    dup_conflict = 0
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ev_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            rec, q, llm_ok, dups = scan_event(ev_dir)
            llm_fail += 0 if llm_ok else 1
            dup_ident += dups[0]
            dup_conflict += dups[1]
            if q:
                quar.append(q)
            if rec:
                recs.append(rec)
        logger.info("扫描完成 {}", date_dir.name)
    return recs, quar, llm_fail, (dup_ident, dup_conflict)


def _pick_objects(ev: EventRec, cap: int) -> list[ObjectRec]:
    """事件内选框: conf 降序取前 cap."""
    return sorted(ev.objects, key=lambda o: (-o.conf, o.object_id))[:cap]


def _sample_uniform(
    evs: list[EventRec],
    budget: int,
    rng: random.Random,
    group_counts: Counter,
    per_group_cap: int,
    cost: Callable[[EventRec], int],
) -> list[tuple[EventRec, int]]:
    """逐月轮转 + (日期,设备,3h 桶) 去过密抽样; 返回 (事件, 选取数) 有序表."""
    by_month: dict[str, list[EventRec]] = {}
    for ev in evs:
        by_month.setdefault(ev.month, []).append(ev)
    months = sorted(by_month)
    for m in months:
        rng.shuffle(by_month[m])
    cursor = dict.fromkeys(months, 0)
    active = set(months)
    picked: list[tuple[EventRec, int]] = []
    spent = 0
    while active and spent < budget:
        for m in sorted(active):
            pool = by_month[m]
            i = cursor[m]
            while i < len(pool):
                ev = pool[i]
                i += 1
                key = (ev.date, ev.device_id, ev.hour // 3)
                if group_counts[key] >= per_group_cap:
                    continue
                cursor[m] = i
                group_counts[key] += 1
                picked.append((ev, cost(ev)))
                spent += cost(ev)
                break
            else:
                cursor[m] = i
                active.discard(m)
    return picked


def _type_events(evs: list[EventRec], etype: int) -> list[EventRec]:
    return [e for e in evs if e.event_type == etype]


def _rows_for_event(
    spec: FamilySpec,
    ev: EventRec,
    objs: list[ObjectRec],
    out_dir: Path,
    source: str = "event_type",
) -> list[ManifestRow]:
    """事件 → manifest 行(crop 族=每框一行; original 族=原图一行; state_carrier=原图+框)."""
    base = dict(  # 公共字段一次拼装(rid 由各分支自定)
        family=spec.name,
        event_dir=ev.dir,
        date=ev.date,
        event_id=ev.event_id,
        event_type=ev.event_type,
        et_name=ev.et_name,
        device_id=ev.device_id,
        hour=ev.hour,
        image_path=ev.image_path or "",
        source=source,
    )
    stem = f"{spec.name}_{ev.date}_{ev.event_id}"
    llm = ev.llm
    if spec.mode == "original" and not spec.with_object_crops:
        return [ManifestRow(**base, mode="original", rid="1", code_band="none", crop_path=str(out_dir / "crops" / f"{stem}_r1.jpg"))]  # fmt: skip
    rows: list[ManifestRow] = []
    rid = "1"
    if spec.with_object_crops:
        rows.append(
            ManifestRow(
                **base,
                mode="original",
                rid=rid,
                code_band="none",
                crop_path=str(out_dir / "crops" / f"{stem}_r{rid}.jpg"),
            )
        )
    for obj in objs:
        pair = llm.get(obj.object_id)
        rows.append(
            ManifestRow(
                **base,
                mode="crop",
                rid=rid,
                object_id=obj.object_id,
                code=obj.code,
                code_band=obj.band,
                conf=obj.conf,
                conf_band=obj.conf_band,
                bbox=obj.bbox,
                llm_doubao=pair.doubao if pair else None,
                llm_qwen=pair.qwen if pair else None,
                llm_agree=pair.agree if pair else None,
                crop_path=str(out_dir / "crops" / f"{stem}_r{rid}o{obj.object_id}.jpg"),
            )
        )
    return rows


def _llm_object_index(
    all_evs: list[EventRec], agree_class: str
) -> list[tuple[EventRec, ObjectRec]]:
    """全库索引中收集双模型一致判为 agree_class 的 (事件, 平台框)."""
    out: list[tuple[EventRec, ObjectRec]] = []
    for ev in all_evs:
        by_id = {o.object_id: o for o in ev.objects}
        for oid, pair in ev.llm.items():
            if (
                pair.doubao
                and pair.doubao == pair.qwen
                and pair.doubao == agree_class
                and oid in by_id
            ):
                out.append((ev, by_id[oid]))
    return out


def _boost_rows(
    spec: FamilySpec,
    boost: dict[str, int],
    all_evs: list[EventRec],
    out_dir: Path,
    rng: random.Random,
    have: set[tuple[str, str]],
) -> list[ManifestRow]:
    """llm 一致类别补样(ET15/16/19 在 event.json 无事件, 仅以 llm 对象类别存在).

    不参与摄像头去过密组上限——补样类样本稀缺(全库 ≤122 个)且是该类别唯一来源,
    被组上限挤掉的代价不可接受; 仅保留单事件上限防同事件多框扎堆.
    """
    rows: list[ManifestRow] = []
    for cls, cap in boost.items():
        pairs = _llm_object_index(all_evs, cls)
        fresh = [(ev, o) for ev, o in pairs if (ev.event_id, o.object_id) not in have]
        rng.shuffle(fresh)
        per_event: dict[str, int] = {}
        n = 0
        for ev, obj in fresh:
            if n >= cap:
                break
            if per_event.get(ev.event_id, 0) >= spec.max_objs:
                continue
            per_event[ev.event_id] = per_event.get(ev.event_id, 0) + 1
            have.add((ev.event_id, obj.object_id))
            rows.extend(
                _rows_for_event(spec, ev, [obj], out_dir, source=f"llm_class:{cls}")
            )
            n += 1
        logger.info(
            "[{}] llm 类别补样 {}: 可用 {} → 取 {}", spec.name, cls, len(fresh), n
        )
    return rows


def build_family(
    spec: FamilySpec,
    evs: list[EventRec],
    all_evs: list[EventRec],
    out_dir: Path,
    seed: int,
    group_counts: Counter,
) -> tuple[list[ManifestRow], dict[str, object]]:
    """按族规格抽样产 manifest 行; 返回 (rows, 该族统计)."""
    rng = random.Random(seed)
    rows: list[ManifestRow] = []
    rare_types = [t for t in spec.event_types if t not in spec.full_types]
    filler = (
        max(rare_types, key=lambda t: len(_type_events(evs, t))) if rare_types else None
    )
    skipped_no_objects = 0
    have: set[tuple[str, str]] = set()

    # 1) 小类全量(免去过密)
    for t in spec.full_types:
        for ev in sorted(_type_events(evs, t), key=lambda e: (e.month, e.event_id)):
            rows.extend(
                _rows_for_event(spec, ev, _pick_objects(ev, spec.max_objs), out_dir)
            )
    # 2) rare 类型: 总量超 rare_cap 才降采样
    for t in rare_types:
        if t == filler:
            continue
        tevs = _type_events(evs, t)
        crops_if_all = sum(min(len(e.objects), spec.max_objs) for e in tevs)
        cap = spec.rare_cap
        if cap is None or crops_if_all <= cap:
            for ev in sorted(tevs, key=lambda e: (e.month, e.event_id)):
                rows.extend(
                    _rows_for_event(spec, ev, _pick_objects(ev, spec.max_objs), out_dir)
                )
        else:
            picks = _sample_uniform(
                tevs,
                cap,
                rng,
                group_counts,
                per_group_cap=PER_GROUP_CAP,
                cost=(lambda e: 1)
                if spec.mode == "original"
                else (lambda e: min(len(e.objects), spec.max_objs)),
            )
            for ev, _n in picks:
                rows.extend(
                    _rows_for_event(spec, ev, _pick_objects(ev, spec.max_objs), out_dir)
                )
    # 3) 无框事件原图顶(fallback_original 族, 上限 target 的 10%, 先于 filler 预留)
    if spec.fallback_original:
        no_objs = [e for e in evs if e.event_type in spec.event_types and not e.objects]
        skipped_no_objects += len(no_objs)
        cap = max(spec.target // 10, 0)
        if cap and no_objs:
            picks = _sample_uniform(
                no_objs,
                cap,
                rng,
                group_counts,
                per_group_cap=PER_GROUP_CAP,
                cost=lambda _e: 1,
            )
            rows.extend(
                _rows_for_event(
                    spec, ev=e, objs=[], out_dir=out_dir, source="no_object"
                )
                for e, _n in picks
            )
    # 4) filler 类型补足预算(逐月轮转去过密)
    if filler is not None:
        tevs = _type_events(evs, filler)
        with_objs = [e for e in tevs if e.objects]
        skipped_no_objects += len(tevs) - len(with_objs)
        picks = _sample_uniform(
            with_objs,
            max(spec.target - len(rows), 0),
            rng,
            group_counts,
            per_group_cap=PER_GROUP_CAP,
            cost=lambda e: min(len(e.objects), spec.max_objs),
        )
        for ev, _n in picks:
            rows.extend(
                _rows_for_event(spec, ev, _pick_objects(ev, spec.max_objs), out_dir)
            )
    # 5) llm 一致类别补样(ET15/16/19 仅以 llm 对象类别存在, 见 CODEBOOK 弱标签组装规则)
    if spec.llm_class_boost:
        rows.extend(
            _boost_rows(spec, spec.llm_class_boost, all_evs, out_dir, rng, have)
        )
    rows = rows[: spec.target] if spec.target < 10**8 else rows

    stats: dict[str, object] = {
        "target": spec.target,
        "n_rows": len(rows),
        "mode": spec.mode,
        "n_crop_rows": sum(1 for r in rows if r.mode == "crop"),
        "n_original_rows": sum(1 for r in rows if r.mode == "original"),
        "by_type": dict(sorted(Counter(f"ET{r.event_type}" for r in rows).items())),
        "by_source": dict(sorted(Counter(r.source for r in rows).items())),
        "months": dict(sorted(Counter(r.date[:7] for r in rows).items())),
        "conf_bands": dict(
            sorted(Counter(r.conf_band for r in rows if r.conf_band).items())
        ),
        "code_bands": dict(sorted(Counter(r.code_band for r in rows).items())),
        "llm_agree_rows": sum(1 for r in rows if r.llm_agree),
        "skipped_no_objects": skipped_no_objects,
    }
    return rows, stats


PER_GROUP_CAP = 4
"""(日期,设备,3h 桶) 同组事件上限.

mobile 相机为车载移动巡查(实测 8 台, 单日单机可爆发 ~1k 事件): 同组内是巡查
沿途不同目标而非同目标重复拍摄, cap=2 会把 ET2 容量压到 283 事件(2026-09-25
实测); cap=4 + 下游 SemDeDup 语义去重(真重复在 dedup 链折叠)双保险.
"""


@app.command()
def main(
    event_root: Path = Path("/var/howell/s4/v0.4/event/mobile"),
    out_dir: Path = Path("/mnt/data/s4det/p0-cluster"),
    seed: int = 42,
    rescan: Annotated[
        bool, typer.Option(help="忽略已有 event_index.jsonl 强制重扫")
    ] = False,
) -> None:
    """扫描 + 抽样 + 落盘(manifest / 索引 / 统计 / 批指纹); 已有索引时直接复用."""
    ledger = LEDGER
    index_path = out_dir / "event_index.jsonl"
    quar: list[QuarantineEntry] = []
    llm_fail = 0
    dup_dropped = (0, 0)
    if index_path.is_file() and not rescan:
        recs = [EventRec(**r) for r in read_jsonl(index_path)]
        logger.info(
            "复用已有索引 {} ({} 事件, --rescan 强制重扫)", index_path, len(recs)
        )
    else:
        recs, quar, llm_fail, dup_dropped = scan_root(event_root)
        logger.info(
            "索引事件 {} | 弃置 {} | llm 解析失败 {} | 平台重复 Id: 全等去重 {} 冲突 {}",
            len(recs),
            len(quar),
            llm_fail,
            dup_dropped[0],
            dup_dropped[1],
        )
        write_jsonl(index_path, recs)
        append_quarantine(ledger, quar)

    evs_by_family = {
        name: [e for e in recs if e.event_type in spec.event_types]
        for name, spec in FAMILY_SPECS.items()
    }
    all_stats: dict[str, object] = {
        "batch": BATCH_ID,
        "source_root": str(event_root),
        "scanned_events": len(recs),
        "event_json_or_image_quarantined": len(quar),
        "llm_parse_fail_events": llm_fail,
        "platform_dup_object_ident_dropped": dup_dropped[0],
        "platform_dup_object_conflict_kept": dup_dropped[1],
    }
    fam_stats: dict[str, object] = {}
    for name, spec in FAMILY_SPECS.items():
        fam_dir = out_dir / name
        rows, stats = build_family(
            spec, evs_by_family[name], recs, fam_dir, seed, Counter()
        )
        write_jsonl(fam_dir / "manifest.jsonl", rows)
        digest = hashlib.sha256((fam_dir / "manifest.jsonl").read_bytes()).hexdigest()
        (fam_dir / "run_config.json").write_bytes(
            orjson.dumps(
                {
                    "batch": BATCH_ID,
                    "family": name,
                    "manifest_sha256": digest,
                    "n_rows": len(rows),
                    "seed": seed,
                    "source_root": str(event_root),
                    "script": "a_manifest.py",
                },
                option=orjson.OPT_INDENT_2,
            )
        )
        fam_stats[name] = stats
        logger.info("[{}] manifest {} 行 | {}", name, len(rows), stats)
    all_stats["families"] = fam_stats
    (out_dir / "sampling_stats.json").write_bytes(
        orjson.dumps(all_stats, option=orjson.OPT_INDENT_2)
    )
    logger.info("全部 manifest 落盘 {}", out_dir)


if __name__ == "__main__":
    app()
