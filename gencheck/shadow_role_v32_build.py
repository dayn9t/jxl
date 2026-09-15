"""Build shadow-validation crop set for iapx role v3.2 integration.

Selects two-person frames from the iapx n001 sample manifest, crops the
target detection (tight bbox, edge-padded to square, resized to 224), runs
role v3.2 inference locally, and packages crops + expected outputs for the
iapx-side shadow validation.

Contract: ~/cc/py/iapx/docs/jxl-deliveries-2026-09-15.md section 2.
Usage: uv run python gencheck/shadow_role_v32_build.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

MANIFEST = Path("/mnt/data/jiang/ws/iapx/n001/samples/manifest.jsonl")
SAMPLES_ROOT = Path("/mnt/data/jiang/ws/iapx/n001")
MODEL_PT = Path("/opt/howell/iap/v0.9/ias/model/2026-09-14_person_role_n_v32.pt")
OUT_DIR = Path("/mnt/data/jiang/ws/iapx/n001/shadow-role-v32")
CROP_SIZE = 224
JPEG_QUALITY = 95
BATCH = 16
EXPECTED_NAMES = [
    "cleaner",
    "customer",
    "leader",
    "manager",
    "not_person",
    "security",
    "teller",
]
# Staff-form windows (dictionary: o15 guide window; 09-03 seated-staff segment).
STAFF_WINDOWS: dict[tuple[str, int], tuple[str, str]] = {
    ("2026-07-03", 2): ("10-32-00", "10-34-59"),
    ("2026-09-03", 2): ("10-01-00", "10-34-59"),
}
STAFF_WINDOW_QUOTA = 8
# (date, source_id) -> general quota; frames inside staff windows excluded.
GENERAL_QUOTAS: dict[tuple[str, int], int] = {
    ("2026-06-22", 1): 2,
    ("2026-06-22", 2): 2,
    ("2026-06-23", 1): 2,
    ("2026-06-23", 2): 1,
    ("2026-07-01", 1): 2,
    ("2026-07-01", 2): 1,
    ("2026-07-02", 1): 2,
    ("2026-07-02", 2): 1,
    ("2026-07-03", 1): 2,
    ("2026-07-03", 2): 1,
    ("2026-07-04", 1): 1,
    ("2026-07-04", 2): 1,
    ("2026-07-06", 1): 2,
    ("2026-07-06", 2): 1,
    ("2026-07-31", 1): 2,
    ("2026-07-31", 2): 1,
}


@dataclass(frozen=True)
class FramePick:
    manifest_line: int  # 1-based line in manifest.jsonl
    file: str
    source_id: int
    date: str
    wallclock: str
    stem: str
    det_index: int
    box_norm: tuple[float, float, float, float]
    det_confidence: float
    stratum: str  # "window:..." or "general:date/src"


def wallclock_sec(wc: str) -> float:
    h, m, rest = wc.split("-")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def load_eligible() -> list[dict]:
    """Two-person frames with at least one upper_body detection."""
    out: list[dict] = []
    with MANIFEST.open() as f:
        for ln, line in enumerate(f, 1):
            r = json.loads(line)
            if r["n_persons"] >= 2 and any(
                d.get("upper_body") for d in r["detections"]
            ):
                r["_line"] = ln
                out.append(r)
    return out


def in_window(rec: dict) -> bool:
    key = (rec["date"], rec["source_id"])
    win = STAFF_WINDOWS.get(key)
    if win is None:
        return False
    t = wallclock_sec(rec["wallclock"])
    return wallclock_sec(win[0]) <= t <= wallclock_sec(win[1])


def pick_det_index(rec: dict) -> tuple[int, dict] | None:
    """Highest-confidence upper_body detection."""
    best = max(
        (d for d in rec["detections"] if d.get("upper_body")),
        key=lambda d: d["confidence"],
        default=None,
    )
    if best is None:
        return None
    return rec["detections"].index(best), best


def evenly_spaced(picks: list[FramePick], k: int) -> list[FramePick]:
    """Deterministic even coverage of the pool's time span."""
    if k <= 0:
        return []
    if k == 1:
        return [picks[len(picks) // 2]]
    if k >= len(picks):
        return picks
    idx = [round(i * (len(picks) - 1) / (k - 1)) for i in range(k)]
    return [picks[i] for i in idx]


def select_frames(records: list[dict]) -> list[FramePick]:
    windows: dict[tuple[str, int], list[FramePick]] = {}
    general: dict[tuple[str, int], list[FramePick]] = {}
    for rec in records:
        chosen = pick_det_index(rec)
        if chosen is None:
            continue
        di, det = chosen
        pick = FramePick(
            manifest_line=rec["_line"],
            file=rec["file"],
            source_id=rec["source_id"],
            date=rec["date"],
            wallclock=rec["wallclock"],
            stem=rec["stem"],
            det_index=di,
            box_norm=tuple(round(v, 6) for v in det["box_norm"]),
            det_confidence=round(det["confidence"], 6),
            stratum="",
        )
        key = (rec["date"], rec["source_id"])
        bucket = windows if in_window(rec) else general
        bucket.setdefault(key, []).append(pick)

    selected: list[FramePick] = []
    for key, picks in windows.items():
        picks.sort(key=lambda p: p.wallclock)
        chosen = evenly_spaced(picks, STAFF_WINDOW_QUOTA)
        for p in chosen:
            selected.append(
                FramePick(**{**p.__dict__, "stratum": f"window:{key[0]}/src{key[1]}"})
            )
    for key, quota in GENERAL_QUOTAS.items():
        picks = general.get(key, [])
        if len(picks) < quota:
            raise SystemExit(
                f"general pool too small for {key}: {len(picks)} < {quota}"
            )
        picks.sort(key=lambda p: p.wallclock)
        for p in evenly_spaced(picks, quota):
            selected.append(
                FramePick(**{**p.__dict__, "stratum": f"general:{key[0]}/src{key[1]}"})
            )
    selected.sort(key=lambda p: (p.date, p.source_id, p.wallclock))
    return selected


def crop_square(
    img: Image.Image, box_norm: tuple[float, float, float, float]
) -> tuple[Image.Image, tuple[int, int, int, int], tuple[float, float]]:
    """Tight bbox crop, edge-replicated pad to square, resize 224.

    Returns (crop, box_px, pad_frac) where pad_frac is (left, top) padding
    as a fraction of the padded square side.
    """
    w, h = img.size
    x1 = max(0, min(w - 1, round(box_norm[0] * w)))
    y1 = max(0, min(h - 1, round(box_norm[1] * h)))
    x2 = max(x1 + 1, min(w, round(box_norm[2] * w)))
    y2 = max(y1 + 1, min(h, round(box_norm[3] * h)))
    tight = np.asarray(img)[y1:y2, x1:x2]
    ch, cw = tight.shape[:2]
    side = max(cw, ch)
    pad_l = (side - cw) // 2
    pad_t = (side - ch) // 2
    square = np.pad(
        tight,
        ((pad_t, side - ch - pad_t), (pad_l, side - cw - pad_l), (0, 0)),
        mode="edge",
    )
    out = Image.fromarray(square).resize(
        (CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR
    )
    return out, (x1, y1, x2, y2), (round(pad_l / side, 4), round(pad_t / side, 4))


def build_crops(
    picks: list[FramePick], crops_dir: Path
) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    failures: list[str] = []
    seen_names: set[str] = set()
    for p in picks:
        name = f"{p.stem}_{p.det_index}.jpg"
        if name in seen_names:
            failures.append(f"crop name collision: {name} ({p.stratum})")
            continue
        seen_names.add(name)
        src = SAMPLES_ROOT / p.file
        try:
            with Image.open(src) as fh:
                img = fh.convert("RGB")
        except (OSError, ValueError) as e:
            failures.append(f"open failed {src}: {e}")
            continue
        crop, box_px, pad_frac = crop_square(img, p.box_norm)
        crop.save(crops_dir / name, quality=JPEG_QUALITY, subsampling=0)
        rows.append(
            {
                "crop": name,
                "source": {
                    "manifest_line": p.manifest_line,
                    "file": p.file,
                    "source_id": p.source_id,
                    "date": p.date,
                    "wallclock": p.wallclock,
                    "stem": p.stem,
                },
                "stratum": p.stratum,
                "det_index": p.det_index,
                "box_norm": list(p.box_norm),
                "box_px": list(box_px),
                "det_confidence": p.det_confidence,
                "pad_frac": {"left": pad_frac[0], "top": pad_frac[1]},
            }
        )
    return rows, failures


def run_inference(rows: list[dict], crops_dir: Path) -> None:
    model = YOLO(str(MODEL_PT), task="classify")
    name_list = [model.names[i] for i in range(len(model.names))]
    if name_list != EXPECTED_NAMES:
        raise SystemExit(f"model names mismatch: {name_list} != {EXPECTED_NAMES}")
    results = model.predict(
        [str(crops_dir / r["crop"]) for r in rows],
        imgsz=CROP_SIZE,
        batch=BATCH,
        device="cuda",
        verbose=False,
    )
    if len(results) != len(rows):
        raise SystemExit(f"inference count mismatch: {len(results)} != {len(rows)}")
    for r, res in zip(rows, results, strict=True):
        vec = res.probs.data.tolist()
        if len(vec) != len(EXPECTED_NAMES):
            raise SystemExit(f"softmax dim mismatch: {len(vec)}")
        top1_idx = int(np.argmax(vec))
        r["softmax_vec"] = [round(v, 6) for v in vec]
        r["top1"] = EXPECTED_NAMES[top1_idx]
        r["top1_idx"] = top1_idx
        r["top1_conf"] = round(vec[top1_idx], 6)


def iter_stats(rows: list[dict]) -> str:
    dates = sorted({r["source"]["date"] for r in rows})
    sources = sorted({r["source"]["source_id"] for r in rows})
    windows = [r for r in rows if r["stratum"].startswith("window:")]
    top1_counts: dict[str, int] = {}
    for r in rows:
        top1_counts[r["top1"]] = top1_counts.get(r["top1"], 0) + 1
    lines = [
        f"- crops / expected rows: {len(rows)}",
        f"- dates ({len(dates)}): {', '.join(dates)}",
        f"- source_ids: {', '.join(f'src{s}' for s in sources)}",
        f"- staff-window frames: {len(windows)} (07-03 guide / 09-03 seated-staff)",
        f"- top1 distribution: {json.dumps(top1_counts, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


def write_readme(rows: list[dict], md5_full: str) -> None:
    staff5 = "[cleaner, leader, manager, security, teller]"
    text = f"""# shadow-role-v32 —— iapx 影子验证对照集（role v3.2）

## 用途

iapx 按自身部署规则对 role v3.2 跑 20-50 帧影子验证时，直接取 `crops/` 下
crop 图逐帧推理，与 `expected.jsonl` 中 jxl 侧预期输出对照。对接契约见
通知单 `~/cc/py/iapx/docs/jxl-deliveries-2026-09-15.md` §2（v3.2 输出
`[1,7]` softmax 全向量，names 字母序，建议消费全向量而非仅 top1）。

## 权重

- staged `.pt`：`/opt/howell/iap/v0.9/ias/model/2026-09-14_person_role_n_v32.pt`（只读）
- md5：`{md5_full}`（短 `2364d5e7`）
- names（字母序 = 输出向量下标序）：`{json.dumps(EXPECTED_NAMES)}`

## 生成方法

- 选帧：manifest 双人帧（`n_persons>=2` 且至少一个 `upper_body==true` 检测），
  每帧取置信度最高的 upper_body 检测做 crop；覆盖 9 个日期 / src1+src2，
  含 07-03 src2 10:32-10:34 引导窗与 09-03 src2 10:01-10:34 在座 staff 段
  两段工作人员形态高概率窗（各 8 帧），其余 24 帧按日期/源配额当日等时间距采样。
- crop 口径：紧贴 bbox（无 margin）→ 边缘复制（edge-replicate）pad 成正方形
  → BILINEAR resize {CROP_SIZE}×{CROP_SIZE} → JPEG q{JPEG_QUALITY} 4:4:4。
  **crop 已预归一化为 {CROP_SIZE} 正方**：无论消费侧 squash-resize 还是
  resize+center-crop，预处理均退化为恒等，消除两侧预处理口径差。
- expected：ultralytics 8.4.75，`task="classify"`，imgsz {CROP_SIZE}，本机 GPU，
  对 `crops/` 落盘 jpg 原样推理；`softmax_vec` 为 names 序七类全向量，
  `top1`/`top1_conf` 为 top1 类与置信度，`source.manifest_line` 为源帧在
  `samples/manifest.jsonl` 的行号（1-based）。
- 复现：jxl `gencheck/shadow_role_v32_build.py`（等距采样，无随机数，确定性）。

## 对照口径建议

- 逐 crop 比较 top1 与 top1_conf。跨 runtime（本侧 pt/ultralytics vs 贵侧
  部署链路）存在数值差：建议 top1 一致且 `|Δconf| <= 0.02` 判通过；
  softmax 逐维差建议 `<= 0.03`。
- **cleaner / leader 两类已知个体级系统性误差**（通知单 §2：非修复项，
  单帧/聚合均无解）。对照时对 staff 五类 {staff5}
  按「工作人员粗类」宽容：期望 cleaner 而实判 leader/manager 之类的
  staff 内部互换不算失败；staff 粗类 vs customer/not_person 的翻转才计差异。

## 内容统计

{iter_stats(rows)}

_生成时间：2026-09-16，jxl 数据交付 agent_
"""
    (OUT_DIR / "README.md").write_text(text)


def main() -> int:
    md5_full = hashlib.md5(MODEL_PT.read_bytes()).hexdigest()
    if not md5_full.startswith("2364d5e7"):
        raise SystemExit(f"model md5 drifted: {md5_full}")
    records = load_eligible()
    picks = select_frames(records)
    if len(picks) != 40:
        raise SystemExit(f"selection size {len(picks)} != 40")
    crops_dir = OUT_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for old in crops_dir.glob("*.jpg"):
        old.unlink()
    rows, failures = build_crops(picks, crops_dir)
    run_inference(rows, crops_dir)
    with (OUT_DIR / "expected.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_readme(rows, md5_full)
    print(f"crops={len(rows)} failures={len(failures)} out={OUT_DIR}")
    for msg in failures:
        print(f"FAIL: {msg}", file=sys.stderr)
    print(iter_stats(rows))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
