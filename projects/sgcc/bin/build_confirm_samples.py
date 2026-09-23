#!/usr/bin/env python3
"""向 4 个空确认集导入首航批次样本（源=gencheck 各池，relations parents 为权威上游）.

台账 #35③ 落地的第一半： relations 已钉各集上游（bad_fit/person_verify ←
pospool_20260919、upper_visibility ← upperpool_20260919、reid_pair ← osnet 系），
本脚本按各集语义选样并经服务 API（vlabel.service.ConfirmClient——确认服务唯一
写者，#22/#23）导入：

- bad_fit_review   ← pospool cands_person.jsonl：bbox 贴帧边（边缘截断）的检出，
  裁片外延 10% 并红框标注（bbox-crop-expansion 规范，vlabel.crop）
- person_verify    ← pospool cands_person.jsonl：置信度最低的检出（最值得人工
  复核「框内是否真人」），同样裁片+红框
- upper_visibility ← upperpool removed.jsonl 全量（双门不一致即复审对象），
  removed_images_boxed/ 已备画框图直传
- reid_pair        ← osnet_calib tight/ 配对裁片横向拼接（stem 遵 relations 的
  <uid_a>__<uid_b> 形），取 cos 分数两端（低=异人倾向、高=同人倾向）

幂等：已存在 stem 跳过（先 GET samples）。默认 dry-run 打印计划，--apply 才写。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from vlabel.crop import expand_xyxy  # base layer 迁移：jxl.det.box_utils → vlabel.crop（38889e6）
from vlabel.meta import BorderExtend
from vlabel.service import ConfirmClient

GENCHECK = Path("/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/crop640_persons/gencheck")
PROJECT, GROUP = "sgcc_n001", "r01"
BATCH = 12

_BOX_COLOR = (220, 20, 20)
_DIVIDER = 4


def _jpeg_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


_BE10 = BorderExtend(left=0.10, right=0.10, top=0.10, bottom=0.10)


def _boxed_crop(frame_path: Path, bbox: list[float]) -> tuple[str, bytes]:
    """帧图 → bbox 外延 10% 裁片 + 红框标注（框质量一目了然）。"""
    stem = frame_path.stem
    im = Image.open(frame_path)
    x0, y0, x1, y1 = expand_xyxy(bbox, _BE10, width=im.width, height=im.height)
    crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        (bbox[0] - x0, bbox[1] - y0, bbox[2] - x0, bbox[3] - y0),
        outline=_BOX_COLOR, width=3,
    )
    return stem, _jpeg_bytes(crop)


def plan_bad_fit(limit: int) -> list[tuple[str, bytes]]:
    """贴边（边缘截断）检出优先，其余按置信度升序补足。"""
    rows = [json.loads(l) for l in (GENCHECK / "pospool_20260919/cands_person.jsonl").open()]
    flat = []
    for row in rows:
        frame = GENCHECK / "pospool_20260919/images" / f"{row['image']}"
        if not frame.exists():  # cands 418 行 ⊃ images 291 张——只选出图帧
            continue
        w, h = Image.open(frame).size
        for i, det in enumerate(row["dets"]):
            x0, y0, x1, y1 = det["bbox"]
            truncated = x0 <= 2 or y0 <= 2 or x1 >= w - 2 or y1 >= h - 2
            flat.append((truncated, det["conf"], frame, i, det["bbox"]))
    flat.sort(key=lambda t: (not t[0], t[1]))
    out = []
    for _, _, frame, i, bbox in flat:
        stem, data = _boxed_crop(frame, bbox)
        out.append((f"{stem}_d{i}", data))
        if len(out) >= limit:
            break
    return out


def plan_person_verify(limit: int) -> list[tuple[str, bytes]]:
    """置信度最低的 person 检出（检测器最不确定 → 最值得复核）。"""
    rows = [json.loads(l) for l in (GENCHECK / "pospool_20260919/cands_person.jsonl").open()]
    flat = []
    for row in rows:
        frame = GENCHECK / "pospool_20260919/images" / f"{row['image']}"
        if not frame.exists():
            continue
        for i, det in enumerate(row["dets"]):
            flat.append((det["conf"], frame, i, det["bbox"]))
    flat.sort(key=lambda t: t[1])
    out = []
    for _, frame, i, bbox in flat:
        stem, data = _boxed_crop(frame, bbox)
        out.append((f"{stem}_d{i}", data))
        if len(out) >= limit:
            break
    return out


def plan_upper_visibility(limit: int) -> list[tuple[str, bytes]]:
    """双门不一致 = removed.jsonl 全量；画框图已备（removed_images_boxed/）。"""
    boxed = GENCHECK / "upperpool_20260919/removed_images_boxed"
    out = []
    for line in (GENCHECK / "upperpool_20260919/removed.jsonl").open():
        row = json.loads(line)
        path = boxed / row["image"]
        out.append((Path(row["image"]).stem, path.read_bytes()))
        if len(out) >= limit:
            break
    return out


def plan_reid_pair(limit: int) -> list[tuple[str, bytes]]:
    """cos 两端各半（低=异人倾向、高=同人倾向），a/b 横向拼接 + 分隔线。"""
    cos_rows = [json.loads(l) for l in (GENCHECK / "osnet_calib_20260921/cos.jsonl").open()]
    seen: set[int] = set()
    deduped = []
    for row in cos_rows:  # 同一 idx 挂多个 tag（多模型标定）——按 idx 去重
        if row["idx"] in seen:
            continue
        seen.add(row["idx"])
        deduped.append(row)
    cos_rows = sorted(deduped, key=lambda r: r["cos"])
    half = max(limit // 2, 1)
    chosen = cos_rows[:half] + cos_rows[-half:]
    tight = GENCHECK / "osnet_calib_20260921/tight"
    out = []
    for row in chosen:
        idx = row["idx"]
        pair_id = f"{idx:03d}"
        a = Image.open(tight / f"{pair_id}_a.jpg")
        b = Image.open(tight / f"{pair_id}_b.jpg")
        b = b.resize((int(b.width * a.height / b.height), a.height))
        canvas = Image.new("RGB", (a.width + b.width + _DIVIDER, a.height), (40, 40, 40))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + _DIVIDER, 0))
        out.append((f"{pair_id}_a__{pair_id}_b", _jpeg_bytes(canvas)))
    return out


PLANS = {
    "bad_fit_review": plan_bad_fit,
    "person_verify": plan_person_verify,
    "upper_visibility": plan_upper_visibility,
    "reid_pair": plan_reid_pair,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="http://127.0.0.1:8790")
    parser.add_argument("--as", dest="identity", default="jiang")
    parser.add_argument("--set", choices=sorted(PLANS), action="append", help="默认全部四集")
    parser.add_argument("--limit", type=int, default=BATCH)
    parser.add_argument("--apply", action="store_true", help="缺省 dry-run 只打印计划")
    args = parser.parse_args()

    client = ConfirmClient.login(args.service, args.identity)
    names = args.set or sorted(PLANS)
    rc = 0
    for name in names:
        planned = PLANS[name](args.limit)
        existing = {s.stem for s in client.samples(PROJECT, GROUP, name)}
        todo = [(stem, data) for stem, data in planned if stem not in existing]
        print(f"== {name}: 计划 {len(planned)}，已存在 {len(planned) - len(todo)}，待导入 {len(todo)}")
        for stem, _ in todo:
            print(f"   + {stem}")
        if not args.apply:
            continue
        for stem, data in todo:
            try:
                # ?stem= 须带图片扩展名（服务端存盘名）；样本标识为去扩展名 stem
                client.add_sample(PROJECT, GROUP, name, stem=f"{stem}.jpg", image=data)
            except Exception as error:  # noqa: BLE001 —— 逐件容错继续，末尾统一退出码
                print(f"   ! {stem}: {error}", file=sys.stderr)
                rc = 1
    if not args.apply:
        print("(dry-run：加 --apply 执行写入)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
