#!/home/jiang/py/jxl/.venv/bin/python
"""从 COCO 下载新背景：dining_table 补集(贴桌面目标) + bench/couch/bed(多样性) → 去重 → resize_960 → backgrounds_960。

dining_table 共 11837 张，coco_dt 已下 2499，取补集。bench/couch/bed 增加场景多样性。
排除 coco_dt 已下载的 image_id（去重）。下载到 coco_dt2(原始名)，resize 成 coco2_bg960_* 入池。
受限网络：as_completed + 并发 5 + timeout 30s 避免限流阻塞。

用法:
  .venv/bin/python tools/gather_coco.py --limit 20   # 测网络
  .venv/bin/python tools/gather_coco.py              # 全量
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx
from PIL import Image

INSTANCES = "/tmp/annotations/instances_train2017.json"
URL = "http://images.cocodataset.org/train2017/{:012d}.jpg"
# (cat_id, 限量)：dining_table 补集贴目标，bench/couch/bed 增多样
TARGETS = [(67, 1500), (15, 500), (63, 500), (65, 500)]
EXISTING = "assets/coco_dt"      # 已下，去重
RAW_OUT = "assets/coco_dt2"      # 新下原始图
BG_OUT = "assets/backgrounds_960"


def collect_ids() -> list[int]:
    d = json.load(open(INSTANCES))
    cat_imgs: dict[int, set[int]] = {}
    for a in d["annotations"]:
        cat_imgs.setdefault(a["category_id"], set()).add(a["image_id"])
    existing = {int(p.stem) for p in Path(EXISTING).glob("*.jpg")}
    chosen: list[int] = []
    seen: set[int] = set()
    for cid, limit in TARGETS:
        avail = sorted(cat_imgs.get(cid, set()) - existing)
        take = [i for i in avail if i not in seen][:limit]
        seen.update(take)
        chosen.extend(take)
        print(f"  cat {cid}: 总{len(cat_imgs.get(cid,()))} 已下{len(existing & cat_imgs.get(cid,set()))} 新取{len(take)}")
    return chosen


async def fetch(client: httpx.AsyncClient, iid: int, raw_dir: Path, sem: asyncio.Semaphore) -> int | None:
    async with sem:
        try:
            r = await client.get(URL.format(iid), timeout=30)
            if r.status_code != 200:
                return None
            (raw_dir / f"{iid:012d}.jpg").write_bytes(r.content)
            return iid
        except (httpx.HTTPError, OSError):
            return None


def to960(raw_dir: Path) -> int:
    n = len(list(Path(BG_OUT).glob("coco2_bg960_*.jpg")))
    ok = 0
    for p in sorted(raw_dir.glob("*.jpg")):
        try:
            im = Image.open(p).convert("RGB")
            W, H = im.size
            if max(W, H) / min(W, H) > 1.6:
                if W >= H:
                    h = W // 2; ov = int(h * 0.1)
                    gs = [im.crop((0, 0, h + ov, H)), im.crop((h - ov, 0, W, H))]
                else:
                    h = H // 2; ov = int(h * 0.1)
                    gs = [im.crop((0, 0, W, h + ov)), im.crop((0, h - ov, W, H))]
            else:
                gs = [im]
            for g in gs:
                g.resize((960, 960), Image.LANCZOS).save(Path(BG_OUT) / f"coco2_bg960_{n:05d}.jpg", quality=88)
                n += 1
            ok += 1
        except OSError:
            continue
    return ok


async def main() -> None:
    ap = argparse.ArgumentParser(description="COCO 新背景下载+resize。")
    ap.add_argument("--limit", type=int, default=0, help="总限量(0=按配比)")
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    ids = collect_ids()
    if args.limit:
        ids = ids[:args.limit]
    print(f"待下载 {len(ids)} 张")

    raw_dir = Path(RAW_OUT); raw_dir.mkdir(exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    ok = 0
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(fetch(client, i, raw_dir, sem)) for i in ids]
        done = 0
        for coro in asyncio.as_completed(tasks):
            if await coro:
                ok += 1
            done += 1
            if done % 200 == 0 or done == len(ids):
                print(f"  下载 {done}/{len(ids)} 成功 {ok}")
    print(f"下载完成 {ok}/{len(ids)} → {raw_dir}")
    nr = to960(raw_dir)
    print(f"resize {nr} 张 → {BG_OUT}/coco2_bg960_*")


if __name__ == "__main__":
    asyncio.run(main())
