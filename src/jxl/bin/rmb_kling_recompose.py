#!/usr/bin/env python3
"""可灵图生图重绘：钱币参考图 + 场景prompt → 生成"纸币散落台面"合成图。

已验证可用（请求被受理）：端点 https://api-beijing.klingai.com/v1/images/generations、
Bearer key、model kling-v3-ai-image。
⚠️ 阻塞：账户余额不足(code 1102)，等充值后运行。
⚠️ 待实测：成功响应结构(task_id 字段)、查询端点、结果图 URL 字段——本工具按官方
   ComfyUI-KLingAI-API 源码 + 阿里云百炼文档的最佳推断编写，充值后需用真实响应调通
   （可能微调 _extract_task_id / _poll 路径 / _extract_images / 多图参考字段）。

异步：POST 建任务 → 轮询 GET → 下载。KLING_API_KEY 从环境变量读，绝不硬编码。
"""
from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import random
import time
from pathlib import Path

import httpx
import typer

app = typer.Typer(add_completion=False, help="可灵图生图重绘：钱币散落台面合成图。")

BASE = "https://api-beijing.klingai.com/v1"
POLL_INTERVAL = 6.0
POLL_TIMEOUT = 420.0
DENOMS = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]

# 程序化 prompt 维度池：台面 × 杂物 × 光照 × 视角，笛卡尔积生成数千独特组合
SURFACES = [
    "深色木质办公桌", "浅色木质办公桌", "银行柜台浅灰色人造石台面", "银行柜台深灰色石材台面",
    "课桌浅木纹桌面", "窗台灰白色石材", "餐桌大理石纹台面", "收银台银色金属台面",
    "会议桌长条木桌面", "茶几玻璃台面", "地面米色瓷砖", "地面木地板",
    "深色皮革沙发扶手", "浅色瓷砖飘窗台",
]
# 票据/纸张类：与纸币最像的硬负样本，每图加权必含 1-2 种
PAPER_CLUTTER = ["几张票据", "一张发票", "一张收据", "一封信件", "几张便签纸", "报纸一角", "打印的表格纸", "几张信笺纸", "一个信封"]
HARD_CLUTTER = ["一部手机", "几本书", "一本笔记本", "一支签字笔", "一个文件夹", "一个水杯", "一串钥匙",
                "一副眼镜", "一个计算器", "一枚印章", "一个钱包", "一只手表", "一个马克杯", "一支钢笔"]
LIGHTING = ["自然日光", "柔和室内光", "暖色调室内光", "侧面自然光", "明亮顶光", "窗边漫射光"]
VIEWS = ["俯视", "略侧的俯视", "正俯视"]


def make_prompt(num_notes: int, rng: random.Random) -> str:
    """程序化生成场景 prompt。~15% 概率手持子集，其余桌面散放+杂物+光照视角随机组合。"""
    if rng.random() < 0.15:  # 手持子集（少数，纸币占画面约1/2 + 手部遮挡）
        return (f"俯视视角，一只手部分握住一张人民币纸币拿在桌面上方，纸币可见约一半，"
                f"手指自然遮挡一角，{rng.choice(LIGHTING)}，画面中除该纸币外不要出现任何其他纸币或卡片状印刷物")
    surface = rng.choice(SURFACES)
    paper = rng.sample(PAPER_CLUTTER, min(rng.randint(1, 2), len(PAPER_CLUTTER)))
    hard = rng.sample(HARD_CLUTTER, min(rng.randint(1, 3), len(HARD_CLUTTER)))
    clutter = paper + hard
    rng.shuffle(clutter)
    cn = {2: "两", 3: "三"}.get(num_notes, "几")
    return (f"{rng.choice(VIEWS)}视角，{cn}张不同面额的人民币纸币自然散落在{surface}上，"
            f"{rng.choice(LIGHTING)}，桌面零散有{'、'.join(clutter)}，"
            f"画面中除指定纸币外不要出现任何其他纸币或卡片状印刷物")


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def _extract_task_id(resp: dict) -> str | None:
    """⚠️ 待实测：成功响应里 task_id 的字段路径，按官方/阿里云多种格式尝试。"""
    for path in (("data", "task_id"), ("data", "id"), ("output", "task_id"), ("task_id",), ("id",)):
        cur: object = resp
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and cur:
            return cur
    return None


async def _submit(client: httpx.AsyncClient, key: str, payload: dict) -> dict:
    r = await client.post(f"{BASE}/images/generations", headers=_headers(key), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


async def _poll(client: httpx.AsyncClient, key: str, task_id: str) -> dict:
    """⚠️ 待实测查询端点与状态字段；官方 image_generator 用 GET /v1/images/generations。"""
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        r = await client.get(f"{BASE}/images/generations/{task_id}", headers=_headers(key), timeout=30)
        data = r.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected poll response type {type(data)}: {str(data)[:200]}")
        status = str(data.get("data", {}).get("task_status")
                     or data.get("output", {}).get("task_status")
                     or data.get("task_status") or "").upper()
        if status in ("SUCCEEDED", "SUCCESS", "SUCCEED"):
            return data
        if status in ("FAILED", "FAIL", "ERROR"):
            raise RuntimeError(f"kling task {task_id} failed: {data}")
        await asyncio.sleep(POLL_INTERVAL)
    raise TimeoutError(f"kling task {task_id} polling timeout")


def _extract_images(data: dict) -> list[str]:
    """⚠️ 待实测结果图 URL 字段；递归找 http(s) 图链，限深防异常结构栈溢出。"""
    urls: list[str] = []

    def walk(o: object, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("image", "image_url", "url") and isinstance(v, str) and v.startswith("http"):
                    urls.append(v)
                else:
                    walk(v, depth + 1)
        elif isinstance(o, list):
            for x in o:
                walk(x, depth + 1)

    walk(data)
    return urls


async def gen_one(
    client: httpx.AsyncClient, key: str, sem: asyncio.Semaphore,
    refs: list[Path], scene: str, out_dir: Path, fidelity: float,
) -> dict:
    rec: dict = {"refs": [str(r) for r in refs], "scene": scene}
    # 同步文件读取/base64 移出 sem 临界区，委托线程池避免阻塞 event loop
    try:
        b64s = await asyncio.gather(*[asyncio.to_thread(_b64, r) for r in refs])
    except OSError as e:
        rec["error"] = f"image_read: {e}"
        return rec
    # ⚠️ 多图参考字段名未实测，fail fast（No Silent Degradation），充值后实测确认再启用多图
    if len(b64s) > 1:
        rec["error"] = "multi-image reference field unverified; use single ref or confirm API field after recharge"
        return rec
    async with sem:
        payload: dict = {
            "model": "kling-v3-ai-image",  # ⚠️ 图生图 model 名待实测
            "prompt": scene,
            "n": 1,
            "aspect_ratio": "16:9",
            "image_fidelity": fidelity,
            "image": b64s[0],
        }
        try:
            resp = await _submit(client, key, payload)
            code = resp.get("code") if isinstance(resp, dict) else None
            if code in (1102, 401, 403, 402):  # 余额不足/鉴权失败 → 致命，标 fatal
                rec["fatal"] = f"code {code}: {resp.get('message', '')}"
                rec["error"] = rec["fatal"]
                return rec
            task_id = _extract_task_id(resp)
            if not task_id:
                rec["error"] = f"no task_id: {str(resp)[:200]}"
                return rec
            result = await _poll(client, key, task_id)
            imgs = _extract_images(result)
            if not imgs:
                rec["error"] = f"no image url: {str(result)[:200]}"
                return rec
            saved = []
            for i, url in enumerate(imgs):
                ir = await client.get(url, timeout=120)
                ir.raise_for_status()
                fp = out_dir / f"{task_id}_{i}.jpg"
                fp.write_bytes(ir.content)
                saved.append(str(fp))
            rec.update(task_id=task_id, out=saved)
            return rec
        except (httpx.HTTPError, RuntimeError, TimeoutError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            return rec


def pick_combos(sources: Path, pool_per_denom: int) -> list[list[Path]]:
    rng = random.Random(7)
    pool: dict[str, list[Path]] = {}
    for d in DENOMS:
        dd = sources / d
        if dd.exists():
            files = sorted(dd.glob("*.jpg"))[:pool_per_denom]
            if files:
                pool[d] = files
    avail = list(pool)
    combos: list[list[Path]] = []
    for r in (2, 3):
        for c in itertools.combinations(avail, r):
            combos.append([rng.choice(pool[d]) for d in c])
    rng.shuffle(combos)
    return combos


@app.command()
def run(
    sources: Path = typer.Option(Path("assets/sources_selected"), "--sources"),
    out: Path = typer.Option(Path("assets/synth"), "--out"),
    fidelity: float = typer.Option(0.85, "--fidelity"),
    concurrency: int = typer.Option(3, "--concurrency"),
    limit: int = typer.Option(5, "--limit", help="生成多少组（测试用小，0=全部）"),
) -> None:
    if concurrency < 1:
        typer.secho("--concurrency 必须 >= 1。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not 0.0 <= fidelity <= 1.0:
        typer.secho("--fidelity 须在 [0, 1]。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    key = os.environ.get("KLING_API_KEY", "")
    if not key:
        typer.secho("未设 KLING_API_KEY 环境变量。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    combos = pick_combos(sources, 10)
    if limit:
        combos = combos[:limit]
    if not combos:
        typer.secho("无可用参考组合。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    prompt_rng = random.Random(42)
    typer.secho(f"可灵图生图 {len(combos)} 组（需余额充足）→ {out}", fg=typer.colors.CYAN)

    async def main() -> list[dict]:
        async with httpx.AsyncClient() as client:
            tasks = [gen_one(client, key, sem, refs, make_prompt(len(refs), prompt_rng), out, fidelity)
                     for refs in combos]
            return [await c for c in asyncio.as_completed(tasks)]

    results = asyncio.run(main())
    (out / "manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if "out" in r)
    fatal = [r["fatal"] for r in results if "fatal" in r]
    typer.secho(f"完成 {ok}/{len(combos)}（失败 {len(results) - ok}）→ {out / 'manifest.json'}", fg=typer.colors.GREEN)
    if fatal:
        typer.secho(f"⚠️ 致命错误 {len(fatal)} 次（余额不足或鉴权失败）: {fatal[0]}", fg=typer.colors.RED, err=True)


if __name__ == "__main__":
    app()
