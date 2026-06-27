#!/home/jiang/py/jxl/.venv/bin/python
"""可灵钱币变体生成：每张钱币 → 5 变体(陈旧/污渍/折痕/破损/笔迹) + VLM 复检面额(丢弃失真)。

读 sources_selected/{denom}/*.jpg（bat67 原图，白底），每张小图(≥300px) → 可灵图生图变体
→ 豆包复检面额(==原面额？丢弃失真) → 输出到 note_variants/{denom}/。
变体后需用 seg_cut 重新抠图（位置/比例可能变）。
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path

import httpx
import typer
from PIL import Image

from jxl.bin.rmb_ground import Backend, load_backend  # noqa: E402
from jxl.bin.rmb_kling_recompose import _extract_images, _poll, _submit  # noqa: E402

app = typer.Typer(add_completion=False, help="可灵钱币变体生成+面额复检。")

CANON = ["1yuan", "5yuan", "10yuan", "20yuan", "50yuan", "100yuan"]
# 5 种变化（保面额约束写在 prompt）
VARIANTS = [
    ("aged", "一张陈旧褪色泛黄磨损的人民币{d}元纸币，纸面发黄有岁月痕迹，面额数字{d}和图案完全保持不变"),
    ("stained", "一张有污渍和油渍的人民币{d}元纸币，纸面有深色污点，面额数字{d}清晰不变"),
    ("creased", "一张有折痕和褶皱的人民币{d}元纸币，折叠痕迹明显，面额数字{d}清晰不变"),
    ("damaged", "一张有边缘破损和缺角的人民币{d}元纸币，边缘磨损，面额数字{d}清晰不变"),
    ("written", "一张有手写笔迹和盖章的人民币{d}元纸币，纸面有涂写痕迹，面额数字{d}清晰不变"),
]


def to_small_b64(p: Path) -> tuple[str, tuple[int, int]]:
    """读图→缩到最小边≥300且长边≤512→base64（可灵要求≥300px）。"""
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if min(w, h) < 300:
        s = 300 / min(w, h)
        im = im.resize((int(w * s), int(h * s)))
    elif max(w, h) > 512:
        im.thumbnail((512, 512))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), im.size


async def check_denom(client: httpx.AsyncClient, key: str, model: str,
                      img_bytes: bytes, denom: str) -> bool:
    """豆包复检：变体面额 == 原面额？"""
    b64 = base64.b64encode(img_bytes).decode()
    try:
        r = await client.post(
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, timeout=40,
            json={"model": model, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "这张纸币是人民币多少元?只回数字。"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
                "max_tokens": 10, "temperature": 0})
        ans = r.json()["choices"][0]["message"]["content"].strip()
        return denom in ans  # 原面额数字在回答里
    except (KeyError, httpx.HTTPError):
        return False


async def gen_variant(client: httpx.AsyncClient, kkey: str, dkey: str, dmodel: str,
                      src: Path, denom: str, vname: str, vprompt: str, out_dir: Path) -> bool:
    b64, _size = to_small_b64(src)
    prompt = vprompt.format(d=denom.replace("yuan", ""))
    try:
        resp = await _submit(client, kkey, {
            "model": "kling-v3-ai-image", "prompt": prompt, "n": 1,
            "aspect_ratio": "1:1", "image_fidelity": 0.78, "image": b64})
        res = await _poll(client, kkey, resp["data"]["task_id"])
        url = _extract_images(res)[0]
        r = await client.get(url, timeout=120)
        if not await check_denom(client, dkey, dmodel, r.content, denom.replace("yuan", "")):
            return False  # 面额失真，丢弃
        (out_dir / f"{src.stem}_{vname}.jpg").write_bytes(r.content)
        return True
    except (httpx.HTTPError, RuntimeError, TimeoutError, KeyError, IndexError):
        return False


@app.command()
def run(
    src: Path = typer.Option(Path("assets/sources_selected"), "--src"),
    out: Path = typer.Option(Path("assets/note_variants"), "--out"),
    concurrency: int = typer.Option(5, "--concurrency"),
    limit_per_denom: int = typer.Option(0, "--limit", help="每面额取前N张原图(0=全部)"),
) -> None:
    kkey = os.environ.get("KLING_API_KEY", "")
    if not kkey:
        typer.secho("未设 KLING_API_KEY", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    _, dkey, dmodel = load_backend(Backend.DOUBAO, "", "")
    sem = asyncio.Semaphore(concurrency)
    jobs: list[tuple] = []
    for d in CANON:
        sd = src / d
        if not sd.exists():
            continue
        files = sorted(sd.glob("*.jpg"))
        if limit_per_denom:
            files = files[:limit_per_denom]
        odir = out / d
        odir.mkdir(parents=True, exist_ok=True)
        for f in files:
            for vname, vprompt in VARIANTS:
                jobs.append((f, d, vname, vprompt, odir))
    typer.secho(f"计划 {len(jobs)} 变体（{len(CANON)}面额 × {len(VARIANTS)}变化）", fg=typer.colors.CYAN)

    async def main() -> int:
        ok = 0
        async with httpx.AsyncClient() as client:
            async def task(j: tuple[Path, str, str, str, Path]) -> bool:
                async with sem:
                    f, d, vn, vp, od = j
                    return await gen_variant(client, kkey, dkey, dmodel, f, d, vn, vp, od)
            done = 0
            for coro in asyncio.as_completed([task(j) for j in jobs]):
                if await coro:
                    ok += 1
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    typer.echo(f"  进度 {done}/{len(jobs)} 成功 {ok}")
        return ok

    ok = asyncio.run(main())
    typer.secho(f"变体成功 {ok}/{len(jobs)}（丢弃面额失真 {len(jobs)-ok}）→ {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
