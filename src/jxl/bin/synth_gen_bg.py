#!/usr/bin/env python3
"""可灵文生图批量生成无钱币桌面背景，供 Copy-Paste 合成。

复用 kling_recompose 的可灵 API 调用与维度池。文生图（无参考图）已验证可用。
背景 prompt：空桌面 + 杂物（含票据硬负样本）+ 明确"不要出现纸币"（防可灵画假钱污染）。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

import httpx
import typer

from jxl.bin.rmb_kling_recompose import (  # noqa: E402
    LIGHTING,
    SURFACES,
    _extract_images,
    _poll,
    _submit,
)

app = typer.Typer(add_completion=False, help="可灵文生图批量生无钱币桌面背景。")


# 明确的非纸币办公用品（避免"纸张/方块"被可灵当纸币画）
DESK_OBJECTS = ["一个笔筒", "一个订书机", "一杯咖啡", "一盆绿植", "一把键盘", "一个鼠标",
                "一本合上的书", "一个鼠标垫", "一盏小台灯", "一个保温杯", "一副耳机", "一个玻璃杯"]


def make_background_prompt(rng: random.Random) -> str:
    """干净办公桌面(明确非纸币物体) + 镜头朝下±30° + 监控画质 + 开头结尾双重禁纸币。"""
    surface = rng.choice(SURFACES)
    objs = rng.sample(DESK_OBJECTS, min(rng.randint(2, 3), len(DESK_OBJECTS)))
    view = rng.choice(["正俯视", "略侧的俯视"])  # 镜头朝下，与垂直夹角<30°
    return (f"一张完全没有任何钱币的干净办公桌面。{view}视角(摄像机镜头朝下，与垂直方向夹角小于30度)，"
            f"{surface}，{rng.choice(LIGHTING)}，监控摄像头画质，"
            f"桌面上只零散放着{'、'.join(objs)}等办公用品。"
            f"严禁出现任何钱、纸币、钞票、硬币、人民币或外币，画面里绝对不能有钱。")


async def gen_bg(client: httpx.AsyncClient, key: str, sem: asyncio.Semaphore,
                 prompt: str, out_dir: Path, idx: int) -> dict:
    async with sem:
        rec: dict = {"idx": idx, "prompt": prompt}
        payload = {"model": "kling-v3-ai-image", "prompt": prompt, "n": 1, "aspect_ratio": "16:9"}
        try:
            resp = await _submit(client, key, payload)
            tid = resp.get("data", {}).get("task_id") if isinstance(resp, dict) else None
            if not tid:
                rec["error"] = f"no task_id: {str(resp)[:150]}"
                return rec
            result = await _poll(client, key, tid)
            imgs = _extract_images(result)
            if not imgs:
                rec["error"] = "no image url"
                return rec
            r = await client.get(imgs[0], timeout=120)
            r.raise_for_status()
            fp = out_dir / f"bg_{idx:04d}.jpg"
            fp.write_bytes(r.content)
            rec["out"] = str(fp)
            return rec
        except (httpx.HTTPError, RuntimeError, TimeoutError, KeyError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            return rec


@app.command()
def run(
    out: Path = typer.Option(Path("assets/backgrounds"), "--out"),
    n: int = typer.Option(20, "--n", help="生成多少张背景。"),
    concurrency: int = typer.Option(4, "--concurrency"),
    seed: int = typer.Option(7, "--seed"),
) -> None:
    key = os.environ.get("KLING_API_KEY", "")
    if not key:
        typer.secho("未设 KLING_API_KEY 环境变量。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    prompts = [make_background_prompt(rng) for _ in range(n)]
    sem = asyncio.Semaphore(concurrency)
    typer.secho(f"生成 {n} 张背景 → {out}", fg=typer.colors.CYAN)

    async def main() -> list[dict]:
        results: list[dict] = []
        async with httpx.AsyncClient() as client:
            tasks = [gen_bg(client, key, sem, prompts[i], out, i) for i in range(n)]
            done = 0
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                done += 1
                if done % 5 == 0 or done == n:
                    typer.echo(f"  进度 {done}/{n}")
        return results

    results = asyncio.run(main())
    (out / "manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if "out" in r)
    typer.secho(f"完成 {ok}/{n} → {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
