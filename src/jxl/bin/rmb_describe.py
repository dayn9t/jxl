#!/usr/bin/env python3
"""VLM 打标：用多模态模型给每张人民币图生成结构化描述，用于筛选"合成素材源"。

读取一个钱币图片目录（默认 bat67），对每张图调用 OpenAI 兼容的多模态端点
（默认局域网 qwen3.5-35b-a3b-fp8），按 12 个维度评估，输出结构化 JSON。
结果写入 <out>/<src_name>.ndjson（每行一图）并打印「面额×适合度」汇总，
供后续筛选阶段挑出高质量、按面额平衡的合成素材源。

配置（环境变量，均非 secret）：
  LLM_BASE_URL  OpenAI 兼容端点（默认局域网 vLLM）
  LLM_MODEL     模型名（默认 qwen3.5-35b-a3b-fp8）
  LLM_API_KEY   可选 Bearer token（局域网内部部署通常不需要）
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from collections import Counter
from enum import StrEnum
from pathlib import Path

import httpx
import typer
from PIL import Image
from pydantic import BaseModel, ValidationError

app = typer.Typer(add_completion=False, help="VLM 打标：给钱币图生成结构化描述用于筛选合成素材源。")

DEFAULT_URL = "http://192.168.18.182:8000/v1"
DEFAULT_MODEL = "qwen3.5-35b-a3b-fp8"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
MAX_IMG_SIDE = 768  # 缩到最长边 768，省 token、加速；钱币纹理判定足够


class Denomination(StrEnum):
    Y1 = "1"
    Y5 = "5"
    Y10 = "10"
    Y20 = "20"
    Y50 = "50"
    Y100 = "100"
    UNKNOWN = "unknown"


class Side(StrEnum):
    FRONT = "front"
    BACK = "back"
    FOLDED = "folded"
    UNKNOWN = "unknown"


class Completeness(StrEnum):
    FULL = "full"
    PARTIAL = "partial"


class ViewAngle(StrEnum):
    FLAT = "flat"
    TILTED = "tilted"
    OBLIQUE = "oblique"


class BgComplexity(StrEnum):
    PLAIN = "plain"
    SIMPLE = "simple"
    COMPLEX = "complex"


class Lighting(StrEnum):
    EVEN = "even"
    SHADOW = "shadow"
    GLARE = "glare"


class Sharpness(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Suitability(StrEnum):
    GOOD = "good"
    OK = "ok"
    BAD = "bad"


class NoteDescription(BaseModel):
    is_real_rmb: bool
    denomination: Denomination
    side: Side
    completeness: Completeness
    view_angle: ViewAngle
    background_complexity: BgComplexity
    has_interference: bool
    lighting: Lighting
    sharpness: Sharpness
    has_artifact: bool
    synthesis_suitability: Suitability
    suitability_reason: str


PROMPT = """你是人民币纸币图像分析专家。判断图中的人民币是否适合作为"图像合成素材源"
（将被抠出后贴到其他背景图上）。严格只输出一个 JSON 对象，不要任何其他文字、不要 markdown 代码块。

按以下维度评估：
- is_real_rmb: 是否真实第五套人民币（排除玩具币/外币/冥币/严重破损/非钱币）
- denomination: "1"|"5"|"10"|"20"|"50"|"100"|"unknown"
- side: "front"(正面)|"back"(背面)|"folded"(折叠)|"unknown"
- completeness: "full"(完整可见)|"partial"(裁切/遮挡)
- view_angle: "flat"(平铺正视,最易抠图)|"tilted"(倾斜)|"oblique"(大角度斜视)
- background_complexity: "plain"(纯色,最易抠)|"simple"|"complex"(杂物多)
- has_interference: 手指/物体是否压在钱币上(true/false)
- lighting: "even"(均匀)|"shadow"(有明显阴影)|"glare"(反光)
- sharpness: "high"|"medium"|"low"
- has_artifact: 是否有水印/马赛克/严重模糊/数字伪影(true/false)
- synthesis_suitability: 综合适合度 "good"(可直接抠)|"ok"(需处理)|"bad"(不可用)
- suitability_reason: 一句话理由

输出 JSON（字段齐全）：
{"is_real_rmb":bool,"denomination":"...","side":"...","completeness":"...",
 "view_angle":"...","background_complexity":"...","has_interference":bool,
 "lighting":"...","sharpness":"...","has_artifact":bool,
 "synthesis_suitability":"...","suitability_reason":"..."}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def encode_image(path: Path) -> str:
    """读图 → 最长边缩到 MAX_IMG_SIDE → JPEG base64 data URL（省 token、加速）。"""
    with Image.open(path) as im:
        img = im.convert("RGB")
        if max(img.size) > MAX_IMG_SIDE:
            img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_json_obj(content: str) -> dict | None:
    """从模型输出中提取首个 JSON 对象；LLM 输出属不可信外部数据，失败返回 None。"""
    m = _JSON_RE.search(content)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def describe_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    path: Path,
    url: str,
    model: str,
    key: str,
) -> tuple[Path, NoteDescription | None, str | None]:
    """打标单图 → (path, 描述或 None, 错误类别或 None)。"""
    async with sem:
        try:
            data_url = await asyncio.to_thread(encode_image, path)
        except OSError as e:
            return path, None, f"image_read: {e}"
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "temperature": 0.1,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = await client.post(f"{url}/chat/completions", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            return path, None, f"http: {e}"
        except (KeyError, IndexError, ValueError) as e:
            return path, None, f"shape: {e}"
        obj = parse_json_obj(content)
        if obj is None:
            return path, None, "no_json"
        try:
            return path, NoteDescription.model_validate(obj), None
        except ValidationError as e:
            return path, None, f"validation: {str(e)[:200]}"


def gather_images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)


def write_ndjson(
    out_path: Path,
    src_root: Path,
    results: list[tuple[Path, NoteDescription | None, str | None]],
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for path, desc, err in results:
            rec: dict[str, object] = {"image": str(path.relative_to(src_root))}
            if desc is not None:
                rec.update(desc.model_dump(mode="json"))
            else:
                rec["error"] = err
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_summary(results: list[tuple[Path, NoteDescription | None, str | None]]) -> None:
    ok = [d for _, d, _ in results if d is not None]
    errs = [e for _, _, e in results if e is not None]
    typer.secho(f"\n共 {len(results)} 张 | 成功 {len(ok)} | 失败 {len(errs)}", fg=typer.colors.CYAN)
    if errs:
        c = Counter(e.split(":", 1)[0] for e in errs)
        typer.echo("  失败类型: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    real = sum(1 for d in ok if d.is_real_rmb)
    typer.echo(f"  真实人民币: {real}/{len(ok)}")
    suit = Counter(d.synthesis_suitability.value for d in ok)
    typer.echo("  适合度: " + ", ".join(f"{k}={suit.get(k, 0)}" for k in ("good", "ok", "bad")))
    by_den: dict[str, Counter] = {}
    for d in ok:
        by_den.setdefault(d.denomination.value, Counter())[d.synthesis_suitability.value] += 1
    typer.secho("  按面额×适合度 (good/ok/bad):", fg=typer.colors.YELLOW)
    for den in ("1", "5", "10", "20", "50", "100", "unknown"):
        if den in by_den:
            c = by_den[den]
            typer.echo(f"    {den}元: good={c.get('good', 0)} ok={c.get('ok', 0)} bad={c.get('bad', 0)}")


@app.command()
def run(
    src: Path = typer.Option(Path("assets/datasets/bat67-rmb-dataset"), "--src", help="钱币图片目录（递归）。"),
    out: Path = typer.Option(Path("assets/descriptions"), "--out", help="输出目录。"),
    url: str = typer.Option(os.environ.get("LLM_BASE_URL", DEFAULT_URL), "--url", help="OpenAI 兼容端点。"),
    model: str = typer.Option(os.environ.get("LLM_MODEL", DEFAULT_MODEL), "--model", help="模型名。"),
    key: str = typer.Option(os.environ.get("LLM_API_KEY", ""), "--key", help="Bearer token（可空）。"),
    concurrency: int = typer.Option(8, "--concurrency", help="并发数。"),
    limit: int = typer.Option(0, "--limit", help="只处理前 N 张（0=全部，测试用）。"),
) -> None:
    """逐图打标 → ndjson + 面额×适合度汇总。"""
    imgs = gather_images(src)
    if limit:
        imgs = imgs[:limit]
    if not imgs:
        typer.secho(f"在 {src} 找不到图片。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    out.mkdir(parents=True, exist_ok=True)
    ndjson = out / f"{src.name}.ndjson"
    typer.secho(f"打标 {len(imgs)} 张 @ {model} ({url})", fg=typer.colors.CYAN)

    sem = asyncio.Semaphore(concurrency)

    async def main() -> list[tuple[Path, NoteDescription | None, str | None]]:
        results: list[tuple[Path, NoteDescription | None, str | None]] = []
        async with httpx.AsyncClient() as client:
            tasks = [describe_one(client, sem, p, url, model, key) for p in imgs]
            for done, coro in enumerate(asyncio.as_completed(tasks), 1):
                results.append(await coro)
                if done % 20 == 0 or done == len(imgs):
                    typer.echo(f"  进度 {done}/{len(imgs)}")
        return results

    results = asyncio.run(main())
    results.sort(key=lambda r: str(r[0]))
    write_ndjson(ndjson, src, results)
    typer.secho(f"\n已写入 {ndjson}", fg=typer.colors.GREEN)
    print_summary(results)


if __name__ == "__main__":
    app()
