#!/home/jiang/py/jxl/.venv/bin/python
"""P2 豆包 vision grounding 复检 det_mine review 集 → YOLO labels。

读 det_mine review/manifest.jsonl，对每图豆包 grounding(text "person")→ bbox，
输出 YOLO labels(cls=0) + _errors.jsonl(错误图, 供重试)。复用 rmb_ground 的
load_backend/parse_detections/encode_image。key 从配置文件读(--cfg), 绝不硬编码。

NOTE: ground_one 与 rmb_ground.ground_one 结构重复(仅 PROMPT 异), 后续可参数化
提取共用函数(S2, 本次未重构)。

用法:
    doubao_relabel <review_manifest.jsonl> <review_images_dir> <out_labels_dir> \
        --cfg <llm.json> --model doubao-seed-2-0-lite-260215
"""
import asyncio
import json
from pathlib import Path
from typing import Annotated

import httpx
import typer

from jxl.bin.rmb_ground import (
    Backend,
    Detection,
    encode_image,
    load_backend,
    parse_detections,
)

app = typer.Typer(add_completion=False, help="P2 豆包 grounding 复检 review 集 → YOLO labels。")

PROMPT = """检测图中所有 person 的位置。严格只输出一个 JSON 数组,不要任何其他文字、不要 markdown。
每个 person 一个对象: {"label":"person","bbox":[x1,y1,x2,y2],"conf":0-1}
- bbox 归一化坐标 [0,1], x1y1=左上, x2y2=右下
- 列出所有可见人(含部分遮挡)
- 若无人, 输出 []
仅输出 JSON 数组。"""


async def ground_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    path: Path,
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[Path, list[Detection], str | None]:
    async with sem:
        try:
            data_url, w, h = await asyncio.to_thread(encode_image, path)
        except OSError as e:
            return path, [], f"image_read: {e}"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}], "temperature": 0.1, "max_tokens": 1024}
        try:
            resp = await client.post(f"{base_url}chat/completions", headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            return path, [], f"http: {e}"
        except (KeyError, IndexError, ValueError) as e:
            return path, [], f"shape: {e}"
        return path, parse_detections(content, w, h), None


@app.command()
def main(  # noqa: PLR0913
    manifest: Annotated[Path, typer.Argument(help="det_mine review/manifest.jsonl")],
    images_dir: Annotated[Path, typer.Argument(help="review 图目录")],
    out_labels: Annotated[Path, typer.Argument(help="输出 YOLO labels 目录")],
    cfg: Annotated[str, typer.Option("--cfg", help="豆包配置文件(base_url+api_key+model)")] = "",
    model: Annotated[str, typer.Option("--model", help="覆盖模型名")] = "doubao-seed-2-0-lite-260215",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 6,
    limit: Annotated[int, typer.Option("--limit", help="只处理前N张(0=全部)")] = 0,
) -> None:
    """读 review manifest → 豆包 grounding → YOLO labels(cls=0) + _errors.jsonl。"""
    if not manifest.is_file():
        typer.secho(f"manifest 不存在: {manifest}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    base_url, api_key, use_model = load_backend(Backend.DOUBAO, model, cfg)
    out_labels.mkdir(parents=True, exist_ok=True)
    recs: list[dict] = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths: list[Path] = [images_dir / r["image"] for r in recs if r.get("image")]
    if limit:
        paths = paths[:limit]
    if not paths:
        typer.secho("无 review 图", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.secho(f"豆包 grounding {len(paths)} review 图 @ {use_model}", fg=typer.colors.CYAN)

    sem = asyncio.Semaphore(concurrency)

    async def run() -> list[tuple[Path, list[Detection], str | None]]:
        results: list[tuple[Path, list[Detection], str | None]] = []
        async with httpx.AsyncClient() as client:
            tasks = [ground_one(client, sem, p, base_url, api_key, use_model) for p in paths]
            done = 0
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                done += 1
                if done % 50 == 0 or done == len(paths):
                    typer.echo(f"  进度 {done}/{len(paths)}")
        return results

    results = asyncio.run(run())
    results.sort(key=lambda r: str(r[0]))
    n_ok = n_empty = n_err = 0
    err_lines: list[str] = []
    for path, dets, err in results:
        lbl = out_labels / (path.stem + ".txt")
        if err:
            n_err += 1
            err_lines.append(json.dumps({"image": path.name, "error": err}, ensure_ascii=False))
            continue  # 错误图: 不写 label, 落 _errors.jsonl 供重试
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            lines.append(f"0 {cx:.6f} {cy:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            n_ok += 1
        else:
            n_empty += 1
    if err_lines:
        (out_labels / "_errors.jsonl").write_text("\n".join(err_lines) + "\n", encoding="utf-8")
    typer.secho(f"有框 {n_ok} | 空(无人) {n_empty} | 错误 {n_err} → {out_labels}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
