#!/home/jiang/py/jxl/.venv/bin/python
"""VLM grounding：让视觉模型定位图中人民币纸币，输出归一化 bbox + 面额标签。

支持多后端，访问信息从 json 配置文件读取（base_url + api_key），
model 可命令行覆盖（默认按后端的视觉模型）。绝不硬编码 key。

后端：
  qwen35  局域网 qwen3.5-35b-a3b-fp8（OpenAI 兼容多模态）
  doubao  火山方舟 doubao-seed-1-6-vision-250815（视觉）
  qwen    dashscope 兼容模式 qwen-vl-max-latest（视觉，grounding 强）

用途：reannotation 多方案对比。在 rmb_yolo 真实图（有 GT bbox）上跑，
对比 IoU/召回/精度，量化各方案可靠性，再用于可灵图生图生成图的重标注。
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from enum import Enum
from pathlib import Path

import httpx
import typer
from PIL import Image
from pydantic import BaseModel, ValidationError, field_validator

app = typer.Typer(add_completion=False, help="VLM grounding：定位纸币输出 bbox+面额，多后端对比。")

MAX_IMG_SIDE = 1024  # grounding 需较高分辨率保定位精度


class Backend(str, Enum):
    QWEN35 = "qwen35"
    DOUBAO = "doubao"
    QWEN = "qwen"


# 每后端：默认 model（视觉/grounding 强）+ 可选配置文件路径（取 base_url+api_key）
BACKEND_DEFAULTS: dict[Backend, dict[str, str]] = {
    Backend.QWEN35: {
        "model": "qwen3.5-35b-a3b-fp8",
        "url": "http://192.168.18.182:8000/v1",
        "cfg": "",  # 局域网无 key
    },
    Backend.DOUBAO: {
        "model": "doubao-seed-1-6-vision-250815",
        "cfg": "/opt/howell/s4/current/ias/cfg/event/road/llm.json",
    },
    Backend.QWEN: {
        "model": "qwen-vl-max-latest",
        "cfg": "/opt/howell/s4/current/ias/cfg/event/road/llm-qwen.json",
    },
}

PROMPT = """检测图中所有人民币纸币的位置与面额。严格只输出一个 JSON 数组，不要任何其他文字、不要 markdown。

每个纸币一个对象：
{"label":"1yuan|5yuan|10yuan|20yuan|50yuan|100yuan|banknote","bbox":[x1,y1,x2,y2],"conf":0-1}
- bbox 为归一化坐标 [0.0,1.0]，x1y1=左上角，x2y2=右下角
- conf 为定位置信度 [0,1]
- 列出所有可见纸币（含部分遮挡的）
- 若无纸币，输出 []

仅输出 JSON 数组，例如：[{"label":"100yuan","bbox":[0.1,0.2,0.9,0.8],"conf":0.95}]"""

_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


class Detection(BaseModel):
    label: str
    bbox: list[float]
    conf: float

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            msg = "bbox must have 4 floats"
            raise ValueError(msg)
        return [max(0.0, min(1.0, f)) for f in v]


def load_backend(backend: Backend, model_override: str, cfg_override: str) -> tuple[str, str, str]:
    """返回 (base_url, api_key, model)。key 仅从配置文件/环境读，不硬编码。"""
    d = BACKEND_DEFAULTS[backend]
    cfg_path = cfg_override or d.get("cfg", "")
    base_url = d.get("url", "")
    api_key = ""
    if cfg_path:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        base_url = cfg.get("base_url", base_url).rstrip("/") + "/"
        # 兼容 dashscope 兼容模式路径（已是 .../v1/）
        api_key = cfg.get("api_key", "")
    api_key = api_key or ""  # qwen35 局域网无 key
    model = model_override or d["model"]
    if not base_url.endswith("/"):
        base_url += "/"
    return base_url, api_key, model


def encode_image(path: Path) -> tuple[str, int, int]:
    """返回 (data_url, w, h)；w/h 用于把像素坐标归一化。"""
    with Image.open(path) as im:
        img = im.convert("RGB")
        if max(img.size) > MAX_IMG_SIDE:
            img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), img.width, img.height


# 不同模型用的 bbox 字段名不一（qwen 用 bbox_2d，豆包用 bbox），统一兼容
_BBOX_KEYS = ("bbox", "bbox_2d", "box", "rectangle", "box_2d")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_detections(content: str, img_w: int, img_h: int) -> list[Detection]:
    """从模型输出提取检测列表。LLM 输出属不可信外部数据，做格式归一化：
    去 <think>/markdown 围栏，兼容多种 bbox 字段名，像素坐标(>1.5)按图尺寸归一化。
    """
    cleaned = _THINK_RE.sub("", content)
    if "</think>" in cleaned:  # qwen3 思考模式：正文在 </think> 之后，去掉含伪 JSON 的思考过程
        cleaned = cleaned.split("</think>", 1)[1]
    m = _ARR_RE.search(cleaned)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    out: list[Detection] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        bbox = next((item[k] for k in _BBOX_KEYS if k in item), None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            coords = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        if max(coords) > 1.5:  # 像素坐标 → 归一化
            coords = [coords[0] / img_w, coords[1] / img_h, coords[2] / img_w, coords[3] / img_h]
        label = str(item.get("label", item.get("class", "banknote")))
        try:
            conf = float(item.get("conf", item.get("confidence", 1.0)))
        except (TypeError, ValueError):
            conf = 1.0
        try:
            out.append(Detection(label=label, bbox=coords, conf=conf))
        except ValidationError:
            continue
    return out


async def ground_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    path: Path,
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[Path, list[Detection], str | None]:
    """Grounding 单图 → (path, 检测列表, 错误或None)。"""
    async with sem:
        try:
            data_url, img_w, img_h = await asyncio.to_thread(encode_image, path)
        except OSError as e:
            return path, [], f"image_read: {e}"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        try:
            resp = await client.post(f"{base_url}chat/completions", headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            return path, [], f"http: {e}"
        except (KeyError, IndexError, ValueError) as e:
            return path, [], f"shape: {e}"
        return path, parse_detections(content, img_w, img_h), None


def gather_images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in _IMG_EXTS)


@app.command()
def run(
    backend: Backend = typer.Option(..., "--backend", "-b", help="grounding 后端。"),
    src: Path = typer.Option(Path("assets/rmb_yolo/images/valid"), "--src", help="图片目录（递归）。"),
    out: Path = typer.Option(Path("assets/grounding"), "--out", help="输出目录。"),
    model: str = typer.Option("", "--model", help="覆盖默认模型名。"),
    cfg: str = typer.Option("", "--cfg", help="覆盖配置文件路径（取 base_url+api_key）。"),
    concurrency: int = typer.Option(6, "--concurrency", help="并发数。"),
    limit: int = typer.Option(0, "--limit", help="只处理前 N 张（0=全部）。"),
) -> None:
    """逐图 grounding → <out>/<backend>_<srcname>.ndjson。"""
    base_url, api_key, use_model = load_backend(backend, model, cfg)
    imgs = gather_images(src)
    if limit:
        imgs = imgs[:limit]
    if not imgs:
        typer.secho(f"在 {src} 找不到图片。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    out.mkdir(parents=True, exist_ok=True)
    ndjson = out / f"{backend.value}_{src.name}.ndjson"
    typer.secho(f"grounding [{backend.value}] {len(imgs)} 张 @ {use_model}", fg=typer.colors.CYAN)

    sem = asyncio.Semaphore(concurrency)

    async def main() -> list[tuple[Path, list[Detection], str | None]]:
        results: list[tuple[Path, list[Detection], str | None]] = []
        async with httpx.AsyncClient() as client:
            tasks = [ground_one(client, sem, p, base_url, api_key, use_model) for p in imgs]
            done = 0
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                done += 1
                if done % 10 == 0 or done == len(imgs):
                    typer.echo(f"  进度 {done}/{len(imgs)}")
        return results

    results = asyncio.run(main())
    results.sort(key=lambda r: str(r[0]))
    n_ok = 0
    n_det = 0
    with ndjson.open("w", encoding="utf-8") as f:
        for path, dets, err in results:
            rec: dict[str, object] = {"image": str(path.relative_to(src))}
            if err:
                rec["error"] = err
            else:
                n_ok += 1
                n_det += len(dets)
                rec["detections"] = [d.model_dump(mode="json") for d in dets]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    typer.secho(f"\n已写入 {ndjson}", fg=typer.colors.GREEN)
    typer.echo(f"  成功 {n_ok}/{len(imgs)} | 检出 {n_det} 个框 | 户均 {n_det / max(n_ok,1):.1f}")


if __name__ == "__main__":
    app()
