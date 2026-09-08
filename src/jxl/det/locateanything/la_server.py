#!/usr/bin/env python3
"""LocateAnything-3B 本地推理服务(FastAPI, 常驻).

运行于独立 la-venv(transformers==4.57.1, 与 jxl 主环境隔离), 由 script/la-serve.sh
启动; 本文件不 import jxl, 依赖仅: torch/transformers(vendored worker) + fastapi.

协议(与 jxl.det.locateanything.client 对应):
  GET  /health → {"status","backend","model","dtype"}
  POST /detect {"query", "image_path"|"image_b64", "max_size"?}
      → {"boxes_px": [[x1,y1,x2,y2],...], "width", "height"}
      boxes_px 为缩放后图像的像素坐标(官方 parse_boxes 输出), 归一化由客户端完成.

约束:
  - 单类单请求(官方多类 query 有 label corruption, issue #69)
  - temperature=0 固定 greedy(官方语义): 官方默认 0.7 为随机采样, 同图+同 query
    重复请求出框不同; 本服务作为标注管线入口(la_relabel/det_mine/la_eval),
    检测框必须可复现
  - max_size 预缩放: RTX 4060 Ti 16GB 现实约束, 大图不缩会爆显存(模型生产上限 2.5K)
  - strict_attn: --attn la_flash 不可用时启动即失败, 不静默回退 SDPA;
    显式传 --attn sdpa 才降级(用户主动选择)
  - NVIDIA License 非商用: 仅研究/评估链路
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, Field

# vendored worker(本目录 vendored/) + batch_utils/kernel_utils(模型 repo 内)入 sys.path
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "vendored"))

from locateanything_worker import LocateAnythingWorker  # noqa: E402

DEFAULT_PORT = 18306
DEFAULT_MAX_SIZE = 1280
ATTN_CHOICES = ("la_flash", "sdpa")

app = FastAPI(title="LocateAnything-3B", description="非商用研究/评估用途")
"""worker/max_size/attn/model 由 main() 挂到 app.state; 端口在模型加载后才绑定."""

_detect_lock = threading.Lock()
"""模型非线程安全, FastAPI 同步端点走线程池, 串行化推理."""


class DetectRequest(BaseModel):
    query: str
    image_path: str | None = None
    image_b64: str | None = None
    max_size: int | None = Field(default=None, ge=0)


def _load_image(req: DetectRequest, default_max_size: int) -> Image.Image:
    """按 path/b64 读图 + RGB + 超限缩放; 坏图抛 HTTPException(400)."""
    if bool(req.image_path) == bool(req.image_b64):
        raise HTTPException(400, "image_path 与 image_b64 必须二选一")
    try:
        if req.image_path:
            img: Image.Image = Image.open(req.image_path)
        else:
            assert req.image_b64 is not None
            img = Image.open(io.BytesIO(base64.b64decode(req.image_b64)))
        img = img.convert("RGB")
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"image_read: {e}") from e
    max_size = req.max_size if req.max_size is not None else default_max_size
    if max_size > 0 and max(img.size) > max_size:
        img.thumbnail((max_size, max_size))  # 保比例, 只缩不放
    return img


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "backend": request.app.state.attn,
        "model": request.app.state.model,
        "dtype": "bfloat16",
    }


@app.post("/detect")
def detect(req: DetectRequest, request: Request) -> dict:
    wk: LocateAnythingWorker = request.app.state.worker
    img = _load_image(req, request.app.state.max_size)
    with _detect_lock:
        # temperature=0 = greedy: 官方默认 0.7 走 Categorical.sample() 随机采样,
        # 出框不可复现(见模块 docstring 约束)
        result = wk.detect(img, [req.query], verbose=False, temperature=0)
        boxes = LocateAnythingWorker.parse_boxes(
            result["answer"], img.width, img.height
        )
    return {
        "boxes_px": [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes],
        "width": img.width,
        "height": img.height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="LocateAnything-3B 模型目录")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--device", default="cuda", help="多卡用 CUDA_VISIBLE_DEVICES 控制"
    )
    parser.add_argument(
        "--attn",
        default="la_flash",
        choices=ATTN_CHOICES,
        help="sdpa 为显式降级(la_flash 不可用时), 默认 strict 失败不静默回退",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=DEFAULT_MAX_SIZE,
        help="送入模型前最长边上限(保比例缩放), 0 表示不缩放",
    )
    args = parser.parse_args()
    if args.max_size < 0:
        parser.error(
            "--max-size 须 >= 0 (0 表示不缩放)"
        )  # 负值静默当 0 = 延迟到推理期 OOM

    sys.path.append(args.model)  # batch_utils/kernel_utils 随模型 repo 分发
    log = logging.getLogger("la_server")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("加载模型 %s (attn=%s, device=%s)...", args.model, args.attn, args.device)
    wk = LocateAnythingWorker(
        model_path=args.model,
        device=args.device,
        use_batch_runtime=True,
        attn=args.attn,
        strict_attn=args.attn != "sdpa",
    )
    app.state.worker = wk
    app.state.attn = args.attn
    app.state.model = args.model
    app.state.max_size = args.max_size
    log.info("就绪 → http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
