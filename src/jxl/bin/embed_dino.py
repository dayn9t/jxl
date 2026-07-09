#!/home/jiang/py/jxl/.venv/bin/python
"""DINOv2 embedding 提取(前景 crop, SSL 前景特征).

对目录下所有图像提取 DINOv2(自监督) embedding, L2 归一化,
存 .npy + 同名 .txt(文件名列表, 顺序对齐). 用于后续 SemDeDup/HDBSCAN 去重.
SSL 特征对前景细粒度远优于 ImageNet supervised(SemDeDup 原论文强调).

模型走 HuggingFace(facebook/dinov2-small 等价 vits14, 384d), 绕开 GitHub release 限速.

典型用法:
    embed_dino /path/to/crops /path/to/embeddings.npy
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
import typer
from jcx.sys.fs import files_in
from loguru import logger
from PIL import Image

# typer CLI 惯用模式: 参数校验异常消息豁免噪声规则

app = typer.Typer(help="DINOv2 embedding 提取(SSL 前景特征, HuggingFace)")

IMG_EXT = ".jpg"
"""图像文件扩展名"""


@app.command()
def main(
    src_dir: Annotated[Path, typer.Argument(help="图像目录")],
    out_npy: Annotated[Path, typer.Argument(help="输出 embedding .npy")],
    model: Annotated[str, typer.Option(help="HF DINOv2 模型名")] = "facebook/dinov2-small",
    batch: Annotated[int, typer.Option(help="batch size")] = 64,
    device: Annotated[str, typer.Option(help="设备 cuda/cpu")] = "cuda",
) -> None:
    """提取目录下所有图像的 DINOv2 embedding(L2 归一化)."""
    from transformers import AutoImageProcessor, AutoModel

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("加载 {} 到 {}", model, dev)
    processor = AutoImageProcessor.from_pretrained(model)
    net = AutoModel.from_pretrained(model).to(dev).eval()
    embed_dim = net.config.hidden_size

    files = files_in(src_dir, IMG_EXT)
    logger.info("图像 {} 张 | embed_dim={}", len(files), embed_dim)

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.zeros((len(files), embed_dim), dtype=np.float32)

    @torch.no_grad()
    def embed_batch(imgs: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=imgs, return_tensors="pt").to(dev)
        outputs = net(**inputs)
        cls = outputs.last_hidden_state[:, 0]  # CLS token (N, embed_dim)
        return F.normalize(cls, dim=-1).float().cpu().numpy()

    buf: list[Image.Image] = []
    buf_idx: list[int] = []
    done = 0
    for i, f in enumerate(files):
        try:
            buf.append(Image.open(f).convert("RGB"))
            buf_idx.append(i)
        except OSError as e:
            logger.warning("读取失败 {}: {}", f.name, e)
        if len(buf) >= batch:
            emb = embed_batch(buf)
            for j, e in zip(buf_idx, emb, strict=False):
                embeddings[j] = e
            done += len(buf)
            buf.clear()
            buf_idx.clear()
            if done % (batch * 20) == 0:
                logger.info("进度 {}/{}", done, len(files))
    if buf:
        emb = embed_batch(buf)
        for j, e in zip(buf_idx, emb, strict=False):
            embeddings[j] = e
        done += len(buf)

    np.save(out_npy, embeddings)
    out_npy.with_suffix(".txt").write_text(
        "\n".join(f.name for f in files), encoding="utf-8"
    )
    logger.info("完成: {} embedding -> {}", done, out_npy)


if __name__ == "__main__":
    app()
