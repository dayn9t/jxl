#!/home/jiang/py/jxl/.venv/bin/python
"""person Re-ID embedding(OSNet-x0_75, MSMT17 权重).

身份级 embedding(同人近、不同人远), 供同人 HDBSCAN 聚类.
绕开 torchreid pip build(依赖链): git clone 源码 sys.path 直接 import osnet 架构.
绕开 GitHub release SSL: 权重从 HF mirror(hf-mirror.com) 下载.

依赖: torchreid 运行时(gdown/tensorboard/h5py/scipy/six/Cython)已装.

典型用法:
    person_reid_embed /path/person_crops /path/reid.npy
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
import typer
from jcx.sys.fs import files_in
from loguru import logger
from PIL import Image
from torchvision import transforms

# 自动 git clone torchreid 源码(不 pip build)
TORCHREID_SRC = Path("/tmp/torchreid_src")
if not TORCHREID_SRC.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/KaiyangZhou/deep-person-reid.git", str(TORCHREID_SRC)],
        check=True,
    )
if str(TORCHREID_SRC) not in sys.path:
    sys.path.insert(0, str(TORCHREID_SRC))

# typer CLI 惯用模式
# ruff: noqa: PLC0415, S603, S607

app = typer.Typer(help="person Re-ID embedding(OSNet)")

IMG_EXT = ".jpg"
"""图像文件扩展名"""
INPUT_HW = (256, 128)
"""OSNet Re-ID 输入尺寸(高, 宽)——行人 reid 标准"""
DEFAULT_WEIGHTS = Path(
    "/home/jiang/ws/sgcc/person/osnet_weights/"
    "osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)


@app.command()
def main(
    src_dir: Annotated[Path, typer.Argument(help="person crop 目录")],
    out_npy: Annotated[Path, typer.Argument(help="输出 reid embedding .npy")],
    weights: Annotated[Path, typer.Option(help="OSNet 权重 .pth")] = DEFAULT_WEIGHTS,
    model_name: Annotated[str, typer.Option(help="OSNet 模型名")] = "osnet_x0_75",
    num_classes: Annotated[int, typer.Option(help="权重类别数(MSMT17 combineall)")] = 4101,
    batch: Annotated[int, typer.Option(help="batch size")] = 64,
    device: Annotated[str, typer.Option(help="设备 cuda/cpu")] = "cuda",
) -> None:
    """提 person crop 的 OSNet Re-ID embedding(512d 身份特征, L2 归一化)."""
    from torchreid import models

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("加载 {} 权重 {} -> {}", model_name, weights.name, dev)
    net = models.build_model(
        name=model_name, num_classes=num_classes, pretrained=False
    ).to(dev).eval()
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    miss, unexp = net.load_state_dict(state, strict=False)
    logger.info("权重 loaded | missing={} unexpected={}", len(miss), len(unexp))

    tf = transforms.Compose(
        [
            transforms.Resize(INPUT_HW),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    files = files_in(src_dir, IMG_EXT)
    logger.info("图像 {} 张 | feat_dim=512", len(files))
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.zeros((len(files), 512), dtype=np.float32)

    @torch.no_grad()
    def embed_batch(imgs: list[Image.Image]) -> np.ndarray:
        tensor = torch.stack([tf(im) for im in imgs]).to(dev)
        feat = net(tensor)  # eval 模式返回 512d feature
        return F.normalize(feat, dim=-1).float().cpu().numpy()

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
    logger.info("完成: {} reid embedding -> {}", done, out_npy)


if __name__ == "__main__":
    app()
