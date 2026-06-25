#!/home/jiang/py/jxl/.venv/bin/python
"""整图 core-set 去重(场景A, 图 person-embedding, 强制减量).

Re-ID(torchreid/insightface) 受阻时的可靠兜底:
图 embedding = 该图所有 person crop 的 DINOv2 均值(person 聚焦, 忽略相同背景),
k-means 在图 embedding 上聚 k 簇, 每 cluster 取离 centroid 最近的图为代表,
强制压到目标数(默认 5700, 即 -60%). 保姿态/场景多样, 不专门合并同人(需 Re-ID).

典型用法:
    samples_core /path/embeddings.npy /path/samples /path/samples_core --target 5700
"""

import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

# typer CLI 惯用模式: 参数校验异常消息豁免噪声规则
# ruff: noqa: TRY003, EM102

app = typer.Typer(help="整图 core-set 去重(图 person-embedding)")

CROP_RE = re.compile(r"(.*)_p\d+\.jpg$")
"""person crop 文件名: <原图stem>_p<序号>.jpg"""


@app.command()
def main(
    embeddings_npy: Annotated[Path, typer.Argument(help="person_crops 的 embeddings.npy")],
    samples_dir: Annotated[Path, typer.Argument(help="samples 目录(images + labels)")],
    out_dir: Annotated[Path, typer.Argument(help="去重输出(samples_core)")],
    target: Annotated[int, typer.Option(help="目标代表数")] = 5700,
) -> None:
    """图 person-embedding k-means core-set, 强制减量到 target."""
    emb = np.load(embeddings_npy).astype(np.float32)
    files = embeddings_npy.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
    assert len(files) == len(emb), f"emb {len(emb)} != files {len(files)}"

    # 图 stem -> [person crop embedding]
    stem2embs: dict[str, list[np.ndarray]] = defaultdict(list)
    for fname, e in zip(files, emb, strict=False):
        m = CROP_RE.match(fname)
        if m:
            stem2embs[m.group(1)].append(e)

    # 图 embedding = person crop 均值(person 聚焦), L2 归一化
    stems = sorted(stem2embs.keys())
    img_emb = np.array([np.mean(stem2embs[s], axis=0) for s in stems], dtype=np.float32)
    img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-9
    logger.info("图 {} 张 | dim {} | 目标 {}", len(stems), img_emb.shape[1], target)

    # k-means core-set(每簇离 centroid 最近的图)
    k = min(target, len(stems))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=2048)
    labels = km.fit_predict(img_emb)
    centers = km.cluster_centers_
    nearest = [None] * k
    nearest_d = [float("inf")] * k
    for i, c in enumerate(labels):
        d = float(np.linalg.norm(img_emb[i] - centers[c]))
        if d < nearest_d[c]:
            nearest_d[c] = d
            nearest[c] = i
    reps = sorted(i for i in nearest if i is not None)

    # 复制代表图 + 标注
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    image_dir = samples_dir / "images"
    label_dir = samples_dir / "labels"
    written = 0
    for i in reps:
        stem = stems[i]
        img = image_dir / f"{stem}.jpg"
        if img.is_file():
            shutil.copy(img, out_img / img.name)
            lbl = label_dir / f"{stem}.txt"
            if lbl.is_file():
                shutil.copy(lbl, out_lbl / lbl.name)
            written += 1

    reduction = 1 - written / max(len(stems), 1)
    logger.info(
        "完成: {} 图 -> {} 代表(-{:.0%}) -> {}",
        len(stems),
        written,
        reduction,
        out_dir,
    )


if __name__ == "__main__":
    app()
