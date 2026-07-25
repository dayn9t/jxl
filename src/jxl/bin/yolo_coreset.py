#!/usr/bin/env python3
"""YOLO 数据集图级 core-set 减量(保视觉多样, 强制减量).

读图 DINOv2 embedding(.npy+.txt), k-means 聚 target 簇, 每簇选离 centroid 最近的代表,
删非代表 images+labels. 用于大规模 YOLO 数据集(如 COCO)压到目标数同时保场景/姿态多样.

embedding 用 person_embed 预先提取(图级 DINOv2 384d).

用法:
    person_embed /path/yolo/images /path/embed.npy
    yolo_coreset /path/embed.npy /path/yolo --target 8000
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

# typer CLI 惯用模式
app = typer.Typer(help="YOLO 图级 core-set 减量(DINOv2 多样性采样)")


@app.command()
def main(
    embeddings_npy: Annotated[
        Path, typer.Argument(help="图 DINOv2 embedding .npy(配 .txt 文件名)")
    ],
    dataset_dir: Annotated[Path, typer.Argument(help="YOLO 数据集(images/+labels/)")],
    target: Annotated[int, typer.Option(help="目标代表数")],
) -> None:
    """k-means core-set: 聚 target 簇, 每簇选离 centroid 最近的代表, 删非代表."""
    emb = np.load(embeddings_npy).astype(np.float32)
    files = embeddings_npy.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
    assert len(files) == len(emb), f"emb {len(emb)} != files {len(files)}"
    logger.info("embedding {} 图 | dim {} | 目标 {}", len(files), emb.shape[1], target)

    k = min(target, len(files))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=2048)
    labels = km.fit_predict(emb)
    centers = km.cluster_centers_
    nearest = [None] * k
    nearest_d = [float("inf")] * k
    for i, c in enumerate(labels):
        d = float(np.linalg.norm(emb[i] - centers[c]))
        if d < nearest_d[c]:
            nearest_d[c] = d
            nearest[c] = i
    reps = {files[i].split(".")[0] for i in nearest if i is not None}
    logger.info("代表 {} / {}", len(reps), len(files))

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    del_img = del_lbl = 0
    for f in images_dir.glob("*"):
        if f.stem not in reps:
            f.unlink()
            del_img += 1
    for f in labels_dir.glob("*"):
        if f.stem not in reps:
            f.unlink()
            del_lbl += 1
    remain = len(list(images_dir.glob("*")))
    logger.info(
        "删 {} 图 + {} label | 剩余 {} 图(-{:.0%})",
        del_img,
        del_lbl,
        remain,
        1 - remain / len(files),
    )


if __name__ == "__main__":
    app()
