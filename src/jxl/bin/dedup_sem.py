#!/usr/bin/env python3
"""前景 crop 去重(SemDeDup 近重复 + k-means core-set 选代表).

DINOv2 是通用视觉模型, 区分不了实例身份(实测 39370 crop 聚成 1 大簇),
但能可靠区分姿态/角度. 故采用研究方案 Stage1+2:
1. SemDeDup: Faiss 余弦近邻(cos>=th) + 并查集, 去近重复姿态/角度, 每簇留 1 代表.
2. k-means core-set: 在 SemDeDup 代表上聚类(k=目标数), 每 cluster 取离 centroid 最近样本,
   保姿态/场景多样性, 压到目标子集.
输出 sem_cluster_map.npy(全 crop -> sem 簇 id, 供场景A整图姿态指纹去重) + 目标数代表 crop.

身份级同实例合并需 Re-ID 模型(OSNet/ArcFace), DINOv2 不支持, 留作可选增强.

典型用法:
    dedup_sem /path/embeddings.npy /path/crops /path/crops_dedup --target 8000
"""

from pathlib import Path
from typing import Annotated

import faiss
import numpy as np
import typer
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

# typer CLI 惯用模式: 参数校验异常消息豁免噪声规则

app = typer.Typer(help="前景 crop 去重(SemDeDup + k-means core-set)")


class UnionFind:
    """并查集(近重复聚类)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


@app.command()
def main(
    embeddings_npy: Annotated[Path, typer.Argument(help="embeddings.npy")],
    crops_dir: Annotated[Path, typer.Argument(help="crops 源目录")],
    out_dir: Annotated[Path, typer.Argument(help="去重后代表输出目录")],
    sem_threshold: Annotated[float, typer.Option(help="SemDeDup 余弦阈值")] = 0.95,
    target: Annotated[int, typer.Option(help="core-set 目标样本数")] = 8000,
    k_neighbors: Annotated[int, typer.Option(help="Faiss 搜索近邻数")] = 50,
) -> None:
    """SemDeDup + k-means core-set, 输出目标数多样代表."""
    import shutil

    emb = np.load(embeddings_npy).astype(np.float32)
    files = embeddings_npy.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
    n = len(files)
    assert len(files) == n, f"embedding {n} != files {len(files)}"
    logger.info("加载 {} embedding, dim={}", n, emb.shape[1])

    # ---- Stage 1: SemDeDup(Faiss 余弦近邻 + 并查集) ----
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    sims, nbrs = index.search(emb, k=min(n, k_neighbors))
    uf = UnionFind(n)
    for i in range(n):
        for j, s in zip(nbrs[i], sims[i], strict=False):
            if j < 0 or j == i:
                continue
            if s >= sem_threshold:
                uf.union(i, j)
    sem_cluster = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    rep_of_sem: dict[int, int] = {}
    for i in range(n):
        rep_of_sem.setdefault(int(sem_cluster[i]), i)
    reps = sorted(rep_of_sem.values())
    logger.info("SemDeDup(cos>={}): {} -> {} 近重复代表", sem_threshold, n, len(reps))

    # ---- Stage 2: k-means core-set(保姿态/场景多样性) ----
    k = min(target, len(reps))
    rep_emb = emb[reps]
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=2048)
    labels = km.fit_predict(rep_emb)
    centers = km.cluster_centers_
    # 每 cluster 取离 centroid 最近的代表
    nearest = [None] * k
    nearest_dist = [float("inf")] * k
    for idx, rep in enumerate(reps):
        c = labels[idx]
        d = float(np.linalg.norm(rep_emb[idx] - centers[c]))
        if d < nearest_dist[c]:
            nearest_dist[c] = d
            nearest[c] = rep
    core_reps = sorted(r for r in nearest if r is not None)

    # ---- 输出: core-set 代表 crop + sem_cluster_map(供场景A) ----
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in core_reps:
        src = crops_dir / files[idx]
        if src.is_file():
            shutil.copy(src, out_dir / src.name)
    np.save(out_dir / "sem_cluster_map.npy", sem_cluster)
    (out_dir / "sem_cluster_files.txt").write_text(
        "\n".join(files), encoding="utf-8"
    )
    logger.info(
        "完成: {} crop -> SemDeDup {} -> core-set {} 代表 | sem_cluster_map 已存(供场景A)",
        n,
        len(reps),
        len(core_reps),
    )


if __name__ == "__main__":
    app()
