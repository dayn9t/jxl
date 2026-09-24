#!/usr/bin/env python3
"""s4det 任务书 A ④——dedup 代表集上的形态聚类(阈值曲线存证 + k-means 主档簇表).

输入(先跑 embed_dino → dedup_sem):
    {fam}/embeddings.npy(+.txt)          DINOv2 embedding(L2 归一化)
    {fam}/dedup/sem_cluster_map.npy      SemDeDup 近重复簇 id(原 crops 顺序)
    {fam}/dedup/sem_cluster_files.txt    对应文件名表

产出:
    {fam}/threshold_curve.json           任务书三档(0.80/0.85/0.90)余弦单链簇数曲线(存证)
    {fam}/kmeans_curve.json              k∈{8,12,16,24,32} 簇大小曲线
    {fam}/clusters.json                  主档 k(默认 16)簇-成员-代表 + 码分档直方

方法依据: 任务书预设的余弦单链阈值法在紧裁片 DINOv2 特征上两端失效——
0.50 链成巨簇 / 0.85 全碎(2026-09-25 实测, threshold_curve.json 为证),
主簇表改用 dedup_sem stage-2 同款 MiniBatchKMeans(L2 归一化下欧氏 ≡ 余弦).
详见 tasks/A-execution.md.

规模语义: size_reps = 去近重复后成员数; size_members = 含近重复帧的原始 crop 张数.

典型用法:
    python a_cluster.py /mnt/data/s4det/p0-cluster/vehicle [--k 16]
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import faiss
import numpy as np
import orjson
import typer
from a_common import read_jsonl
from loguru import logger

from jxl.bin.dedup_sem import UnionFind

app = typer.Typer(
    add_completion=False, help="任务书 A: 形态聚类(三档阈值曲线 + 主档簇表)."
)

EMBED_MODEL = "facebook/dinov2-small"
"""embedding 模型(DINOv3 为 HF gated repo 无法拉取, 2026-09-25 记录于 A-execution.md)."""


def _load_repset(fam_dir: Path) -> tuple[np.ndarray, list[str], list[int]]:
    """加载 embedding + sem 去重映射; 返回 (代表 embedding, 代表文件, 代表→近重复张数)."""
    emb = np.load(fam_dir / "embeddings.npy").astype(np.float32)
    files = (fam_dir / "embeddings.txt").read_text(encoding="utf-8").splitlines()
    sem_map = np.load(fam_dir / "dedup" / "sem_cluster_map.npy")
    dedup_files = (
        (fam_dir / "dedup" / "sem_cluster_files.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if files != dedup_files or len(files) != len(emb) or len(sem_map) != len(emb):
        raise ValueError(f"embedding/sem_map 文件表不对齐: {fam_dir}")
    sem_sizes: Counter = Counter(int(s) for s in sem_map)
    reps: list[int] = []
    seen: set[int] = set()
    for i, s in enumerate(sem_map):
        sid = int(s)
        if sid not in seen:
            seen.add(sid)
            reps.append(i)
    return (
        emb[reps],
        [files[i] for i in reps],
        [sem_sizes[int(sem_map[i])] for i in reps],
    )


def _components(
    emb: np.ndarray, threshold: float, k_neighbors: int = 50
) -> list[list[int]]:
    """Faiss 内积(=余弦, 已归一化)近邻 + 并查集 → 连通分量(每分量 = 索引表)."""
    n = emb.shape[0]
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    sims, nbrs = index.search(emb, k=min(n, k_neighbors))
    uf = UnionFind(n)
    for i in range(n):
        for j, s in zip(nbrs[i], sims[i], strict=False):
            if j >= 0 and j != i and s >= threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def _to_clusters(
    rep_emb: np.ndarray,
    rep_files: list[str],
    rep_dup_counts: list[int],
    band_of: dict[str, str],
    comps: list[list[int]],
) -> list[dict[str, object]]:
    """分量 → 簇表(代表=最靠近簇均值者; 按含近重复总张数降序编 id)."""
    clusters_out: list[dict[str, object]] = []
    for comp in comps:
        sub = rep_emb[comp]
        mean = sub.mean(axis=0)
        mean /= max(np.linalg.norm(mean), 1e-9)
        rep_pos = comp[int(np.argmax(sub @ mean))]
        bands = Counter(band_of.get(f, "none") for f in (rep_files[i] for i in comp))
        clusters_out.append(
            {
                "members": sorted(rep_files[i] for i in comp),
                "size_reps": len(comp),
                "size_members": int(sum(rep_dup_counts[i] for i in comp)),
                "rep": rep_files[rep_pos],
                "rep_code_band": band_of.get(rep_files[rep_pos], "none"),
                "code_hist": dict(sorted(bands.items(), key=lambda kv: -kv[1])),
            }
        )
    clusters_out.sort(key=lambda c: (-int(c["size_members"]), str(c["rep"])))  # type: ignore[arg-type]
    for k, c in enumerate(clusters_out, 1):
        c["cluster_id"] = f"c{k:02d}"
    return clusters_out


def _kmeans_components(
    rep_emb: np.ndarray, k: int, seed: int
) -> tuple[list[list[int]], object]:
    """球面化 k-means(L2 归一化 embedding, 欧氏排序 ≡ 余弦排序; dedup_sem stage-2 同款)."""
    from sklearn.cluster import MiniBatchKMeans

    kk = min(k, rep_emb.shape[0])
    km = MiniBatchKMeans(n_clusters=kk, random_state=seed, n_init=3, batch_size=2048)
    labels = km.fit_predict(rep_emb)
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    return list(groups.values()), km


def cluster_family(
    fam_dir: Path,
    thresholds: list[float],
    primary_k: int,
    kmeans_ks: list[int],
    seed: int = 42,
) -> None:
    """单族聚类: 阈值三档曲线(任务书档位, 存证) + k-means 主档簇表落盘.

    方法依据(A-execution.md 详述): 单链并查集在紧裁片 DINOv2 特征上两端失效
    (0.50 链成巨簇 / 0.85 全碎, 2026-09-25 实测), 阈值档保留为曲线存证;
    主簇表用 dedup_sem stage-2 同款 MiniBatchKMeans(k=主档).
    """
    rep_emb, rep_files, rep_dup_counts = _load_repset(fam_dir)
    logger.info(
        "[{}] 代表集 {} (原 {})", fam_dir.name, len(rep_files), sum(rep_dup_counts)
    )

    curve: dict[str, object] = {}
    for th in thresholds:
        comps = _components(rep_emb, th)
        sizes = sorted((len(c) for c in comps), reverse=True)
        curve[f"{th:.2f}"] = {
            "n_clusters": len(comps),
            "size_hist_top20": sizes[:20],
            "n_singletons": sum(1 for s in sizes if s == 1),
        }
        logger.info(
            "[{}] cos≥{}: {} 簇 | top: {}", fam_dir.name, th, len(comps), sizes[:10]
        )
    (fam_dir / "threshold_curve.json").write_bytes(
        orjson.dumps(
            {
                "family": fam_dir.name,
                "embedding_model": EMBED_MODEL,
                "n_reps": len(rep_files),
                "thresholds": curve,
            },
            option=orjson.OPT_INDENT_2,
        )
    )

    # k-means 曲线 + 主档簇表
    kcurve: dict[str, object] = {}
    comps_by_k: dict[int, list[list[int]]] = {}
    for k in kmeans_ks:
        comps, _km = _kmeans_components(rep_emb, k, seed)
        comps_by_k[k] = comps
        sizes = sorted((len(c) for c in comps), reverse=True)
        kcurve[str(k)] = {"n_clusters": len(comps), "size_hist": sizes}
        logger.info("[{}] k={}: 簇大小 {}", fam_dir.name, k, sizes)
    (fam_dir / "kmeans_curve.json").write_bytes(
        orjson.dumps(
            {
                "family": fam_dir.name,
                "n_reps": len(rep_files),
                "kmeans": kcurve,
            },
            option=orjson.OPT_INDENT_2,
        )
    )
    comps = comps_by_k[min(kmeans_ks, key=lambda k: abs(k - primary_k))]

    band_of = {
        Path(str(r["crop_path"])).name: str(r["code_band"])
        for r in read_jsonl(fam_dir / "manifest.jsonl")
    }
    clusters_out = _to_clusters(rep_emb, rep_files, rep_dup_counts, band_of, comps)
    payload = {
        "family": fam_dir.name,
        "embedding_model": EMBED_MODEL,
        "method": "kmeans",
        "k": min(kmeans_ks, key=lambda k: abs(k - primary_k)),
        "n_clusters": len(clusters_out),
        "n_reps": len(rep_files),
        "n_members_incl_dups": sum(rep_dup_counts),
        "clusters": clusters_out,
    }
    (fam_dir / "clusters.json").write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2)
    )
    logger.info(
        "[{}] 主档 k={}: {} 簇 → clusters.json",
        fam_dir.name,
        payload["k"],
        len(clusters_out),
    )


@app.command()
def main(
    family_dir: Annotated[
        Path, typer.Argument(help="族目录(需已跑 embed_dino+dedup_sem)")
    ],
    thresholds: Annotated[
        str, typer.Option(help="阈值曲线档(存证)")
    ] = "0.80,0.85,0.90",
    k: Annotated[int, typer.Option(help="主档簇数(出簇表)")] = 16,
    ks: Annotated[str, typer.Option(help="k-means 曲线 k 档")] = "8,12,16,24,32",
) -> None:
    """单族聚类(阈值曲线存证 + k-means 主档簇表)."""
    ths = [float(t) for t in thresholds.split(",")]
    cluster_family(family_dir, ths, k, [int(x) for x in ks.split(",")])


if __name__ == "__main__":
    app()
