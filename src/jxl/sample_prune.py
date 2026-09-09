"""训练样本簇内削减(近重复优先)+隔离池计划. 机制 spec 见 docs/2026-09-09-训练样本量控制机制设计.md."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PrunePlan:
    keep: tuple[int, ...]
    pool: tuple[int, ...]
    meta: tuple[dict, ...]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def build_clusters(emb: np.ndarray, threshold: float) -> list[list[int]]:
    """cos >= threshold 的样本并查集聚簇(同 dedup_sem 语义)."""
    sim = emb @ emb.T
    uf = _UnionFind(len(emb))
    for i in range(len(emb)):
        for j in range(i + 1, len(emb)):
            if sim[i, j] >= threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(emb)):
        groups.setdefault(uf.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def decide(emb: np.ndarray, confs: list[float], ratio: float,
           keep_conf_below: float = 0.5) -> PrunePlan:
    """簇内削减决策: 主键近重复优先, 次键高置信; 保代表/低置信/孤立簇."""
    clusters = build_clusters(emb, threshold=0.90)  # spec §2.1: 比 0.95 宽的冗余带
    sim = emb @ emb.T
    protected: set[int] = set()
    candidates: list[tuple[float, float, int, int]] = []  # (nn_sim, conf, cluster_id, idx)
    for cid, members in enumerate(clusters):
        if len(members) == 1:
            protected.add(members[0])
            continue
        center = emb[members].mean(axis=0)
        center = center / np.linalg.norm(center)
        rep = max(members, key=lambda i: float(emb[i] @ center))
        protected.add(rep)
        for i in members:
            if confs[i] < keep_conf_below:
                protected.add(i)
            elif i != rep:
                nn = max(float(sim[i, j]) for j in members if j != i)
                candidates.append((nn, confs[i], cid, i))
    n_pool = min(len(candidates), int(len(emb) * ratio))
    candidates.sort(key=lambda t: (-t[0], -t[1]))  # 近重复优先, 同近重复删高置信
    pool = sorted(i for *_, i in candidates[:n_pool])
    keep = tuple(sorted(set(range(len(emb))) - set(pool)))
    meta = tuple({"idx": i, "cluster_id": cid, "nn_sim": nn, "conf": c, "removed_round": 0}
                 for nn, c, cid, i in candidates[:n_pool])
    return PrunePlan(keep=keep, pool=tuple(pool), meta=meta)
