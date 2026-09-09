import numpy as np

from jxl.sample_prune import PrunePlan, build_clusters, decide


def _emb(rows: list[list[float]]) -> np.ndarray:
    a = np.array(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_build_clusters_groups_near_duplicates():
    # 3 向量: [1,0] 与 [0.99,0.14] 近重复(cos≈0.99), [0,1] 独立
    clusters = build_clusters(_emb([[1, 0], [0.99, 0.14], [0, 1]]), threshold=0.95)
    clusters = sorted(sorted(c) for c in clusters)
    assert clusters == [[0, 1], [2]]


def test_decide_keeps_representative_and_low_conf():
    # 簇 {0,1,2}(全近重复): conf [0.9, 0.9, 0.3] → 代表=簇中心最近者保留;
    # idx2 conf<0.5 保留; 只有 idx0/1 可删。孤立簇 {3} 永不删。
    plan = decide(_emb([[1, 0], [0.995, 0.1], [0.99, 0.14], [0, 1]]),
                  confs=[0.9, 0.9, 0.3, 0.8], ratio=0.5, keep_conf_below=0.5)
    assert isinstance(plan, PrunePlan)
    assert 3 in plan.keep
    assert 2 in plan.keep          # 低置信保护
    assert set(plan.pool) <= {0, 1}
    assert len(plan.keep) + len(plan.pool) == 4
    for m in plan.meta:
        assert set(m) == {"idx", "cluster_id", "nn_sim", "conf", "removed_round"}


def test_decide_ratio_caps_removal():
    # 5 孤立簇(互不相似), ratio=0.3 → 无簇内冗余可删, pool 为空
    emb = _emb(np.eye(5, dtype=np.float32).tolist())
    plan = decide(emb, confs=[0.9] * 5, ratio=0.3)
    assert plan.pool == ()
