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


def test_plan_command_writes_json(tmp_path, monkeypatch):
    import json as j

    from typer.testing import CliRunner

    from jxl.bin.sample_prune import app

    imgs = tmp_path / "ds" / "images"
    imgs.mkdir(parents=True)
    for s in "abcd":
        (imgs / f"{s}.jpg").write_bytes(b"x")
    np.save(tmp_path / "emb.npy", _emb([[1, 0], [0.99, 0.14], [0, 1], [0.99, 0.12]]))
    # embed 侧车: 缺失即 FATAL(No Silent Degradation), 此处按 stems 排序如实记录
    (tmp_path / "emb.txt").write_text("\n".join(f"{s}.jpg" for s in "abcd"), encoding="utf-8")
    confs = tmp_path / "c.jsonl"
    confs.write_text("".join(j.dumps({"stem": s, "confs": [0.9]}) + "\n" for s in "abcd"))
    out = tmp_path / "plan.json"
    r = CliRunner().invoke(app, ["plan", str(tmp_path / "ds"), "--embeddings", str(tmp_path / "emb.npy"),
                                 "--confs", str(confs), "--ratio", "0.25", "--out", str(out)])
    assert r.exit_code == 0, r.output
    plan = j.loads(out.read_text())
    assert set(plan) == {"stems", "keep", "pool", "meta"}
    assert len(plan["stems"]) == 4
    # b/d 与 a 近重复且高置信 → ratio 0.25 → 恰删 1 个(近重复度更高者)
    assert len(plan["pool"]) == 1
    assert plan["stems"][plan["pool"][0]] in {"b", "d"}
    # stems 侧车: 行序=嵌入行序=stems 排序
    sidecar = tmp_path / "plan.json.stems.txt"
    assert sidecar.read_text(encoding="utf-8").splitlines() == plan["stems"]


def test_plan_requires_embed_sidecar(tmp_path, monkeypatch):
    import json as j

    from typer.testing import CliRunner

    from jxl.bin.sample_prune import app

    imgs = tmp_path / "ds" / "images"
    imgs.mkdir(parents=True)
    for s in "abcd":
        (imgs / f"{s}.jpg").write_bytes(b"x")
    np.save(tmp_path / "emb.npy", _emb([[1, 0], [0.99, 0.14], [0, 1], [0.99, 0.12]]))
    confs = tmp_path / "c.jsonl"
    confs.write_text("".join(j.dumps({"stem": s, "confs": [0.9]}) + "\n" for s in "abcd"))
    # 无 emb.txt 侧车 → 无法对齐行序, 必须 FATAL 而非静默跳过
    r = CliRunner().invoke(app, ["plan", str(tmp_path / "ds"), "--embeddings", str(tmp_path / "emb.npy"),
                                 "--confs", str(confs), "--out", str(tmp_path / "plan.json")])
    assert r.exit_code != 0, r.output
    assert "侧车缺失" in r.output


def test_apply_command_splits_dataset(tmp_path):
    import json as j

    from typer.testing import CliRunner

    from jxl.bin.sample_prune import app

    for sub in ("images", "labels"):
        (tmp_path / "ds" / sub).mkdir(parents=True)
    for s in "abcd":
        (tmp_path / "ds/images" / f"{s}.jpg").write_bytes(b"x")
        (tmp_path / "ds/labels" / f"{s}.txt").write_text("")
    plan = {"stems": ["a", "b", "c", "d"], "keep": [0, 2], "pool": [1, 3],
            "meta": [{"idx": 1, "cluster_id": 0, "nn_sim": 0.99, "conf": 0.9, "removed_round": 0},
                     {"idx": 3, "cluster_id": 0, "nn_sim": 0.98, "conf": 0.9, "removed_round": 0}]}
    pj = tmp_path / "plan.json"
    pj.write_text(j.dumps(plan))
    r = CliRunner().invoke(app, ["apply", str(tmp_path / "ds"), str(pj),
                                 "--out-dir", str(tmp_path / "pruned"), "--pool-dir", str(tmp_path / "pool")])
    assert r.exit_code == 0, r.output
    assert sorted(p.stem for p in (tmp_path / "pruned/images").glob("*.jpg")) == ["a", "c"]
    assert sorted(p.stem for p in (tmp_path / "pool/images").glob("*.jpg")) == ["b", "d"]
    meta = [j.loads(line) for line in (tmp_path / "pool/pool_meta.jsonl").open()]
    assert [m["stem"] for m in meta] == ["b", "d"]
    assert meta[0]["reviews"] == []


def test_plan_rejects_emb_row_order_mismatch(tmp_path, monkeypatch):
    import json as j

    from typer.testing import CliRunner

    from jxl.bin.sample_prune import app

    imgs = tmp_path / "ds" / "images"
    imgs.mkdir(parents=True)
    for s in "abcd":
        (imgs / f"{s}.jpg").write_bytes(b"x")
    # 嵌入行序故意错开(行序=b,a,d,c), 侧车如实记录 → 与 stems 排序不一致必须 FATAL
    np.save(tmp_path / "emb.npy", np.arange(8, dtype=np.float32).reshape(4, 2))
    (tmp_path / "emb.txt").write_text("\n".join(f"{s}.jpg" for s in "badc"), encoding="utf-8")
    confs = tmp_path / "c.jsonl"
    confs.write_text("".join(j.dumps({"stem": s, "confs": [0.9]}) + "\n" for s in "abcd"))
    r = CliRunner().invoke(app, ["plan", str(tmp_path / "ds"), "--embeddings", str(tmp_path / "emb.npy"),
                                 "--confs", str(confs), "--out", str(tmp_path / "plan.json")])
    assert r.exit_code != 0, r.output
    assert "行序" in r.output


def test_pool_review_backflow_and_fuse(tmp_path, monkeypatch):
    import json as j

    from typer.testing import CliRunner

    from jxl.bin import sample_prune as sp

    for s in ("b", "d", "e", "f"):
        (tmp_path / "pool/images").mkdir(parents=True, exist_ok=True)
        (tmp_path / "pool/images" / f"{s}.jpg").write_bytes(b"x")
    meta_rows = [
        {"stem": s, "idx": i, "cluster_id": 0, "nn_sim": 0.99, "conf": 0.9,
         "removed_round": 2 if s == "f" else 0, "reviews": []}
        for i, s in enumerate("bdef")]
    (tmp_path / "pool/pool_meta.jsonl").write_text("".join(j.dumps(m) + "\n" for m in meta_rows))
    monkeypatch.setattr(sp, "_infer_confs", lambda m, d: {"b": 0.5, "d": 0.9, "e": 0.0, "f": 0.4})
    r = CliRunner().invoke(sp.app, ["pool-review", str(tmp_path / "pool"),
                                    "--model", "x.pt", "--out", str(tmp_path / "bf.jsonl")])
    assert r.exit_code == 0, r.output
    bf = {j.loads(line)["stem"]: j.loads(line) for line in (tmp_path / "bf.jsonl").open()}
    assert bf["b"]["backflow"] is True      # 0.9 -> 0.5, 降 0.4 > 0.2
    assert bf["d"]["backflow"] is False     # 持平
    assert bf["e"]["backflow"] is True      # 检出丢失
    assert bf["f"]["reason"] == "fused"     # 熔断, removed_round=2
    meta = {j.loads(line)["stem"]: j.loads(line) for line in (tmp_path / "pool/pool_meta.jsonl").open()}
    assert meta["f"]["removed_round"] == 2  # 熔断不增轮
    assert meta["b"]["removed_round"] == 1
    assert meta["d"]["removed_round"] == 0  # 稳定不回流不增轮(增轮只数回流周期)
