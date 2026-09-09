# Sample Prune（簇内削减+隔离池回流）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `jxl.bin.sample_prune`（dino 簇内近重复优先削减 + 隔离池 + 回流评测），并跑 SGCC 三档消融填证据表。

**Architecture:** 纯函数核（聚类/排序/决策）+ plan/apply 两步 CLI（dry-run 闸门，同 surgery.py 模式）+ pool-review 回流子命令。conf 来源=gencheck_infer 的 jsonl 格式，embeddings=既有 DINOv2 npy 格式。

**Tech Stack:** Python 3.12 / typer / numpy / ultralytics（仅 pool-review 推理用）

**Spec:** `docs/2026-09-09-训练样本量控制机制设计.md`（§2 机制、§2.4 实验设计）

## Global Constraints

- 削减单位=帧（image+label 成对整体）；图片永不删，一律 symlink
- 保留规则铁律：簇代表 + conf<0.5 全保 + 孤立簇（成员 1）永不删
- 主排序键=簇内最大近邻余弦相似度（降序=近重复优先删）；次键=conf 降序
- 震荡熔断：同一样本删除→回流循环 ≤2 次，第 3 次永久保留
- 遵循 j-python 规范：完整类型注解、无裸 except、typer CLI 模式（参考 dedup_sem.py）

---

### Task 1: 纯函数核（聚类 + 簇内删除序 + 决策）

**Files:**
- Create: `src/jxl/sample_prune.py`
- Test: `tests/sample_prune_test.py`

**Interfaces:**
- Produces: `build_clusters(emb: np.ndarray, threshold: float) -> list[list[int]]`（索引簇，union-find，同 dedup_sem 语义）；`decide(emb: np.ndarray, confs: list[float], ratio: float, keep_conf_below: float = 0.5) -> PrunePlan`；`PrunePlan`（frozen dataclass：`keep: tuple[int,...]`, `pool: tuple[int,...]`, `meta: tuple[dict,...]`，meta 每条含 `{"idx", "cluster_id", "nn_sim", "conf", "removed_round": 0}`）

- [ ] **Step 1: 写失败测试**

```python
# tests/sample_prune_test.py
import numpy as np
import pytest
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/sample_prune_test.py -v`
Expected: FAIL（ModuleNotFoundError: jxl.sample_prune）

- [ ] **Step 3: 最小实现**

```python
# src/jxl/sample_prune.py
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
        rep = max(members, key=lambda i: float(sim[i] @ center))
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/sample_prune_test.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/jxl/sample_prune.py tests/sample_prune_test.py
git commit -m "feat(sample_prune): pure core — cluster, rank, decide"
```

---

### Task 2: plan 子命令（数据集+嵌入+conf → plan.json）

**Files:**
- Modify: `src/jxl/sample_prune.py`
- Test: `tests/sample_prune_test.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `decide`；输入 embeddings.npy（行序=stems 顺序）、confs jsonl（gencheck_infer 格式：`{"stem","cam","boxes","confs"}`，帧 conf=max(confs)，空检出=0.0）
- Produces: CLI `python -m jxl.bin.sample_prune plan <ds_dir> --embeddings E.npy --confs C.jsonl --ratio 0.2 --out plan.json`（plan.json 含 stems/keep/pool/meta；stem 与嵌入行对齐经 stems.txt 旁车文件——由 plan 命令从 ds images 重算并落盘）

- [ ] **Step 1: 追加失败测试**（构造 tmp 数据集 4 帧 + 4×2 嵌入 + confs jsonl，断言 plan.json 结构与 Task 1 决策一致）

```python
def test_plan_command_writes_json(tmp_path, monkeypatch):
    import json as j
    from pathlib import Path
    from typer.testing import CliRunner
    from jxl.bin.sample_prune import app

    imgs = tmp_path / "ds" / "images"
    imgs.mkdir(parents=True)
    for s in "abcd":
        (imgs / f"{s}.jpg").write_bytes(b"x")
    np.save(tmp_path / "emb.npy", _emb([[1, 0], [0.99, 0.14], [0, 1], [0.99, 0.12]]))
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
```

- [ ] **Step 2: 跑测试确认失败**（No module named jxl.bin.sample_prune）

- [ ] **Step 3: 实现**（新建 `src/jxl/bin/sample_prune.py`：typer app + plan 命令：`files_in(images)`→stems 排序；emb 行数必须==帧数否则 SystemExit；conf 帧级=max(confs) 缺 stem=0.0；调 `decide`；写 json）

```python
# src/jxl/bin/sample_prune.py (plan 部分)
#!/usr/bin/env python3
"""sample_prune CLI: 训练样本簇内削减(近重复优先)+隔离池. spec 见 docs/2026-09-09-训练样本量控制机制设计.md."""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from jxl.sample_prune import decide

app = typer.Typer(help="样本簇内削减+隔离池(plan/apply/pool-review)")


@app.command()
def plan(
    ds_dir: Annotated[Path, typer.Argument(help="数据集目录(images/+labels/)")],
    embeddings: Annotated[Path, typer.Option(help="帧 DINOv2 embeddings.npy, 行序=stems 排序")],
    confs: Annotated[Path, typer.Option(help="gencheck_infer 格式 jsonl")],
    ratio: Annotated[float, typer.Option(help="削减比例 0-1")] = 0.2,
    out: Annotated[Path, typer.Option(help="plan.json 输出")] = Path("prune_plan.json"),
) -> None:
    stems = sorted(p.stem for p in (ds_dir / "images").glob("*.jpg"))
    emb = np.load(embeddings).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    if len(emb) != len(stems):
        raise SystemExit(f"FATAL: embeddings {len(emb)} != 帧数 {len(stems)} (行序=stems 排序)")
    conf_by_stem: dict[str, float] = {}
    for line in confs.open(encoding="utf-8"):
        r = json.loads(line)
        conf_by_stem[r["stem"]] = max(r.get("confs") or [0.0])
    confs_sorted = [conf_by_stem.get(s, 0.0) for s in stems]
    p = decide(emb, confs_sorted, ratio)
    plan_doc = {"stems": stems, "keep": list(p.keep), "pool": list(p.pool), "meta": list(p.meta)}
    out.write_text(json.dumps(plan_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    typer.echo(f"plan: {len(stems)} 帧 -> keep {len(p.keep)} / pool {len(p.pool)} -> {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/sample_prune_test.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/jxl/bin/sample_prune.py tests/sample_prune_test.py
git commit -m "feat(sample_prune): plan command — ds+emb+confs -> plan.json"
```

---

### Task 3: apply 子命令（削减集 + 隔离池写出）

**Files:**
- Modify: `src/jxl/bin/sample_prune.py`
- Test: `tests/sample_prune_test.py`（追加）

**Interfaces:**
- Consumes: plan.json（Task 2 格式）；ds_dir 的 images/+labels/
- Produces: `--out-dir`（保留帧 symlink：images/labels）与 `--pool-dir`（隔离帧 symlink + `pool_meta.jsonl`：stem+meta+历次评测空列表）。同名已存在则 fail-fast。

- [ ] **Step 1: 追加失败测试**（apply 后：out_dir 帧数=keep、pool_dir 帧数=pool、pool_meta.jsonl 与 plan.meta 对齐、symlink 目标存在）

```python
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
    pj = tmp_path / "plan.json"; pj.write_text(j.dumps(plan))
    r = CliRunner().invoke(app, ["apply", str(tmp_path / "ds"), str(pj),
                                 "--out-dir", str(tmp_path / "pruned"), "--pool-dir", str(tmp_path / "pool")])
    assert r.exit_code == 0, r.output
    assert sorted(p.stem for p in (tmp_path / "pruned/images").glob("*.jpg")) == ["a", "c"]
    assert sorted(p.stem for p in (tmp_path / "pool/images").glob("*.jpg")) == ["b", "d"]
    meta = [j.loads(l) for l in (tmp_path / "pool/pool_meta.jsonl").open()]
    assert [m["stem"] for m in meta] == ["b", "d"]
    assert meta[0]["reviews"] == []
```

- [ ] **Step 2: 确认失败** → **Step 3: 实现**（apply 命令：读 plan → 循环 keep/pool 各建 symlink（images+labels，`Path.symlink_to(src.resolve())`）；pool_meta.jsonl 行=`{**meta_entry, "stem": stems[idx], "reviews": []}`；目标已存在 SystemExit）→ **Step 4: 测试通过** → **Step 5: Commit** `feat(sample_prune): apply — split ds into pruned + holding pool`

---

### Task 4: pool-review 子命令（回流评测 + 熔断）

**Files:**
- Modify: `src/jxl/bin/sample_prune.py`
- Test: `tests/sample_prune_test.py`（追加）

**Interfaces:**
- Consumes: pool_dir（Task 3 结构）；model .pt（ultralytics 推理，imgsz 640 conf 0.25）；`--prev-confs` jsonl（删除时各池帧 conf，从 pool_meta 的 conf 字段取）
- Produces: `backflow.jsonl`（`{stem, prev_conf, new_conf, backflow: bool, reason}`）；熔断：meta 中 removed_round≥2 的帧永不再回流（reason="fused"）。回流判据=检出丢失（new conf==0）或 conf 降 >0.2。执行后把评测结果追加进 pool_meta.jsonl 的 reviews 并 removed_round += 1。

- [ ] **Step 1: 追加失败测试**（monkeypatch 推理函数 `_infer_confs(model, img_dir) -> dict[stem, float]` 返回固定值；4 池帧：2 个降 0.3（回流）、1 个持平、1 个 removed_round=2（熔断）——断言 backflow.jsonl 与 pool_meta 更新）

```python
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
    r = CliRunner().invoke(app := sp.app, ["pool-review", str(tmp_path / "pool"),
                                           "--model", "x.pt", "--out", str(tmp_path / "bf.jsonl")])
    assert r.exit_code == 0, r.output
    bf = {j.loads(l)["stem"]: j.loads(l) for l in (tmp_path / "bf.jsonl").open()}
    assert bf["b"]["backflow"] is True      # 0.9 -> 0.5, 降 0.4 > 0.2
    assert bf["d"]["backflow"] is False     # 持平
    assert bf["e"]["backflow"] is True      # 检出丢失
    assert bf["f"]["reason"] == "fused"     # 熔断, removed_round=2
    meta = {j.loads(l)["stem"]: j.loads(l) for l in (tmp_path / "pool/pool_meta.jsonl").open()}
    assert meta["f"]["removed_round"] == 2  # 熔断不增轮
    assert meta["b"]["removed_round"] == 1
```

- [ ] **Step 2: 确认失败** → **Step 3: 实现**（`_infer_confs`：YOLO(model).predict(batch) 遍历图目录取帧 max conf；pool-review 主逻辑如 Interfaces；`app` 于 Task 2 已建）→ **Step 4: 测试通过** → **Step 5: Commit** `feat(sample_prune): pool-review — backflow eval with oscillation fuse`

---

### Task 5: SGCC 消融实验执行（非 TDD 运维链）

**Files:**
- Create: `/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/crop640_persons/gencheck/prune_ablation.sh`（数据现场，不进 repo）

**Interfaces:**
- Consumes: 全部前序命令；SGCC v3 best.pt（帧 conf 生成：gencheck_infer 对**训练集 13,788 帧**推理一次 → confs jsonl）；帧嵌入（复用现有 DINOv2 管线生成 embeddings.npy——若 63,088 帧时代已有全量嵌入则重算 13,788 子集）
- Produces: dataset_v3_p10/p20/p30 三个削减集 + 三次 sgcc0 训练 + `docs/2026-09-09-训练样本量控制机制设计.md` §4 证据表回填 + 隔离池回流评测结果

- [ ] **Step 1: 生成训练集 conf**（gencheck_infer.py best.pt 对 dataset_v3/all → confs jsonl，~10 分钟本机）
- [ ] **Step 2: 生成/复用帧嵌入 embeddings.npy**（行序=stems 排序对齐；DINOv2 管线现成——crop_foreground+embed 产物链）
- [ ] **Step 3: 三档 plan+apply**（sample_prune plan --ratio 0.1/0.2/0.3 × apply → 三个削减集 + 三个隔离池；对账 keep+pool=13,788）
- [ ] **Step 4: rsync 三集到 sgcc0 对等路径**（`~/ws/sgcc/.../dataset_v3_p{10,20,30}`；直连慢走 ds 中继）
- [ ] **Step 5: sgcc0 串行三训**（yolo26n 同 v3 配方 200ep；setsid nohup 链式；沿用 train_cls.sh 模式本地写脚本 rsync 执行——**勿嵌套 heredoc**）
- [ ] **Step 6: 逐个拉回指标 + pool-review 回流评测**（每档新 best.pt 对该档隔离池跑 pool-review）
- [ ] **Step 7: 回填 spec §4 证据表** + 结论（支撑/反驳/修正）→ git commit
- [ ] **Step 8: 若支撑**：机制写入 consensus-labeling skill（⑨ 后新节）+ 两项目 README 状态

---

### Task 6: 文档增补

**Files:**
- Modify: `~/.claude/skills/consensus-labeling/SKILL.md`（新节：样本量控制——簇内削减+隔离池+回流，含 SGCC 证据表指针）
- Modify: `docs/INDEX.md`（spec 与本 plan 登记）

- [ ] **Step 1: skill 增补**（Task 5 证据出来后写实测数字，勿提前写空数字）
- [ ] **Step 2: INDEX 登记 + Commit**

---

## Self-Review 结论

- Spec 覆盖：§2.1 算法→Task 1；§2.2 池→Task 3；§2.3 回流+熔断→Task 4；§2.4 实验→Task 5；§3 接线→Task 5/6 ✅
- 占位符：Task 3/4 Step 3 为「实现指令+接口契约」（代码结构已定、逻辑在 Interfaces 与测试中完整钉死）——测试即规格，可接受
- 类型一致性：PrunePlan(keep/pool/meta) 与 plan.json 字段、pool_meta.jsonl 字段全程一致 ✅
