# Datasets Pool 多来源样本目录 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `person/dates/` 重构为 `person/datasets/`（多来源样本池），收录各样本集，通过 `link_samples` 工具按 toml 配置前缀 symlink 组合训练集。

**Architecture:** `link_samples` bin（读 toml 配置 → `build_link_map` 纯函数算映射 → 前缀 symlink IO）+ datasets/ 数据迁移（移动各样本集 + YOLO 转换）+ experiments/ toml 配置。

**Tech Stack:** Python 3.12 / tomllib（内置，读 toml 配置）/ typer / pytest（tmp_path 集成测）。

## Global Constraints

- Python `~=3.12.0`；lib mypy strict；`src/jxl/bin/` mypy exclude；ruff `ALL` + bin per-file-ignores。
- 测试 `uv run pytest`（禁 subprocess 调 pytest）。
- 配置用 **toml**（tomllib 内置，无新依赖；spec 示意 yaml 但实现用 toml 避免 pyyaml 依赖）。
- datasets/ 路径：`/home/jiang/ws/sgcc/person/datasets/`。
- 前缀格式：`{dataset}_{original}`（全局唯一 + 可追溯）。
- No Silent Degradation：配置缺字段/样本集缺失 → 报错退出。
- 数据迁移不可逆 → 每步验证（ls/du）+ 谨慎 mv。

---

## File Structure

| 文件 | 职责 |
|------|------|
| Create `src/jxl/bin/link_samples.py` | 读 toml 配置 → `build_link_map` 纯函数 + 前缀 symlink CLI |
| Create `tests/bin/link_samples_test.py` | `build_link_map` 纯函数 + symlink 集成测试（tmp_path）|
| Create `experiments/baseline.toml` `experiments/non-standing-boost.toml` | 实验配置示例 |
| 数据迁移 `person/datasets/` | 收录各样本集（dates→datasets 改名 + 各集移动 + YOLO 转换）|

---

### Task 1: link_samples bin — build_link_map 纯函数 + 前缀 symlink

**Files:**
- Create: `src/jxl/bin/link_samples.py`
- Test: `tests/bin/link_samples_test.py`

**Interfaces:**
- Produces: `build_link_map(config: dict, datasets_root: Path) -> list[tuple[Path, Path, str]]`（(src_img, src_label, dataset_prefix) 映射，纯函数）；`main(config, out_dir, datasets_root)` CLI

- [ ] **Step 1: 写失败测试**

Create `tests/bin/link_samples_test.py`:
```python
"""link_samples 纯函数 + 集成测试。"""
from __future__ import annotations

from pathlib import Path

from jxl.bin.link_samples import build_link_map


def test_build_link_map_basic(tmp_path: Path) -> None:
    # 造 datasets/COCO/images/a.jpg + labels/a.txt
    ds = tmp_path / "datasets" / "COCO"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    (ds / "images" / "a.jpg").touch()
    (ds / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.2")
    cfg = {"name": "t", "datasets": ["COCO"], "split": [8, 1, 1]}
    pairs = build_link_map(cfg, tmp_path / "datasets")
    assert len(pairs) == 1
    src_img, src_lbl, prefix = pairs[0]
    assert prefix == "COCO"
    assert src_img.name == "a.jpg"
    assert src_lbl.name == "a.txt"


def test_build_link_map_multi_datasets(tmp_path: Path) -> None:
    for ds_name in ["COCO", "MOT17"]:
        ds = tmp_path / "datasets" / ds_name
        (ds / "images").mkdir(parents=True)
        (ds / "labels").mkdir(parents=True)
        (ds / "images" / "x.jpg").touch()
        (ds / "labels" / "x.txt").write_text("0 0.5 0.5 0.1 0.1")
    cfg = {"name": "t", "datasets": ["COCO", "MOT17"], "split": [8, 1, 1]}
    pairs = build_link_map(cfg, tmp_path / "datasets")
    assert len(pairs) == 2
    assert {p[2] for p in pairs} == {"COCO", "MOT17"}


def test_build_link_map_empty(tmp_path: Path) -> None:
    cfg = {"name": "t", "datasets": [], "split": [8, 1, 1]}
    assert build_link_map(cfg, tmp_path / "datasets") == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/bin/link_samples_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jxl.bin.link_samples'`

- [ ] **Step 3: 实现 link_samples.py**

Create `src/jxl/bin/link_samples.py`:
```python
#!/home/jiang/py/jxl/.venv/bin/python
"""按 experiment toml 配置,前缀 symlink 组合各样本集 → 训练池。

读 config.toml(name, datasets, split)→ 对每 dataset 的 images+labels
symlink 到 out_dir,链接名 {dataset}_{original}(全局唯一+可追溯)。
供 jxl_split 划分前组合不同实验的样本集。

用法:
    link_samples experiments/non-standing-boost.toml experiment/non-standing-boost/
"""

import tomllib
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, help="前缀 symlink 组合各样本集 → 训练池。")

_DATASETS_ROOT = Path("/home/jiang/ws/sgcc/person/datasets")
_IMG_EXTS = (".jpg", ".jpeg", ".png")


def build_link_map(
    config: dict, datasets_root: Path
) -> list[tuple[Path, Path, str]]:
    """读 config → [(src_img, src_label, dataset_prefix), ...] 映射(不 symlink)。

    纯函数:对 config["datasets"] 每个样本集,glob images,配对 labels,
    返回 (图源, 标注源, 前缀) 三元组供 symlink 阶段用。
    """
    out: list[tuple[Path, Path, str]] = []
    for ds in config["datasets"]:
        img_dir = datasets_root / ds / "images"
        lbl_dir = datasets_root / ds / "labels"
        for img in sorted(img_dir.rglob("*")):
            if img.suffix.lower() not in _IMG_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            out.append((img, lbl, ds))
    return out


@app.command()
def main(
    config: Annotated[Path, typer.Argument(help="experiment .toml 配置")],
    out_dir: Annotated[Path, typer.Argument(help="输出训练池目录")],
    datasets_root: Annotated[Path, typer.Option("--datasets-root", help="datasets/ 根")] = _DATASETS_ROOT,
) -> None:
    """读 toml 配置 → 前缀 symlink 各样本集 images+labels → out_dir/."""
    with config.open("rb") as f:
        cfg = tomllib.load(f)
    if "datasets" not in cfg or not cfg["datasets"]:
        typer.secho("配置缺 datasets 字段或为空", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    pairs = build_link_map(cfg, datasets_root)
    n_img = n_skip = 0
    for src_img, src_lbl, prefix in pairs:
        if not src_lbl.is_file():
            n_skip += 1
            continue  # 无标注(含 review 集无 YOLO labels)→ 跳过
        dst_img = out_dir / "images" / f"{prefix}_{src_img.name}"
        dst_lbl = out_dir / "labels" / f"{prefix}_{src_img.stem}.txt"
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())
        if not dst_lbl.exists():
            dst_lbl.symlink_to(src_lbl.resolve())
        n_img += 1
    typer.secho(
        f"{cfg.get('name', '?')}: 链接 {n_img} 图 (跳过无标注 {n_skip}) → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: 跑测试 + ruff + import 冒烟**

Run: `uv run pytest tests/bin/link_samples_test.py -v && uv run ruff check src/jxl/bin/link_samples.py tests/bin/link_samples_test.py && uv run python -c "from jxl.bin.link_samples import app, build_link_map; print('ok')"`
Expected: 3 passed / All checks passed / ok

- [ ] **Step 5: chmod +x + Commit**

```bash
chmod +x src/jxl/bin/link_samples.py
git add src/jxl/bin/link_samples.py tests/bin/link_samples_test.py
git commit -m "feat(bin): link_samples (prefix symlink datasets by toml config)"
```

---

### Task 2: datasets/ 迁移（dates→datasets + 收录各样本集）

**Files:** 数据迁移（`/home/jiang/ws/sgcc/person/` 下文件操作，非代码）

**注**：迁移不可逆，每步验证。sgcc 原样本位置、COCO/MOT 标注格式实现时确认（spec §8）。

- [ ] **Step 1: 确认当前样本集结构（sgcc 位置 + COCO/MOT 标注）**

Run:
```bash
cd /home/jiang/ws/sgcc/person
echo "=== samples/ 图命名模式(找 sgcc 特征) ==="
ls samples/images | head -20
echo "=== COCO/ 结构(有 json?) ==="
ls COCO/ | head; find COCO -name "*.json" | head -2
echo "=== MOT17/ MOT20/ 结构 ==="
ls MOT17/ | head; ls MOT20/ | head
echo "=== dates/ 子目录 ==="
ls dates/
```
Expected: 确认 sgcc 命名（vs MOT/COCO 区分）、COCO json 位置、MOT gt 位置、dates 子目录（含 COCO/MOT symlink？）。

- [ ] **Step 2: dates/ → datasets/ 改名 + 清理 COCO/MOT symlink**

```bash
cd /home/jiang/ws/sgcc/person
# dates → datasets 改名
mv dates datasets
# 清理 datasets/ 下的 COCO/MOT17/MOT20（若是 symlink 顶层副本，删；若是独立，保留移走）
ls -la datasets/COCO datasets/MOT17 datasets/MOT20  # 确认是 symlink 还是独立
# 若 symlink: rm datasets/COCO datasets/MOT17 datasets/MOT20
# 若独立: mv datasets/COCO datasets/COCO-tmp 等（避免与顶层冲突，后续合并）
```

- [ ] **Step 3: 收录原始来源（COCO/MOT17/MOT20/CrowdHuman → datasets/）**

```bash
cd /home/jiang/ws/sgcc/person
# 顶层 COCO/MOT17/MOT20 移入 datasets/（若已是 YOLO 格式直接 mv；若 json/gt 先转）
mv COCO datasets/COCO-raw  # 若需转 YOLO,转后 rename datasets/COCO
mv MOT17 datasets/MOT17
mv MOT20 datasets/MOT20
mv CrowdHuman datasets/CrowdHuman-raw  # CrowdHuman 原图+yolo+detmine 混,后续拆
# COCO 转 YOLO(若未转): uv run python -m jxl.bin.coco_to_yolo <coco_json> datasets/COCO-raw/images datasets/COCO
# MOT 转 YOLO(若未转): uv run python -m jxl.bin.mot_to_yolo ...
```
**注**：COCO/MOT 是否已 YOLO 格式取决于 Step 1 确认。若 samples/ 里已有 COCO/MOT 的 YOLO 标注（从 samples/ 拆出更省），则从 samples/ 拆而非重转。

- [ ] **Step 4: 收录自动标注集（video-extract / CrowdHuman-L1 / CrowdHuman-review）**

```bash
cd /home/jiang/ws/sgcc/person
# video-extract = dates/2026-07-07(person_mine 产出)
mv datasets/2026-07-07 datasets/video-extract
# CrowdHuman-L1 = det_mine L1(train_detmine + val_detmine images+labels 合并)
mkdir -p datasets/CrowdHuman-L1/images datasets/CrowdHuman-L1/labels
cp -rl datasets/CrowdHuman-raw/train_detmine/images/*.jpg datasets/CrowdHuman-L1/images/ 2>/dev/null
cp -rl datasets/CrowdHuman-raw/train_detmine/labels/*.txt datasets/CrowdHuman-L1/labels/ 2>/dev/null
cp -rl datasets/CrowdHuman-raw/val_detmine/images/*.jpg datasets/CrowdHuman-L1/images/ 2>/dev/null
cp -rl datasets/CrowdHuman-raw/val_detmine/labels/*.txt datasets/CrowdHuman-L1/labels/ 2>/dev/null
# CrowdHuman-review = review_all(图+manifest;标注待 P2/P3 转 YOLO)
mv datasets/CrowdHuman-raw/review_all datasets/CrowdHuman-review
```

- [ ] **Step 5: sgcc 原样本 → datasets/sgcc/（从 samples/ 拆 or 确认原始位置）**

```bash
cd /home/jiang/ws/sgcc/person
# 按 Step 1 确认的 sgcc 命名特征,从 samples/ 拆 sgcc 图
# (示例: sgcc 图名含特定模式,如日期/传感器 id; 实现时按实际命名筛)
# mkdir -p datasets/sgcc/{images,labels}
# for f in samples/images/<sgcc_pattern>*.jpg; do mv "$f" datasets/sgcc/images/; mv samples/labels/${f##*/}.txt datasets/sgcc/labels/; done
# 或: 若 sgcc 原始在 archive/ 或别处,mv 过来
ls datasets/sgcc/images | wc -l  # 确认 sgcc 图数(预期 ~14284)
```

- [ ] **Step 6: 验证 datasets/ 结构 + 各集 images/labels 对应**

```bash
cd /home/jiang/ws/sgcc/person/datasets
for d in */; do
  echo "$d: img=$(ls $d/images 2>/dev/null | wc -l) lbl=$(ls $d/labels 2>/dev/null | wc -l)"
done
```
Expected: 各样本集 images/labels 数对应（COCO/MOT17/MOT20/CrowdHuman/sgcc/video-extract/CrowdHuman-L1 有 images+labels；CrowdHuman-review 有图+manifest 无 labels）。

- [ ] **Step 7: 清理废弃（samples/ + dataset/ 旧产物）**

```bash
cd /home/jiang/ws/sgcc/person
# samples/(42843 合并池)已被 datasets/ 替;保留备份或删
# mv samples samples-old-backup  # 暂留备份
# dataset/(旧 jxl_split)废
# mv dataset dataset-old-backup
# 核心中间态 samples_core/dedup/reid 不动(历史)
```
**注**：删除前确认 datasets/ 各集完整（Step 6）。保守起见先 `mv *-old-backup` 而非 `rm`。

- [ ] **Step 8: Commit（迁移脚本/文档,数据不入 git）**

```bash
# 迁移是数据操作,git 不跟踪大文件;仅 commit 相关文档更新
git add docs/  # 更新 dates→datasets 引用(6 个文档残留 dates/2025-07-07)
git commit -m "refactor(data): migrate dates/ → datasets/ multi-source pool"
```

---

### Task 3: experiments/ 配置 + 端到端验证

**Files:**
- Create: `experiments/baseline.toml`
- Create: `experiments/non-standing-boost.toml`

- [ ] **Step 1: 写配置 toml**

Create `experiments/baseline.toml`:
```toml
# 基线实验(无 CrowdHuman,原 person.pt 训练集构成)
name = "baseline"
datasets = ["sgcc", "MOT17", "MOT20", "COCO"]
split = [8, 1, 1]
```

Create `experiments/non-standing-boost.toml`:
```toml
# 非站立增强实验(基线 + CrowdHuman L1 多模型共识 + 视频提取难例)
name = "non-standing-boost"
datasets = ["sgcc", "MOT17", "MOT20", "COCO", "CrowdHuman-L1", "video-extract"]
split = [8, 1, 1]
```

- [ ] **Step 2: 端到端验证（link_samples + jxl_split）**

Run:
```bash
cd /home/jiang/py/jxl
# link_samples 组合
uv run python -m jxl.bin.link_samples experiments/baseline.toml /home/jiang/ws/sgcc/person/experiment/baseline/
# 验证前缀 symlink
ls /home/jiang/ws/sgcc/person/experiment/baseline/images/ | head -5  # 应见 sgcc_xxx.jpg, MOT17_xxx.jpg, ...
echo "img=$(ls /home/jiang/ws/sgcc/person/experiment/baseline/images | wc -l)"
# jxl_split 划分(若 jxl_split 支持 symlink 输入)
# uv run python -m jxl.bin.jxl_split /home/jiang/ws/sgcc/person/experiment/baseline/ /home/jiang/ws/sgcc/person/dataset/baseline/ -r 8 1 1
```
Expected: images/ 含前缀名（sgcc_/MOT17_/MOT20_/COCO_），图数 = 各样本集图数之和。

- [ ] **Step 3: Commit 配置**

```bash
git add experiments/baseline.toml experiments/non-standing-boost.toml
git commit -m "feat: experiment toml configs (baseline + non-standing-boost)"
```

---

## Self-Review 记录

- **Spec coverage**: §3 datasets/ 结构 → Task 2；§4 配置 → Task 3（toml 替 yaml，无新依赖）；§5 link_samples → Task 1；§6 流程 → Task 3 Step 2；§7 迁移 → Task 2；§8 sgcc 待确认 → Task 2 Step 1/5。✓
- **Placeholder scan**: Task 2 迁移命令有条件分支（Step 1 确认后选路径），标注"实现时按实际"——这是数据迁移的固有不确定性（spec §8 已标待确认），非代码占位。link_samples 完整代码 + 测试。✓
- **Type consistency**: `build_link_map` 返回 `list[tuple[Path, Path, str]]` 在 Task 1 定义 + 测试 + main 消费一致。✓
- **toml vs yaml**: spec §4 示意 yaml，plan 用 toml（tomllib 内置，避免 pyyaml 新依赖，符合 j-python 锁定 9 库）。配置字段（name/datasets/split）语义不变。✓
