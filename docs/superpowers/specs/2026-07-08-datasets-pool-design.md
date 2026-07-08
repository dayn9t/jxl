# Datasets Pool 多来源样本目录 + symlink 组合设计（2026-07-08）

> 把 `person/dates/` 重构为 `person/datasets/`（多来源样本池），收录各样本集（原始+自动标注），通过 `link_samples` 工具按 experiment yaml 配置**前缀 symlink** 组合训练集，支持不同实验不同样本组合。

## 1. 背景与目标

当前 `person/` 样本集散落且混乱：
- `dates/` 下混入 COCO/MOT17/MOT20（重复顶层）
- `samples/`（42843）合并池来源混杂（sgcc+MOT+COCO+CH L1+dates 混在一起，不可拆）
- 处理产物（samples_core/dedup/reid）与原始来源并列
- 无组合机制——不同实验要不同样本组合（如"非站立增强" vs "基线"）只能手动

**目标**：
- `datasets/` 统一收录各样本集（原始+自动标注，YOLO 格式）
- `experiments/*.yaml` 配置组合（配置即实验记录，可复现）
- `link_samples` 工具前缀 symlink 组合（全局唯一 + 可追溯来源）
- 不同实验选择性链接达成不同训练目标

## 2. 已确认决策

| 决策 | 选择 |
|------|------|
| 目录名 | `datasets/`（替代 `dates/`）|
| 收录范围 | 原始来源 + 自动标注集（**不含**历史处理 core/dedup/reid）|
| 组合机制 | 配置文件驱动（yaml）+ `link_samples` 前缀 symlink |
| 前缀格式 | `{dataset}_{original}`（全局唯一 + 可追溯）|

## 3. datasets/ 结构

```
person/datasets/
  COCO/              # 原始 COCO（coco_to_yolo 转 YOLO）
  MOT17/  MOT20/     # MOT（mot_to_yolo 转）
  CrowdHuman/        # 原始（crowdhuman_to_yolo 转，train_yolo/val_yolo 合并）
  sgcc/              # 监控原样本（从 samples/ 拆出或确认原始位置，见 §8）
  video-extract/     # dates/2026-07-07（person_mine 产出，5840，视频帧+标注）
  CrowdHuman-L1/     # det_mine 自动标注（多模型共识，12028）
  CrowdHuman-review/ # det_mine 高争议（5154，待 P2/P3）
```
每个样本集统一 `images/+labels/`（YOLO 格式，cls=0 person）。

## 4. 配置文件（experiments/）

```yaml
# experiments/baseline.yaml  ← 基线（无 CrowdHuman）
name: baseline
datasets: [sgcc, MOT17, MOT20, COCO]
split: [8, 1, 1]
```
```yaml
# experiments/non-standing-boost.yaml  ← 非站立增强
name: non-standing-boost
datasets: [sgcc, MOT17, MOT20, COCO, CrowdHuman-L1, video-extract]
split: [8, 1, 1]
```
不同 yaml = 不同训练目标。配置即实验记录（可复现、可对比）。

## 5. link_samples 工具（新 bin）

`src/jxl/bin/link_samples.py`：

```bash
link_samples <config.yaml> <out_dir>
```
- 读 config.yaml（`name`, `datasets: [...]`, `split`）
- 对每 `dataset`：symlink `datasets/<dataset>/images/*.jpg` → `<out_dir>/images/<dataset>_<original>.jpg`
- 同 labels：`<out_dir>/labels/<dataset>_<original>.txt`
- **前缀保证全局唯一**（COCO_xxx vs MOT17_xxx vs CrowdHuman-L1_xxx）+ 可追溯来源
- 验证 images↔labels 对应（每图有 label，反之；无 label 的图跳过或报错）
- 输出 `<out_dir>/{images,labels}/`（前缀 symlink）

**接口**：
```python
def link_samples(config: Path, out_dir: Path) -> None:
    """读 experiment yaml → 对各 dataset 前缀 symlink images+labels → out_dir/."""
```

## 6. 完整流程

```
1. 收录: 各样本集 → datasets/<name>/images+labels（YOLO，用现有 coco/mot/crowdhuman_to_yolo 转）
2. 配置: experiments/<goal>.yaml 列样本集组合
3. 链接: link_samples <yaml> experiment/<name>/   # 前缀 symlink
4. 划分: jxl_split experiment/<name>/ dataset/<name>/ -r 8 1 1
5. 训练: yolo_train dataset/<name>/data.yaml --model yolo26n.pt ...
```

**前缀机制收益**：
- 全局唯一（无文件名冲突）
- 可追溯（前缀 = 来源样本集）
- images↔labels 对应（同前缀+名）
- 实验可复现（yaml 记录组合）

## 7. 迁移计划

| 现状 | 迁移到 | 操作 |
|------|--------|------|
| `dates/` | `datasets/` | 改名 + 重构（去 COCO/MOT symlink）|
| `COCO/` `MOT17/` `MOT20/` | `datasets/COCO/` 等 | 移动 + YOLO 转换（coco/mot_to_yolo）|
| `CrowdHuman/` | `datasets/CrowdHuman/` | 移动（train_yolo/val_yolo 合并为 images+labels）|
| `dates/2026-07-07/` | `datasets/video-extract/` | 移动（person_mine 产出）|
| `CrowdHuman/train_detmine` + `val_detmine` L1 | `datasets/CrowdHuman-L1/` | 合并 L1（images+labels）|
| `CrowdHuman/review_all/` | `datasets/CrowdHuman-review/` | 移动（manifest + 图；标注待 P2/P3 转 YOLO）|
| `sgcc` 原样本 | `datasets/sgcc/` | 从 `samples/` 拆出（文件名特征）or 确认原始位置（见 §8）|
| `samples/`（42843 合并池）| 废弃 | 被 datasets/ + experiment 替；保留作"全集"备份或删 |
| `dataset/`（旧 jxl_split）| 废弃 | 被 `dataset/<name>/`（实验级）替 |
| `samples_core/dedup/reid` | 不迁 | 历史中间态，不动（可后续清理）|

## 8. 待确认（实现时）

- **sgcc 原样本位置**：sgcc 监控原样本（14284 图，person.pt 训练存档提）当前混在 `samples/`（24975 的一部分）。迁移时需从 `samples/` 按**文件名特征**拆出 sgcc 图（sgcc 命名 vs MOT/COCO 命名区分），或确认 sgcc 原始独立位置（`archive/`？`dates/` 某子目录？）。实现时 `ls samples/images | head` 看命名模式确认。
- **COCO/MOT 原始标注**：`COCO/` 当前 0 txt（json 标注），`MOT17/20` 14/8 txt（gt）。迁移时用 `coco_to_yolo` / `mot_to_yolo` 转 YOLO（若未转）。
- **CrowdHuman-review 标注**：review 5154 标注在 `manifest.jsonl`（非 YOLO labels）。作为样本集收录时，标注待 P2（在线 grounding）或 P3（人工）转 YOLO；或暂以 manifest 形式收录（P2/P3 消费）。

## 9. 关联

- 现有转换工具：`coco_to_yolo` / `mot_to_yolo` / `crowdhuman_to_yolo`（src/jxl/bin/）
- 划分工具：`jxl_split`
- 样本集存档：`docs/2026-07-08-CrowdHuman样本集存档.md`、`docs/2026-07-07-det-mine多模型难例挖掘.md`
- person.pt 训练：`docs/2026-06-27-person模型训练部署.md`（待用 datasets/ + experiment 重训）
