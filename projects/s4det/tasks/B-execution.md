# 任务书 B 执行记录（jxl 会话，2026-09-25）

> 对应 [B-infra.md](./B-infra.md)。四项全完成，commit 见 jxl 仓 git log（feat s4det B1 / fix n001 / chore B4 / feat B2）。

## B① 弱标签抽取器 ✅

- 工具：`projects/s4det/bin/b_preannot.py`（复用 event_index.jsonl 单一数据源；平台重复 Id 去重在索引管线完成：全等丢弃 1,079、冲突 0）
- 产物：`/mnt/data/s4det/preannot/manifest.jsonl` **56,999 行**（schema 与任务书示例逐字段一致，crop_path=null 惰性）
- 全量口径：agree 率 10.9%（agree=true 6,218 行=种子直通候选）；conf hi 20,277 / mid 15,832 / lo 20,890（lo 不入种子直通）；code_verif 与 CODEBOOK 全量交叉吻合（water_region 3,248 与码表逆向计数精确一致；ebike_single 12,464；ambig_g12 23,629 单独标记）
- **验收冒烟**：20 事件（ET2=6/ET3=6/ET14=5/其他 3），字段缺失 0，event.json 与 img_path 全部可回溯 ✓
- 叶类投影：按任务书约定未做，等 taxonomy.md 用户审批冻结后另跑投影步

## B② 多类 TargetProfile 方案评估与实施 ✅

### 取舍记录（两案同家族对比）

| 维度 | (a) 每叶类一 toml + 类独立 det_mine 链 + 数据集层合并 | (b) TargetProfile 扩多类 spec（改 3 工具） |
|---|---|---|
| SRP(#1) | det_mine 保持「单类 × N 校验器」单一职责 | det_mine 兼类间编排，单测面膨胀 |
| 问题本性 | 校验器栈**本就按类重配**（shtm 方案 §3-3 实证：prompt/闭集类表/权重 strong 定义逐类不同）——类差异放 per-class profile 数据里 | 需 profile 内嵌逐类 validator 配置表，schema 膨胀 |
| 改动面/风险 | **0 公共工具改动**（profile=数据 + 薄编排）；consensus_dataset 多 `--dump`+`--classes` 原生支持 | det_mine/la_relabel/doubao_arbitrate 深改（争议分/cascade/manifest 逐类化），P3 前引回归风险 |
| 算力共享 | 无（每类独立 pass） | 校验器侧理论可共享，但 target-model（被校验在役检测器）逐类不同，共享上限低 |
| 设计原则 | 组合优于继承(#3)、扁平模块优先(元原则)：类链=原子模块，编排=组合层 | 中央 fat schema，违反元原则 |

**裁决：(a)**。被否案 (b) 的理由：其唯一实质收益（校验器单 pass 多概念）共享上限低且将来可在 (a) 的类链内单独替换 validator 实现，不需要中央 schema；改动 3 个 battle-tested 工具的风险不成比例。

### 实施

- profiles：`projects/s4det/targets/{ebike,garbage_bag}.toml`（garbage_bag 演示 rfdetr_cls_id 省略→自动剔除闭集校验器机制；weights 暂为 yolov8m.pt 冒烟替身，s4det v0 就绪后换专项权重）
- 编排：`projects/s4det/bin/b_multirun.sh`（`<out_root> class=src,...`：每类独立候选帧池 → det_mine（profile 驱动 cls_id）→ consensus_dataset 多类合并）
- 踩坑记录：det_mine 须显式 `--dump-validators` 才落票文件；合并 images_dir 用**原始候选池**（det_mine 产物 images/ 只含非丢弃帧）；`--classes` 是 typer list 选项须逐类重复传参

### 冒烟（s4det 两粗类端到端）

- 数据：vehicle/garbage 族各 24 张 crop（p0-cluster 产物）
- ebike：L0 丢 9 / L1 自动 11 / review 4；garbage_bag：L0 丢 13（含空框共识=负样本，符合标注规范 #7）/ L1 自动 8 / review 3
- 合并：`22 帧 / 7 框 → dataset`，data.yaml `names: 0: ebike / 1: garbage_bag`，labels cls id 与 profile output_cls_id 一致 ✓
- 全流程 exit 0：`b_multirun.sh <out> "ebike=<src>,garbage_bag=<src>"`

## B③ n001-pipeline.sh 断链修复 ✅

- report 阶段改 `uv run --no-sync vlabel-consensus-report / vlabel-review-pack`（2026-09-21 迁 vlabel 后旧路径死亡；--no-sync 防误触依赖同步）
- 冒烟：crop640_persons 共识根 → **13,788 图分级 T1 8,069 / T2 2,348 / T3 1,003**；review_pack 1,003/1,003 条 → 51 网格 0 缺图 ✓
- 注：该根 images 在 `$root/images`（旧布局），冒烟用 /tmp symlink 组装管线原生形态，未动数据盘

## B④ 数据根建仓 ✅

- `/mnt/data/s4det/`（任务书 A 期已建，B④ 补全规划）：`p0-cluster/`（A 产）+ `preannot/`（B① 产）+ `quarantine/`（台账，自项目仓迁入为单一数据源）
- README 含目的/产生脚本/结构/运行状态/操作归属/日志（j-dir-readme 全项）
- 隔离台账 579 条：s4 会话裁决全量「确认丢弃」（legacy 无 event.json 结构性缺失，盘点 §8.3），resolution 已逐条回写，j-todo 闭环
