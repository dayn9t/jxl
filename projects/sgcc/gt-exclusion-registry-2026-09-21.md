# GT 排除登记（detection-v1）——jxl 侧指针页

> 2026-09-21 ｜ 权威登记（单一数据源）：`/mnt/data/jiang/ws/iapx/n001/gt/detection-v1/EXCLUSION-REGISTRY.md`（含帧清单 `frame-freeze-v1.jsonl`，559 行）
> 背景：iap P1-2 检测级 GT（生产域独立评估集）Phase 0，草案 `iap:docs/plans/2026-09-20-p1-2-detection-gt-brainstorm.md` §4.6-1。

## 排除规则（对本仓数据飞轮生效）

**`frame-freeze-v1.jsonl` 所列 559 帧（n001 共识三窗 2026-06-22 / 07-03 / 07-06，src 1+2）禁止进入任何 crop640 train/val/test 数据集版本**，包括以其为源图的合成派生（g5 类）。这批帧的使命是**评估**（衡量 person 模型在生产域的真实行为），一旦被训练消费，评估即变成训练集考试。

操作要求：

1. 新数据批次（正样本 crop / 负样本 / 合成）入库前，与 freeze 清单对账：crop 名 `{src}_0_{date}_{mkv_stem}_{idx:05d}.jpg` 经 n001 cache `frames[idx].pts_secs` → manifest 反查 wallclock，命中即剔除（键换算详见权威登记 §2.3）。
2. g5 类合成若继续生产，**不得**再从这 559 帧取源图。

## 对账结论摘要（2026-09-21，12 个 dataset_v* 版本并集 15,314 图像名）

| 消费形态 | 帧数 | 其中 |
|---|---|---|
| 原帧已入 train/val/test | **344** | train 284 / val 31 / test 29（split 归属跨版本稳定） |
| 仅 g5 合成派生入 train | **152** | 灰域待裁决（426 张 g5 源图 100% 追溯到本批帧） |
| 完全未消费（GT 可用） | **63** | 唯一可直接收编的存量 |

即：**v1 前的既有版本已把这批帧吃掉 344/559**——属历史事实，无需回滚；登记的意义是止损（余 215 帧不再流失）与裁决输入（g5 灰域 + 是否重建更大量干净 GT 帧）。

v7 新增 615 正样本（s1×350 + s2×265，2026-09-05 拍摄批次）与本清单零重叠，无涉。
