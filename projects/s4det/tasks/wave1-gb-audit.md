# wave1-gb-audit：garbage_bag 装配集污染面审查（2026-09-27）

正本报告（逐框清单/票据/清洗坐标）：`/mnt/data/s4det/p3-seedset/batch2/mining/gb_audit/report.md`

## 背景

T4-b1 两起预标注事故（①qwen 双子把人身上背包共识成 garbage_bag；②~10px 小目标幻觉）同款根因排查：批 2 装配集 123 个 cls=1 框为同旧 prompt（无排除项）链路产物，疑背包已入训练集。修复基线 commit c9277f6（prompt 排除项 + downscale=2 门控 + 同帧去重）。

## 方法

1. 机器复审：新 prompt（targets/garbage_bag.toml vlm_prompt，sha256_12=da80856e9805）+ downscale=2；qwen38-local 全量 → qwen-flash 二审无支持帧；支持判据 IoU>=0.4 或 contain>=0.7。
2. 目视终裁：40 非干净框逐框全分辨率裁片 analyze_image 判 A/B/C/D；蒙太奇法因两轮自相矛盾弃用。
3. 终判矩阵：机器无支持+目视 B/C=确认污染；+A=VLM 漏检保留；+D=人工小窗；suspect+A=clean；suspect+B/C=确认污染。
4. 对照：机器 clean 抽 5 框目视 5/5 全 A。

## 数字

- 批 2（123 框）：**确认污染 25（20.3%）**、人工小窗 1（0.8%）、干净保留 97（clean 89 + vlm_miss 8）。
- 污染构成：车罩/挡风被/篷布类 10、其他类混淆 8（衣物/石墩/护栏/轮胎/车斗/包裹/散片/几何体）、人携带包类 7（事故1 同款）、小目标存疑 1（事故2 同款，28x26px）。
- 分来源：arb 旧仲裁 12 / L1 v0mini dump 12 / L0 1 —— v0mini 循环污染链（批1→v0mini→批2 dump）实证。
- 批 1 抽样 15 框：污染 2（13.3%，均为桶类混淆）→ 外推 303 框约 40 框（Wilson 95% CI 11~115）。

## 建议（裁决归主会话）

1. 批 2 确认污染 25 框按 `gb_audit/cleanup_list.md` 删框清洗（只删框不删帧；执行前可 VLabel 抽检——目视为单模型单轮）。
2. 存疑 1 框（2026-07-06_6a4ae8c9974f378d6d2099b0_r3 i1, 28x26px）建议直接删或隔离台账。
3. 批 1 全量同法复审（脚本可复用）或记隔离台账；批 1 清洗后裁决 v0mini 重训。
4. prompt 候选改进：显式排除「车罩/防尘罩/挡风被/篷布/苫布等覆盖物」（复审旁证显示新 prompt 已压住该类，显式化巩固）。
5. doubao-vl 在 garbage_bag 旧仲裁污染票中占比最高，权重复盘走 s4det 差异化机制。
