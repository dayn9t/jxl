# OSNet from-scratch 训练配方 v1（低帧率跟踪阶段③·第一轮基线）

> 2026-11-14，jxl 训练线。裁决链：用户 A 案开工（iap-37 转达）→ **全新训练**补裁
> （from scratch，理由：OSNet×center3 新生产口径非同一任务谱系）→ 训练机 n902 用户指定
> （4060 Ti 16G，半栈机）。评估挂 iap `docs/EVAL-BENCHMARKS.md` #2 同尺，金标 60 对
> 终裁在 iap 侧（专题一准则：一切以金标实测裁决）。

## 1. 数据（staging 完成 2026-11-14）

| 项 | 值 |
|---|---|
| 正本 | `/mnt/data/jiang/ws/iap/gold/n001/samples/`（数据盘，71G 工作区） |
| staging | `sgcc-n902:~/staging/jxl-reid-osnet/`（469M，sha256 终验 1,290/1,290） |
| 训练对 | `pairs/train.jsonl` 1,716 对（session-anchor 正对/session-distant 远负/person-change 换人硬负；identities 256，正对并查集） |
| 留出对 | `pairs/eval.jsonl` 328 对（same 289 / diff 39；**held-out 严禁进训练**） |
| 帧图 | 1,288 张整帧（pair 涉及去重；crop = join `manifest.jsonl` 取 `detections[det_index].box_norm` 现抠） |
| v2.1 同源可比 | 数据协议与 v2.1 微调完全同源（同一 pair 档案）——from scratch 与预训练线的唯一差异 = 初始化 |

第二轮预留：center3（v0.13.9）上线后现场回流样本（贴新分布）+ v2.4 负对素材
（`drift_pairs_v24_neg_r01.jsonl` 110 异人对，j 侧数据盘）。

## 2. 模型与初始化（遵 2026-11-14 补裁）

- **架构**：OSNet x0.75（现役同构骨干——center3 机制作用在特征层，骨干不变）
- **初始化**：随机初始化（torch 默认 kaiming；**禁 ImageNet/MSMT17 预训练权重**——
  与「全新训练」裁决冲突，实验中若发现预训练初始化显著优势按程序呈报用户再定，不自行改判）
- 参考量级：x0.75 = 2.2M 参数 ONNX；n902 16G 显存容纳从零训（8-10G 估算）有余量

## 3. Loss 与采样（v2.1 实证配方沿用）

- batch-hard triplet（显式压最远正对/拉最近负对——v2.1 翻正主因）+ CE（identity 256 类）
- batch 组织按 pair 会话 identity 采样（同源协议）
- from scratch 预期：收敛所需 epoch 显著长于微调线（v2.1 微调 42 ep 收敛带波动；
  scratch 起步 150-200 ep + 早停 patience，具体以曲线定稿）

## 4. 评估协议（EVAL-BENCHMARKS #2 同尺）

1. 定标门：留出 328 对 same_p5 > diff_p95 gap（v2.1 在档：v2c gap +0.051；
   通用权重基线七格全败在档）
2. 管道门：thr 0.60 管道复测（F1 0.9677 / 换人 6/6 口径）
3. AUC：金标对档案 ROC（center3 对照口径 AUC 0.9542 在档）
4. 全部数字带模型 ID + 参数快照落档（能力账纪律）

## 5. 执行

- 工具：数据盘 `gencheck/osnet_ft.py`（v2.1 训练脚本）适配 scratch 模式
  （随机初始化分支 + epoch/调度参数）——改动落 jxl 侧工具位，同 commit 删旧调用形态
- 起训纪律（iap-37 操作令）：起训前 nvidia-smi 复核水位（当前 avs 1.8G 在用）；
  `setsid nohup` 脱离会话；不与半栈 avs 争显存峰值
- 里程碑：M1 staging+配方 ✅（本文档）→ M2 起训（长跑，曲线可查）→
  M3 金标评估+对照报告回执 iap

## 6. 对照与呈报

- 对照基准 = 现役 `osnet_x0_75_ft_v21`（预训练域微调线）在档数字
- **呈报触发**：scratch 线七格定标仍全败而预训练线可过门（或反向显著优势）→
  连同证据呈用户裁决初始化策略，不自行改判（iap-37 指令原文遵守）
