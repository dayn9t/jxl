---
name: data-flywheel
description: n001 数据飞轮执行入口（SGCC）。当新视频数据拷回/落盘 n001、iap 跑批或 date-scope 增量批完成、用户提到数据飞轮/信号盘点/负样本挖掘/hard-negative、或发现零 session/闭馆日/检测异常形态时使用——执行批后信号盘点五件套与入池门流程。
---

# n001 数据飞轮（jxl 执行入口）

> 触发纪律：**新数据到达即执行本 skill**，不依赖对话提及（2026-09-18 教训：
> 组件动作做全 ≠ 飞轮在转；触发挂在管道尾巴，不挂在下次对话的运气）。
> 制度文档（为什么/跨方契约）：iap `docs/DATA-FLYWHEEL-SOP.md`——本 skill 只管怎么跑。

## 模式声明（2026-09-19 用户裁决）

**当前默认 = 全模型例行沉淀**：每批新数据，四个视觉模型（person 检测器正样本、
upper 分类器、role 分类器、OSNet 身份认定）的样本**全部**提取+自动标注+入池，
**不论当轮是否重训**——沉淀是资产积累，训练节奏另定。「先评估再选择是否沉淀」
是系统稳定后的**未来降级模式**，现在不启用。

## 触发条件（任一）

- 新视频数据拷回/落盘 `/var/howell/iap/v0.10/ias/sh-sgcc/n001/video/`（**只读红线**）
- iap 跑批/`--begin/--end` date-scope 增量批完成（每日 ~108 段）
- 闭馆静场日形态（src1 座椅区静物误检 + src2 真空——07-05/09-06 实证模式）
- 用户提及：数据飞轮 / 信号盘点 / 负样本 / hard-negative / 新数据怎么用

## 第一步：信号盘点五件套（每批 ~10 分钟）

**第 0 律：批判性消费**——任何交接报告/统计快照的关键数字先自行重算核实
（2026-09-18 实例：报告称 10,121 框恒定尺寸，重算 bbox 语义后才确认口径），
不直接采信转述数字。

| # | 信号 | 怎么查 |
|---|---|---|
| 1 | detect.failed sidecar | `find <video_root> -name "*.failed" -newer <上批标记>`，逐个归因 |
| 2 | session 密度 | 当批段/日×src vs 前 5 批均值；**对 0 日必查**（0=信号不是安静） |
| 3 | conf 分布 | `gencheck/domain_profile.py`（只读缓存，无参复现全表）p50/p90 vs 上批 |
| 4 | step6 模糊率 | iap inspector 产物统计（u2pp 无热词基线 ~23%） |
| 5 | session 形态 | 过切/换人/合并组探针（benchmark 六合并组口径） |

判读阈值表：见 SOP（iap `docs/DATA-FLYWHEEL-SOP.md`）。全绿 → 记录本批结果即收（飞轮空转也是转）。

## 第二步：异常分支 → 挖掘

- FP 簇（恒定框/静场日）：`gencheck/neg0906_extract.py` 模式——聚簇去重
  （小时 × 位置桶 × conf topK）→ 640×640 crop + manifest.jsonl
- reid 切分异常：`gencheck/osnet_drift_mine.py`（**先过 split_points 防错标门**，
  2026-09-17 勘误教训：跨换人点对是毒标注）
- ASR 模糊：留样本作热词 A/B 与评估集素材（语音层参照集裸奔中）

## 第三步：入池门（不可省——09-08 毒标注教训）

```bash
cd <gencheck>
python3 vlm_gate.py --dir <候选目录> --mode neg-person   # 全检，并发≤3 硬红线
```

- 全过（exit 0）→ images + 空 txt 入池（形态同 dataset_v5_glass）
- 有人/错误 → 剔除或复检；**每簇首张人工看一眼**（verifier 会错，SAM 3 EV 也只近人）
- manifest 登记来源（mkv/time/bbox/conf/cluster）——可追溯、可撤销

## 第四步：重训评估与回归（jxl）

- 攒批触发：候选池新增 ≥50 张 或 生产紧急回归；**非每日重训**
- 刻度三条：多 seed（**<0.006 单次 mAP 差异不可判**，2026-09-18 烤机结论）+ FP 探针
  （v5 基线 1/帧 → 目标 0）+ eval held-out 冻结（防自我强化）
- 训练机 sgcc3；每转记录 `projects/sgcc/README.md` 状态表 + 通知单往返 iap

## 第六步：全模型例行沉淀（默认模式，2026-09-19 起）

| 模型 | 样本形态 | 自动标注方式 | 入池位置 |
|---|---|---|---|
| person 检测器（正样本） | 整帧 + YOLO txt（框来自检测缓存） | 抽帧（日期×src×小时分层）→ spark VLM 验证「有人且框贴合」→ 通过者带框入池 | `gencheck/pospool/<date>/` |
| person 检测器（负样本） | 640×640 crop + 空 txt | FP 信号簇挖掘（第二步）→ VLM 全检无人 | `gencheck/neg_<date>/` |
| upper 分类器 | crop + upper_body 标签 | 缓存 upper_body 字段（upper_n_v2 预填）→ VLM 复核上半身可见性一致性 | `gencheck/upperpool/<date>/` |
| role 分类器 | session target_crop + role 标签 | step5 高置信判定作代理级标签（⚠️ 代理级，按 j-eval-benchmarks 分级记录） | `gencheck/rolepool/<date>/` |
| OSNet 身份认定 | 同人对（jsonl pair） | osnet_drift_mine（**v2.1 口径筛+split_points 门**——v2.2 证伪教训：用当前最强权重口径+配比甜点区，训练时再定并入量） | `osnet_ft/drift_pairs_<date>.jsonl` |

**自标注局限（记录在案）**：正样本框来自模型自身检出（SAM 引擎阶段 2 形态——
高置信预填+验证），模型漏检侧不覆盖；upper/role 标签含自产成分。例行沉淀的
已知边界，重大版本重训前应补人工抽检。

**新段 reid 切分画像**（随沉淀附带）：新段 session 密度/切分形态 vs 历史段——
若显著漂移，reid「终态」结论需重开。

## 资产指针

| 资产 | 位置 |
|---|---|
| 制度 SOP（跨方契约） | `~/cc/next/iap/docs/DATA-FLYWHEEL-SOP.md` |
| 设计底稿（SAM 三代引擎/防坍缩） | `projects/sgcc/research/2026-09-08-数据飞轮调研.md` + 同日深挖二轮 |
| 工具 | `gencheck/{vlm_gate.py, neg0906_extract.py, domain_profile.py}` |
| 首转记录 | neg0906（66 张，2026-09-18）→ v6 候选 |
| 教训与触发映射 | memory `data-flywheel-trigger` |

## 红线

spark 并发 ≤3；`/var/howell` 严格只读；eval held-out 328 对冻结不增不改；
入池门（VLM 全检 + 人工抽检）任何情况下不可跳过。
