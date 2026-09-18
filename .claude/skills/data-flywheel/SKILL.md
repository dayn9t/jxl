---
name: data-flywheel
description: n001 数据飞轮执行入口（SGCC）。当新视频数据拷回/落盘 n001、iap 跑批或 date-scope 增量批完成、用户提到数据飞轮/信号盘点/负样本挖掘/hard-negative、或发现零 session/闭馆日/检测异常形态时使用——执行批后信号盘点五件套与入池门流程。
---

# n001 数据飞轮（jxl 执行入口）

> 触发纪律：**新数据到达即执行本 skill**，不依赖对话提及（2026-09-18 教训：
> 组件动作做全 ≠ 飞轮在转；触发挂在管道尾巴，不挂在下次对话的运气）。
> 制度文档（为什么/跨方契约）：iap `docs/DATA-FLYWHEEL-SOP.md`——本 skill 只管怎么跑。

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
