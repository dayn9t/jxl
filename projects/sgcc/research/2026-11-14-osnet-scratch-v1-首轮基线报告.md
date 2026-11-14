# OSNet from-scratch 首轮基线报告（scratch_v1，M3）

> 2026-11-14。M1 配方 `2026-11-14-osnet-scratch-训练配方-v1.md`；工具 commit 58e4897/047a1a9/2e42fe1；
> 产物 `gencheck/osnet_scratch/runs/scratch_v1/`（best.pth/eval_report/train_log 已取回）。
> 训练：n902（4060 Ti），150ep 跑满 ~12 分钟，best @ ep80。

## 结果：gate FAIL（差 0.009），触发预训练优势呈报条款

同协议同尺对照（held-out 328 对，same 289 / diff 39，余弦距离口径）：

| 线 | 初始化 | same_p5 | same_p95 | diff_p5 | diff_p95 | gap（same_p95−diff_p5） | gate |
|---|---|---|---|---|---|---|---|
| **v2.1（现役生产）** | MSMT17 预训练 | 0.0389 | **0.3296** | **0.4433** | 0.6664 | **−0.1138** | **PASS** |
| **scratch_v1（本轮）** | 随机 | 0.0156 | 0.3363 | 0.3273 | 0.8408 | +0.0090 | FAIL |

- gap 曲线：ep5 0.135 → ep35 0.071 → **ep80 0.009（best）** → 后期回升（ep95 0.319/ep125 0.227，常数 lr 后期震荡）
- **均值分离 scratch 反而略优**（same_mean 0.126/diff_mean 0.610 vs v2.1 的 0.153/0.574）——差距全部集中在**尾部判别**：diff_p5 0.327 vs 0.443（最难负对的分离），预训练初始化在此显著占优
- scratch 的 diff_p95 0.841 比 v2.1 的 0.666 更开——分布整体更散，长尾重叠更深

## 证据两面（呈报附件）

**预训练优势成立的证据**：
- 同尺差距 0.123（gap −0.114 vs +0.009），非噪声量级
- 尾部（p5/p95 正是门判据）全方位落后；150ep 内 gap 从未为负
- 训练动态：best 后回升 + 震荡，未见「再训就过门」趋势

**scratch 尚未调优的反证（差距可能收窄的因素）**：
- 常数 lr 1e-3 无衰减（v2.1 微调线也是常数 3e-4——但微调不需要衰减，scratch 需要）；ep80 触底后回升正是该上 cosine/step 衰减的信号
- 无 RandomErasing（OSNet 原生训练标配增强）
- 早停 patience 20 偏紧（曲线震荡期可能误杀低点后的恢复）
- 一轮 12 分钟的成本可再扫 2-3 组超参

## 呈报裁决点（用户 2026-11-14「全新训练」令的条款触发）

scratch 线同协议 gate FAIL 而预训练线 PASS，差距集中在尾部判别——**按指令呈报再定，不自行改判**。选项：
A. 继续 scratch：加 lr 衰减 + RandomErasing + patience 放宽，再扫 2-3 轮（每轮 ~12 分钟，成本低；若差距是「未调优」所致可填平）
B. 承认预训练初始化：回到 MSMT17 基座（v2.1 谱系）做 center3 口径的迭代——「全新训练」令解除的正当理由即本报告
C. A 先行一轮，仍 FAIL 则转 B（时间盒）

## 附章：B 谱系双微调对照终局（2026-11-14 晚补）

用户裁 B（「尽量从已有的模型迁移训练」，经 iap-24 转达）后补跑两轮微调对照，同尺四线终局：

| 线 | 初始化 | lr | best gap | gate | 备注 |
|---|---|---|---|---|---|
| scratch_v1 | 随机 | 1e-3 | +0.0090 | FAIL | 150ep，best@80 |
| finetune_v1（iap-24 代跑）| MSMT17 | 1e-3（默认）| −0.0100 | 勉强 PASS | 150ep 裸配方 |
| **ft_ref_v1（jxl）** | MSMT17 | **3e-4（v2.1 配方）** | **−0.0668** | **PASS** | 60ep，best@15 |
| v2.1 现役（2026-09 在档）| MSMT17 | 3e-4 + EXTRA_PAIRS | −0.1138 | PASS | 生产口径 |

结论修正（较 iap-24 首报「裸配方追不上调优成品」）：**lr 3e-4 即达 −0.0668**（比裸配方 −0.010 提升一个量级），与 v2.1 剩余差距 0.047 的主要嫌疑 = EXTRA_PAIRS 姿态漂移增量对（v2.1 加载、两轮对照未加载）。第二轮正式迁移的配方启示：**3e-4 + EXTRA_PAIRS + center3 现场回流数据**。

## 附章二：数据盘二次故障（2026-11-14 晚实录）

对照收官时 /mnt/data 全树 IO 错误（11-12 SATA 故障复发，osnet_scratch/ 与 gold/n001/gencheck 等全部探测不可读写）。**训练线数据面无恙**：n902 staging 自成一体（1288 帧样本副本 + manifest/pairs/gt + crops 索引 + base_weights.pth + 三 run 全部产物），数据窗等待期训练线不受影响。v2.1 历史 eval 数字已固化本报告（原档暂不可读）。修复决策归用户/运维；修好后产物应从 n902 归位数据盘。

## 遗留与去向

- 三 run 产物在 n902 `~/staging/jxl-reid-osnet/work/runs/{scratch_v1,ft_ref_v1,finetune_v1}`（数据盘故障期间的权威副本）；本机数据盘取回件暂不可读（随盘修复归位）
- gap 逐 epoch 曲线在 train_log.jsonl（能力账纪律：模型 ID + 全参数快照在 eval_report meta）
- 第二轮触发 = center3 上线 + 现场回流数据到位（与 iap 数据窗批同窗）
