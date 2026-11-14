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

## 遗留与去向

- best.pth 已取回留档；n902 staging work 区保留（第二轮可直接 --resume 或新 run）
- gap 逐 epoch 曲线在 train_log.jsonl（能力账纪律：模型 ID scratch_v1 + 全参数快照在 eval_report meta）
