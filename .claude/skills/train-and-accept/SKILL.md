---
name: train-and-accept
description: sgcc3/sgcc0 远程训练与检测器验收流程（SGCC/SHTM 通用）。当要发训练（检测器重训/ReID 微调/烤机）、启动看门狗、收割训练结果、执行验收（塌陷段/FP 探针/多 seed/跨日期扩展）、或模型上线双切时使用。命令细节见 memory training-on-sgcc3，本 skill 管步骤序列与验收口径。
---

# 远程训练与验收（sgcc3/sgcc0）

> 知识库（命令/环境/教训细节）：memory `training-on-sgcc3`——本 skill 不复制，
> 只编排步骤。适用 jxl 双项目（SGCC 检测器/ReID、SHTM V2.x 重训）。

## 1. 发训

1. 同步：代码+数据 rsync 对等 `$HOME`（`-aL --partial --mkpath`，细节在 memory）
2. 环境：uv + **devpi 源**（`UV_DEFAULT_INDEX=http://192.168.18.146:3141/...`，
   勿用阿里镜像——lock 哈希不匹配）
3. 启动：`setsid nohup` 脱离 + 流式日志落盘（断 VPN 无碍）；小步验证显存余量
   （16G 卡 batch 警示：改 batch 后与历史数字不严格可比，报告注明）

## 2. 看门狗（必开）

`burnin_watchdog.sh` 形态：10min 采样温度/功耗/SM 时钟/利用率/存活 + Xid 巡检。
判据：Xid/NVRM 出现**即停上报**（不带病推进）；util>80% 时 SM 持续 <1000MHz = 降频；
温度 >85°C。注意 RTL8126A 网卡 "XID 64a" 是硬件版本号假阳性，只认 `NVRM: Xid`。

## 3. 收割

GPU 曲线统计（util>80% 时段的 SM 时钟下界）+ **Xid/NVRM 终核** + summary json +
训练稳定性记录（耗时一致性）——烤机视角数据进报告（机器长训资格实证）。

## 4. 验收口径（检测器）

| 项 | 口径 | 备注 |
|---|---|---|
| 塌陷段 | N/N 帧闭合 | v5 先例 226/226 |
| FP 探针 | 座椅区 FP/帧 | v5 基线 1/帧 → 目标 0（neg0906 系） |
| 专项 recall | 目标形态 miss 数 | 如玻璃反光 0.911→0.930 |
| 多 seed | **<0.006 单次 mAP 差不可判** | 烤机 2026-09-18 结论，跨版本必带带宽 |
| 跨日期扩展 | 零分钟塌陷/帧覆盖≥99.9%/独有检出真人率抽检 | 3 日期以上 |
| 负面代价 | FP 增量计数 + 生产是否放大 | 离线 FP 不必然生产放大 |

验收素材入池先过看图门（skill `data-flywheel` 第三步 / `vlm_gate.py`）。

## 5. 上线（jxl 只 stage，切换 iap/iapx 裁决）

dated 直指 + symlink 双切 + 等价冒烟 → 转入 skill `model-delivery` 第 4-5 步。

## 6. 烤机专项（若目的含稳定性验证）

负载时长目标 ≥17h（sgcc3 已实证：61°C/零降频/零 Xid 水平）；训练负载自带
计算正确性自检（loss/NaN），优于纯 gpu-burn；收割按第 3 步 + 独立烤机报告。
