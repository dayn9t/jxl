# spark 主机 VLM 服务探查归档（2026-09-08）

> 只读探查 + API 实测。**核心结论：spark 就是 192.168.18.182 本身**——不是并列新资源，
> 而是既有 QWEN35 backend（`rmb_ground.py` / `rmb_describe.py` 的 DEFAULT_URL）所在的那台 DGX Spark。
> 模型 qwen3.5-35b-a3b-fp8 **实测具备视觉能力（确为 VLM）**，OpenAI 兼容 API，局域网可直连。


## ★ VLM 选型三分与夜间通道（2026-09-10 用户裁决，醒目）

VLM 任务三个选择：① 在线 doubao（商用付费）② 本地 spark（本档案，免费批量主力）
③ **GLM 5.3 Flash——当前驱动编程会话的大模型本身，支持视觉**（图片理解/分类可直接由它做，无需任何 API）。

**夜间通道**：任务时间在深夜~凌晨 → **优先 ③ GLM 5.3 Flash 直接处理**。
完整选型表见 memory `lan-vlm-resources`（每次会话自动加载）。

## 主机与 GPU 规格

| 项 | 值 |
|---|---|
| hostname / 架构 | spark-a54b / aarch64（DGX Spark） |
| GPU | NVIDIA GB10（统一内存，driver 580.159.03, CUDA 13.0） |
| 内存 | 121Gi 统一内存（CPU/GPU 共享；`nvidia-smi` memory.total 显示 N/A 属 GB10 正常现象） |
| CPU / 磁盘 | 20 核 / 3.7T NVMe（已用 226G） |
| 负载实况 | 探查时 GPU util 96%，内存 115/121Gi（vLLM 独占 0.92）——已近满载 |
| 多租户 | 用户 jiang（uid 1001，sudo+docker 组）为我们；另有 yangyaofei（127.0.0.1:11000 应用，与本服务无关）；同机跑 Dify 全家桶（nginx 443/20002、plugin 5003、weaviate/redis/postgres） |

## VLM 服务详情

- 引擎/部署：docker 容器 `qwen35-35b`，镜像 `nvcr.io/nvidia/vllm:26.04-py3`，restart=unless-stopped，2026-06-03 起已运行 3 个月
- 启动命令（容器 Cmd）：
  `vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --tensor-parallel-size 1 --max-model-len 65536 --gpu-memory-utilization 0.92 --quantization fp8 --served-model-name qwen3.5-35b-a3b-fp8`（env `HF_HUB_OFFLINE=1`）
- 模型：`qwen3.5-35b-a3b-fp8`（root `Qwen/Qwen3.5-35B-A3B-FP8`），上下文 **65536**
- 权重路径：宿主机 `/home/jiang/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8`，挂载进容器 `/root/.cache/huggingface`
- 另有 `models--Qwen--Qwen3.6-27B` 已下载未部署（是否 VL 未验证）
- **视觉能力实测（2026-09-08）**：
  - 32x32 纯红 jpg（RGB 200,30,30）→ 答"深红色、砖红色或铁锈红" ✓
  - 对照 32x32 纯蓝 jpg（RGB 30,60,220）→ 答"深蓝色或宝蓝色，接近纯蓝" ✓
  - 结论：image_url（data URI base64）调用正常，模型真实"看到"图片，**是 VLM**（natively multimodal，无需单独 VL 版模型名）
- 文本实测：chat/completions 200 OK；默认输出带 "Thinking Process:" 思考前缀（混在 `content` 字段）——做结构化 JSON 输出时需抑制（`chat_template_kwargs: {"enable_thinking": false}`，未实测）或解析时剥离

## 接入方式

- Endpoint：`http://192.168.18.182:8000/v1`（OpenAI 兼容：/v1/models、/v1/chat/completions；无鉴权）
- 监听 0.0.0.0:8000 → **局域网直连**，无需 ssh 隧道；本机实测 curl 通，ping ~12.6ms
- ssh 免密别名 `spark`（~/.ssh/config：HostName 192.168.18.182, User jiang, Port 22）
- ds（192.168.18.147）已在实际调用该端点（容器日志见其请求记录）——n001 crop640 标注链路即走此服务

## 与既有资源的关系（重要更正）

- **spark = 192.168.18.182，同一台机**。"spark 与 182 并列/互补"的前提不成立——不存在两份 35B 服务
- jxl 仓库引用：`src/jxl/bin/rmb_ground.py:45`、`src/jxl/bin/rmb_describe.py:33` 均指向 `http://192.168.18.182:8000/v1`
- 实践参考：`projects/sgcc/research/2026-09-08-VLM多属性标注实践.md`（qwen3.5-35b 批量推理/crop 标注/vLLM continuous batching 实录）——本档案是其基础设施侧补充
- 与商用 doubao 分工：本地 qwen 免费可批量（3.5 万 crop 蒸馏造集主力）；doubao 作高质量仲裁二审/终审刻度（0-1）

## 运维

- 重启：`ssh spark 'docker restart qwen35-35b'`（权重加载需数分钟）
- 日志：`docker logs qwen35-35b --tail N`（json-file driver，未配 rotation 上限）
- 开机自启：restart=unless-stopped ✓；同机 Dify 容器组（docker-* 前缀）为 compose 管理，重启 vLLM 不影响它们
- 权重更新：替换 `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8` 后需重建/重启容器

## 适用场景建议

- 大 crop 批量标注/属性蒸馏主力（免费、65536 上下文、continuous batching）——即 2026-09-08 多属性分类器构想的既定方案
- 仲裁二审可用但需注意：thinking 前缀 + 35B-A3B 刻度表现需先标定（对照 `n001-crop640-vlm-arbitration` 的 VLM 刻度记录：qwen 0-1000 / doubao 0-1）
- 不适合：与 ds（RTX 5080）训练任务抢资源时无关（异机）；但 spark 自身内存已近满，加新模型须先停现有服务

## 未知项与风险

- 容器以 root 运行、2026-06-03 由谁启动无记录可考（权重在 jiang cache，推测 jiang/团队部署，未确证）
- :8000 无鉴权且暴露整个 /16 局域网——内网环境可接受，敏感场景需加反代鉴权
- 并发吞吐未基准测试（探查时 GPU 96%，实时有 ds 流量；批量任务避开或错峰）
- `enable_thinking=false` 抑制思考、function calling、图像分辨率上限均未实测
- Qwen3.6-27B 已在缓存但未部署，用途与规格不明——动它前先问清来源
- 探查基于 2026-09-08 快照；容器 up 3 个月未重启，长期稳定性尚可但无监控告警

## Prompt 长度红线（2026-09-11 实证，SHTM hardcase_prune）

- **长规则 prompt（~700 字全判据）会把 qwen3.5-35b-A3B（3B 激活 MoE）压向「none」塌缩**：
  SHTM 难例削减 pilot 中 bucket_conflict 15→26，26 例肉眼核对 spark 几乎全错
- 紧凑 prompt（~340 字，判歧规则压成一行）即恢复正常；同 pilot 自动率 28%→48%
- 批量判定任务给 A3B 模型的规则 ≤ 一屏；细则靠「双模型一致闸门」兜，不靠加长 prompt
- 对照：doubao-seed-2.0-lite 同长 prompt 无此劣化（50B 级激活？未考据，但实测稳健）
