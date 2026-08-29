# LocateAnything-3B 接入（总索引）

> 本文是 jxl 中 LocateAnything-3B（NVIDIA 3B 开放词汇定位 VLM）**全部知识的串联入口**：
> 调研 → 选型决策 → 实施 → 部署运维 → 溯源，五层各自的位置与读法见下方知识地图。
> 状态：**已实施并入 main**（commit `6e4c75b`，2026-08-29，GPU 实测 + 12-agent 工作流审核）。
> 许可红线：NVIDIA License **非商用**（research/evaluation only）——只进研究/评估链路，不进商用标注管线。

## 知识地图（按抽象层级）

| 层 | 位置 | 定位 | 何时读 |
|---|---|---|---|
| **总索引（本文）** | `docs/2026-08-29-LocateAnything-3B-接入.md` | 串联五层 + 速查 | 任何时候先看这 |
| 调研细节层 | `docs/2026-08-29-LocateAnything-3B-调研/`（12 文件，入口 [README.md](2026-08-29-LocateAnything-3B-调研/README.md)） | 全量细节：数字表格/逐字引用/社区原话/检索过程 | 需要证据与细节时 |
| 跨项目提炼层 | `~/.claude/kb/30-areas/vlm-vision-grounding/20260829-locateanything-3b-model-card.md`（本机个人知识库） | 选型决策级模型卡（3KB 精华），含 `[[vlm-grounding-coordinate-protocols]]` 坐标协议对照 | 其他项目复用结论时 |
| 运维状态层 | `~/.claude/projects/-home-jiang-cc-py-jxl/memory/locateanything-backend-plan.md`（会话记忆） | 实施状态 + 部署踩坑实录（渠道/wheel 版本/显存数字） | 重新部署或排障时 |
| 活文档层 | `src/jxl/det/locateanything/*.py` docstring + `script/la-*.sh` 注释 | 协议约定/许可边界/部署坑，随代码演进 | 改代码前 |

## 怎么用（三步）

```bash
# 1. 一次性环境安装（venv + 依赖 + 权重，幂等可重跑）
script/la-setup.sh          # 产物: /home/jiang/cc/py/jxl/.la-venv + models/LocateAnything-3B

# 2. 启动推理服务（常驻 FastAPI，:18306，la_flash strict，~7.9GB 显存）
script/la-serve.sh          # la_flash 不可用时显式降级: script/la-serve.sh --attn sdpa

# 3a. det_mine 校验器（单类 query，逐图 ~1.8s）
uv run python src/jxl/bin/det_mine.py <frames_dir> <out_dir> \
    --validators la --consensus 1 --validator-weights la:1.0 --target-model <yolo.pt>

# 3b. Detector2D 检测器（交互式，逐类单请求）
from jxl.det.locateanything.d2d_locateanything import D2dLocateAnything
det = D2dLocateAnything(model_path, opt, names=["person"])  # 构造即探活
```

架构：独立 la-venv（Python 3.10, transformers==4.57.1）常驻服务 ↔ jxl 主环境 httpx 客户端
（`LaClient` → `D2dLocateAnything` / `det_mine.detect_la`），功能性核心 `boxes.py` 纯函数单测覆盖（70 tests）。

## 关键事实速查

| 维度 | 事实 | 细节出处 |
|---|---|---|
| 能力定位 | 开放词汇/语义查询/GUI/文档场景互补件；COCO mean 54.7 **低于** GDINO-T 56.6，不是 YOLO 替代者 | 调研 02 |
| 许可 | NVIDIA 非商用 + Qwen Research License 双重阻断；中文教程许可证标注有误（以 HF LICENSE 原文为准） | 调研 07 |
| 已知缺陷 | 18 类多类 label corruption（issue #69）→ **单类单请求**是本接入的协议约定 | 调研 08 |
| 无 confidence | conf 恒 1.0；fp 数学不受影响，fn 的种子参考框边缘性受影响（`boxes.py` docstring） | 代码 |
| 重复框 | 官方已知问题 → 客户端 IoU 0.85 去重 | 代码 |
| 坐标协议 | [0,1000] 千分制 + `<box><x1><y1><x2><y2></box>`；官方 `parse_boxes()` 解析 | 调研 03 §6 |
| 本机实测 | RTX 4060 Ti：加载 2.2s / 单图 1.83s / 峰值 15924 MiB（与他人共卡，极限贴边） | 记忆 |
| 下载渠道 | hf-mirror 已死（308 回源）、HF 直连 ~410KB/s → **ModelScope `nv-community/LocateAnything-3B`**（~60MB/s） | 记忆 |
| flash-attn | 官方 pins 未列但 la_flash 必需；wheel `2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310`（sm_89 实测可用） | 记忆 / la-setup.sh |
| Python 3.10 | decord==0.6.0 wheel 最高 cp310、numpy==1.25.0 无 py3.12 wheel | la-setup.sh |
| MoonViT | 无需单独下载（视觉塔从主分片本地初始化，实测无网络拉取） | 记忆 |

## 关键决策记录（为什么这么做）

1. **FastAPI 常驻服务而非进程内加载**：transformers 4.57.1 与主环境 5.13.0 硬冲突须隔离；16GB 卡只容一份 7.9GB 实例，常驻服务天然单实例共享；对齐 rmb_ground 的 httpx 客户端先例（用户选定）。
2. **strict_attn 默认开启**：官方 batch_utils 对 la_flash 不可用会静默回退 SDPA（No Silent Degradation 违例）——启动即失败，`--attn sdpa` 才是显式降级。
3. **单类单请求**：官方多类 label corruption 规避；多类需求由调用方逐类循环（D2d）或天然单类（det_mine）。
4. **max_size 默认 1280**：本机显存现实约束（模型生产上限 2.5K，4K 大图会 OOM）；归一化坐标不受缩放影响。
5. **坏图=stem 缺席 dict**：与 gdino/rfdetr 校验器同构；服务中途挂掉整体 fail-fast，不逐图吞成"全损坏"。

## 问题路由（什么问题查哪份调研文件）

| 问题 | 文件 |
|---|---|
| 模型原理/PBD/训练四阶段 | 01-官方与论文 |
| 和 Grounding DINO/YOLO 比怎么样 | 02-评测对比 |
| 怎么部署/推理 API/微调/显存 | 03-部署与微调 |
| 中文社区有什么坑 | 04-国内社区 |
| 英文社区/第三方生态（llama.cpp/vLLM workaround） | 05-国际社区 |
| 用例/预标注评估 | 06-应用场景 |
| 能不能商用/替代路径 | 07-许可与商用路径 |
| 多类缺陷/v2 何时修 | 08-v2状态与已知缺陷 |
| 训练数据合规 | 09-数据集逐源许可 |
| 引用出处 | 10-来源总清单 |

## 溯源

- 论文：arXiv 2605.27365 ｜ 官方页：research.nvidia.com/labs/lpr/locate-anything
- 权重：HF `nvidia/LocateAnything-3B` / ModelScope `nv-community/LocateAnything-3B`
- 代码：NVlabs/Eagle `Embodied/`（vendored 副本在 `src/jxl/det/locateanything/vendored/`，含许可注记）
