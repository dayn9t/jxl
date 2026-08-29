# NVIDIA LocateAnything-3B 调研存档（2026-08-29）

> 背景：英伟达 2026-05-26 发布的 3B「定位万物」统一视觉定位模型（Eagle VLM 家族 Embodied 分支，ECCV 2026）。本文档群是多路调研工作流的完整存档。
>
> **已实施**：本模型已接入 jxl（det/locateanything/ 模块 + 独立 venv FastAPI 服务 + det_mine la 校验器），
> 调研→实施→运维的总索引见 [../2026-08-29-LocateAnything-3B-接入.md](../2026-08-29-LocateAnything-3B-接入.md)。
>
> **知识库归并**：本主题的跨项目通用结论已提炼进个人知识库专题 `~/.claude/kb/30-areas/vlm-vision-grounding/`——
> - 模型参考卡：`[[20260829-locateanything-3b-model-card]]`（选型决策级摘要，本项目及后续项目以该卡为引用锚点）
> - 坐标协议已补进 `[[20260710-vlm-grounding-coordinate-protocols]]`（第五种协议形态：[0,1000] 整数 + 结构 token 块）
> - 专题导航：`[[vlm-vision-grounding/INDEX]]`
>
> 本文档群保留**全量细节**（数字表格、逐字引用、社区原话、检索过程），是知识库笔记的溯源层。

## 结论速览（30 秒版）

1. **能力**：PBD 并行框解码（box 当原子单元单步并行出坐标），Hybrid 12.7 BPS（H100）= 10× Qwen3-VL、2.5× Rex-Omni；LVIS 长尾 mean F1 50.7（+3.8 vs Rex-Omni，@0.95 31.1 vs 20.7）、ScreenSpot-Pro 60.3（3B 打赢 GUI-Owl-32B）、M6Doc 70.1（+14.5）、pointing 7 项全胜。**弱项**：COCO mean 54.7 仍低于 Grounding DINO-Swin-T 56.6，DocLayNet 低于 DocLayout-YOLO——不是「YOLO/Grounding DINO 替代者」，是开放词汇/语义查询/GUI/文档场景的互补件。
2. **许可（选型第一输入）**：权重 NVIDIA License **非商用**（research or evaluation only，NVIDIA 独享商用例外，无豁免渠道）+ 底座 Qwen2.5-3B（Qwen Research License）双重阻断；LoRA 微调产物强制延续非商用；「商用预标注→训自有检测器」闭环的入口行为本身大概率违约。**中文教程（知乎/阿里云/deephub）把许可证错标为 "NVIDIA Open Model License"——错误信息，以 HF LICENSE 原文为准。**
3. **数据集** `NVEagle/LocateAnything-Data`（12M 图/138M 查询/785M 框，2.41 TB）：无统一许可逐源继承，≥8 个已证实 NC 源（BDD100K/nuImages/MOT17/20/CrowdHuman/SKU-110K/Flickr30K/DeepFashion2/PartImageNet），7 源图像需自行 hydration。
4. **已知缺陷**：18 类多类检测 label corruption（issue #69，官方承认；规避 = `</c>` 多类 batch inference + Hybrid/Slow）；v2 承诺 2026-08 上旬发布，**截至 2026-08-29 已逾期约 3 周零沟通**；visual-prompt 权重未放出（LoRA 脚本已发）。
5. **部署**：推理仅官方支持 `transformers==4.57.1` + `trust_remote_code=True`；vLLM/SGLang/TensorRT-LLM/NIM/TAO 均无官方支持；BF16 显存 7.4–7.9 GB（12 GB 消费卡可跑，RTX 3090 grounding ~300 ms）；Linux only。
6. **商用替代**：GUI pointing → Nemotron 3 Nano Omni（Open Model License 可商用，ScreenSpot-Pro 57.8 同档，官方承认 LocateAnything 能力已贡献入该产品）；密集检测场景无证据，维持 Grounding DINO / Apache 系。
7. **对本项目（jxl）**：自动预标注是官方点名用例（其 138M 训练数据本身就是模型辅助标注流水线产物，与本项目 det-mine 多模型共识骨架同构），学术/评估场景合规可用；商用管线维持现状。模型无 track_id、无 confidence——视频预标注需外接 tracker + 几何一致性打分替代阈值过滤。

## 文件索引

| 文件 | 内容 | 要点 |
|---|---|---|
| [01-官方与论文.md](01-官方与论文.md) | 官方页/arXiv/repo/HF 卡精读 | 时间线、8 类任务模板、PBD 原理、四阶段训练、作者团队、口径差异记录 |
| [02-评测对比.md](02-评测对比.md) | benchmark 全表 + 竞品 | F1 协议可比性警示、强项/弱项诚实细读、BPS/显存、第三方独立评测盘点 |
| [03-部署与微调.md](03-部署与微调.md) | 工程师视角 | 权重获取、许可证核验、transformers 锁版、vLLM 社区方案、显存实测、LoRA/四阶段微调、输入输出契约 |
| [04-国内社区.md](04-国内社区.md) | 中文渠道 | 传播时间线、知乎/CSDN/B站/ModelScope/Datawhale、社区事实错误清单、中文教程推荐度排序 |
| [05-国际社区.md](05-国际社区.md) | 英文渠道 | Reddit 双峰反应、HN 冷淡、GitHub issue 编年、第三方生态（locate-anything.cpp 等）、API 直读热度数据 |
| [06-应用场景.md](06-应用场景.md) | 场景与生态定位 | 9 大官方用例逐个展开、Eagle 家族定位、Nemotron/Cosmos 关系、竞争格局、**KITTI 视频预标注可用性评估** |
| [07-缺口补扫A-许可与商用路径.md](07-缺口补扫A-许可与商用路径.md) | 许可证逐字核读 | 非商用条款三特征、豁免渠道结构性障碍、传染性边界、Nemotron 间接商用、GO/NO-GO 映射 |
| [08-缺口补扫B-v2状态与已知缺陷.md](08-缺口补扫B-v2状态与已知缺陷.md) | 缺陷与 v2 | issue #69 全文分析、v2 逾期多源证据、多类规避三层方案、visual-prompt 状态 |
| [09-缺口补扫C-数据集逐源许可.md](09-缺口补扫C-数据集逐源许可.md) | 数据集合规 | 逐源许可证核查表（8 NC 源）、hydration 机制、混训/蒸馏合规结论 |
| [10-来源总清单.md](10-来源总清单.md) | 全部来源 | 135 条去重来源（官方/论文/社区/许可证原文） |

## 调研方法（可信度说明）

- **工作流**：`locate-anything-research`（Run ID `wf_f80b013f-e63`）——6 路并行调研（官方论文/评测/部署/国内社区/国际社区/应用生态）→ 完备性审查 → 3 路缺口定向补扫；共 10 agents、236 次工具调用、约 100 万 token 调研量。
- **方法约束**：每路 ≥8 组中英文搜索词 + ≥3 页深读；关键数字双来源交叉验证；事实/社区观点/单一来源分层标注；许可证与数据集条款逐字核读原文。
- 各文件附录含本路来源清单与未能确认项（gaps）。
- 与本项目相关的既有知识：多模型共识自动标注骨架见知识库 `[[20260710-multimodel-consensus-hard-mining]]`（提炼自本项目 det-mine 设计）。
