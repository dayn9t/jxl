# s4det T4 预标注：VLM 池参数与 prompt 调优调研（2026-09-27）

> 任务：以「大模型产出最好检测标注」为目标，对照官方文档与第三方经验审计现值，产出建议清单 + 实验设计。
> 实测对拍**等 T4-b1 225 帧 GT 编辑完成后**进行（本文只调研不改代码）。
> 背景正本：KB `~/.claude/kb/30-areas/vlm-vision-grounding/`（服务清单 20260913 / 分辨率门控 20260927），本文不重复。

## 0. 现值快照（审计基线）

检测路径 `src/jxl/bin/vlm_pool.py::call_person`（t4_preannot 走此路）：

| 项 | 现值 | 备注 |
|---|---|---|
| temperature | 0.0（全部四票） | 硬编码 |
| max_tokens | 1500（call_person 硬编码） | 注意：非 120；120/768 是 call_role_vote（菜单投票）路径 |
| 图 | JPEG q88，downscale=2 → 960×540，MAX_SIDE=1024 上限 | 分辨率门控用户已裁决（KB 20260927） |
| prompt | toml vlm_prompt（中文指令 + 排除项）→ pool_prompt 换 JSON bbox_2d 契约 | 范本 garbage_bag.toml |
| doubao | thinking disabled（extra_payload） | |
| glm | 恒思考不可关；reasoning_content 独立解析 | |
| qwen38-local | chat_template_kwargs.enable_thinking=false | |

## 1. 官方文档发现

### 1.1 Qwen3-VL（dashscope qwen3-vl-flash-2026-01-22）

**A. 官方 grounding prompt 范式（cookbook 2d_grounding.ipynb）**——多目标检测官方推荐句式：

```
Locate every instance that belongs to the following categories: "plate/dish, scallop,
wine bottle, ...". Report bbox coordinates in JSON format.
```

自定义输出 schema 的官方做法是**在 prompt 内联逐字 JSON 例**（Example 4）：

```
...report bbox coordinates, vehicle type and vehicle color in JSON format like this:
{"bbox_2d": [x1, y1, x2, y2], "label": "vehicle", "type": "car, bus, ...", "color": "..."}
```

要点：
- 坐标系为**训练内建的 0-1000 相对坐标**（qwen2.5-vl 绝对坐标→qwen3-vl 相对），prompt **无需**教刻度；我们 prompt 里的刻度说明无害可留。
- 官方全部示例**零样本**（无 few-shot 图文示例）。
- system prompt 可改输出格式（bbox-format JSON 是默认）——但阿里云文档明确「建议不设置 System Message，指令走 User Message」，我们现值合规。

**B. 官方已知限制（cookbook §7 Known Limitations，直接相关）**：
1. **单类实例数显著超过 40-50 或密集堆叠时，模型可能进入无限生成循环**——现值 max_tokens=1500 是硬顶（防跑飞），但 38 框 × ~30 tok ≈ 1140 tok 已贴近上限，**密堆类（garbage_pile/leaves_pile/piled_goods）有截断风险**，且我们尚未记录 finish_reason 无法发现。
2. **密集成簇目标偶尔被合并为单框**，官方明示「可通过细化 prompt 缓解」——garbage_bag.toml 的「密集成堆时框整体垃圾堆，可辨个体袋逐个框」正是官方建议的 prompt 级缓解，方向正确。

**C. 采样参数官方口径（docs 站 generation-parameters 页 + README）**：
- 任务表：OCR/文本抽取 temp **0.1**（"Low temperature for accuracy"）；事实类任务 0.1-0.3；**确定性任务用 greedy（do_sample=False）**。
- README vLLM 快速上手样例本身即 `temperature=0, top_k=-1`（greedy）。
- cookbook API 调用**不设 temperature**（用服务默认）——官方对 grounding 未给专属温度行，但归类为确定性/抽取型任务无疑。
- **结论：现值 temp=0.0 与官方口径一致，保留。**

**D. 图像规格**：cookbook API 封装带 `min_pixels=64*32*32, max_pixels=9800*32*32`（≈1M–10M px 感知预算控制）；阿里云百炼文档：qwen2.5-vl 在 **480×480–2560×2560 范围内定位鲁棒，范围外偶发检测框漂移**——960×540 在范围内（qwen3-vl 未单列，沿用同代口径）。qwen3-vl-plus/flash **enable_thinking 默认 false**（百炼文档），我们未传该参 = 吃默认，正确。

### 1.2 Doubao-seed-2-1-pro-260915（火山方舟）

**官方 grounding 文档**（全量抓取）要点：
- 坐标输出：**`<bbox>x_min y_min x_max y_max</bbox>` 标签**，归一化到 1000×1000（即 0-999），左上原点；支持 `<point>x y</point>`。
- 多目标检测官方 prompt 范式（与 qwen 同构的指令式 + 逐字格式例）：

```
请检测图像中所有属于 "plate（盘子）、photo（照片）、kid（小孩）、cup（杯子）" 类别的物体。
对于每个物体，请提供其类别和边界框，格式为：
[{"category": category, "bbox": "<bbox>x1 y1 x2 y2</bbox>"}, ...]
```

- **官方示例不设 temperature / thinking**（用服务默认）——即官方未对 grounding 给低温建议，我们的 0.0 是自行外推（见差异表 D2）。
- 明确类别列表（含中英对照括注）「使识别更准确」。
- 官方示例模型已是 seed-2-1-pro（doc 例 260628；我们 260915 更新，无换代必要）。
- 实测行为差异（KB 已录）：260915 实际吐 JSON `bbox_2d` 而非 `<bbox>` 标签——parse_boxes 双协议（标签优先、JSON 兜底）已兼容，两种官方/实测形态都吃得到。
- thinking：官方函数调用文档口径「关闭 thinking 会提高效率」；grounding 文档示例不传 thinking。KB 实测开思考 >120s 超时。**disabled 保留。**

### 1.3 GLM-5.3-flash（bigmodel）

官方模型页（vlm/glm-5.3-flash）与深度思考页：
- **官方推荐参数：`temperature: 1、top_p: 0.95、reasoning_effort: max`**——注意这是为其推理/Agent 剖面调的，**不是坐标抽取剖面**；与池内其他家的低温口径冲突（见差异表 D3，实验裁决）。
- **强制思考官方确认**：「GLM-5.3、GLM-5.3-FLASH 不再支持关闭思考（thinking.type 传 disabled 将会报错）」；`reasoning_effort: low` 仅 GLM-5.3（非 Flash）支持——5.3-flash 无降思考档位。思考走独立 reasoning_content（现值解析正确）。
- 官方建议 `thinking.clear_thinking: false`（响应保留思考内容，配合流式）。
- **官方完全无 grounding/bbox 文档**：视觉页只有 UI 复刻/Office 渲染/Blender/CAD 等 Agent 场景，无坐标输出规范、无图像分辨率限制——0-1000 刻度纯靠 prompt 引导 + 实测标定（KB t7_calib divisors_seen=[1000] 已做）。视觉 MCP 文档也不含 bbox 提示词。
- 输出上限 128K；思考 token 计费且「max_tokens 须大于思考占用」（第三方实测报错同 KB catvote 教训）——grounding 路径 1500 需监控截断。

## 2. 第三方经验共识（VLM 检测标注 prompt 工程）

1. **结构化 JSON 约束**：官方（qwen cookbook 内联 schema 例 / doubao 逐字格式例）与论文实践一致——**给逐字输出例**优于纯描述式格式说明。我们 POOL_JSON 契约行已是逐字例，合规。
2. **负例排除条款**：有效。MissingBench-Verified（OpenReview）与 Positive-Negative Prompting 研究均证负例示范降低错误率；我们 garbage_bag 排除项双事故治愈（KB 20260927）是域内实证。**每类必带排除行维持。**
3. **few-shot**：官方 cookbook 全零样本；API 场景下图文 few-shot（多图 in-context 带框示例）成本高、收益无公开证据（DetPO 类 ICL 检测工作存在但未获可引细节）。**不建议引入**——除非实验证明必要。
4. **穷尽性技巧**：官方句式 "Locate every instance that belongs to..."（穷尽语义 + 类别清单）；doubao 文档明确「明确类别列表使识别更准确」——比泛指「所有垃圾」更好的是列子类（塑料袋/编织袋/…）。我们 vlm_prompt 已是「袋装垃圾(塑料袋/编织袋盛装垃圾)」带子类括注，同构。
5. **temperature 0 vs 低温**：跨源共识——确定性抽取任务 greedy/低温（qwen 官方 0.1/greedy；社区同）；例外是 GLM 官方 1.0（推理剖面）与 doubao 官方不设（默认）。**temp=0 是安全默认，两处例外用实验裁决。**
6. **重复采样投票**：
   - 单模型多次采样 + 多数决（self-consistency，Efficient Test-Time Scaling for Small VLMs）与检测侧 **WBF（Weighted Boxes Fusion）** 是成熟做法：WBF 按 IoU 聚簇后**置信度加权平均坐标**，通常优于 NMS 式「取代表框」。我们共识聚类现取簇代表框（greedy_iou_match + consensus），**最终 GT 框坐标可升级 WBF 平均**——位置精度（IoU 提升）直接受益，改动只在合并纯函数层。
   - 训练侧（非本调研范围但记录）：per-object co-teaching（arXiv 2511.09955，KITTI mAP@0.5 31→47）说明伪标签噪声在训练阶段仍可再滤一层，与 s4det 训练链相关、与预标注参数无关。

## 3. 现值 vs 官方/第三方建议差异表

| # | 模型/层 | 项 | 官方/第三方建议 | 现值 | 是否改 | 理由 |
|---|---|---|---|---|---|---|
| D1 | qwen-flash | temperature | greedy/0.1（确定性任务官方口径） | 0.0 | **不改** | 一致 |
| D2 | doubao-vl | temperature | 官方 grounding 示例**不设**（服务默认≈1.0） | 0.0 | **实验裁决** | 唯一无低温官方背书的成员；0.0 是外推 |
| D3 | glm-flash | temperature | 官方推荐 **1.0**+top_p 0.95（推理剖面） | 0.0 | **实验裁决** | 官方口径与抽取任务直觉冲突；glm 仅仲裁票 |
| D4 | 全池 | max_tokens(grounding) | qwen 官方：>40-50 实例可能无限循环；密集输出 ~30tok/框 | 1500 硬编码，无 finish_reason 记录 | **改（小）** | 密堆类截断风险；加 finish_reason=length 日志，必要时升 3072（qwen-flash 路径优先） |
| D5 | qwen 双子 | prompt 范式 | 官方极简句式即可（刻度训练内建） | 中文指令+排除项+逐字 JSON 例 | **不改** | 更富但不冲突；可做 A/B 佐证（E5） |
| D6 | doubao | prompt 范式 | 官方：类别清单 + 逐字格式例（JSON 内嵌 bbox 标签） | 同构（JSON bbox_2d） | **不改** | 实测吐 bbox_2d JSON，双协议解析已兼容 |
| D7 | doubao | thinking | 官方 FC 文档：关思考提效；grounding 示例不传 | disabled | **不改** | 官方+KB 超时实测双证 |
| D8 | glm | 思考/max_tokens | 强制思考确认；max_tokens 须 > 思考量 | 1500（grounding） | **监控** | 思考 ~130tok 实测，密帧可能更长；随 D4 一并记 finish_reason |
| D9 | qwen38-local | enable_thinking | vllm qwen3 系通用开关（KB+社区一致） | false | **不改** | 已实证 |
| D10 | 全池 | 图 960×540 | qwen 鲁棒区间 480-2560 内；doubao/glm 官方无分辨率建议 | downscale=2 | **不改** | 用户已裁决（KB 20260927），不重开 |
| D11 | 共识层 | 框合并 | WBF 加权平均坐标优于代表框（第三方共识） | IoU 聚簇取代表框 | **实验裁决** | 纯函数层改动，GT 框质量直接受益 |
| D12 | 全池 | system msg | 阿里云：建议不设 System Message | 未设 | **不改** | 合规 |

## 4. 建议实验（全部等 T4-b1 GT 就绪后跑，同一评测器/同帧集）

**E1. doubao temperature 消融**（优先级最高——官方零低温背书的唯一成员）
- 臂：temp=0.0（现值）vs 服务默认（删 temperature）vs 0.7；thinking 保持 disabled。
- 指标：F1@IoU0.5、P/R、parse_error 率、延迟。帧集：T4-b1 225 帧 GT。
- 预期：若默认/0.7 更优则 doubao 单票脱钩全局 0.0（per-model temperature 进 CANDIDATES spec）。

**E2. glm temperature 消融**（官方 1.0 vs 现值 0.0）
- 臂：temp=0.0 vs 1.0+top_p=0.95（官方推荐组合）；可选 clear_thinking=false。
- 指标：仲裁票质量——其对 split 残量的救回正确率 + F1。glm 是池内最弱票（0.482），若 1.0 显著抬升，仲裁价值升格。

**E3. WBF 坐标融合 vs 代表框共识**（只动合并纯函数，API 零成本）
- 臂：现 greedy 簇代表框 vs WBF（簇内坐标加权平均，权重=模型 strength 分层）。
- 指标：共识框 vs GT 的平均 IoU、F1@IoU0.5（同票集重放即可，不需重新调 API——用 T4-b1 已落盘的原始票）。

**E4（小改先行，非实验）**：call_person 加 finish_reason 记录 + 密堆类帧抽检截断率；若 length>0 再升 max_tokens（qwen→3072）。

**E5（佐证性 A/B）**：qwen 双子「现 prompt vs 官方极简句式」对拍——验证排除项/要点行不伤召回（KB 已有两事故图治愈证据，预期现 prompt 胜出或持平，落个数字归档）。

## 5. 来源 URL 全录（引用落 URL 准则）

**Qwen 官方**
- cookbook 2D grounding（prompt 范式 + 已知限制）: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb
- 生成参数页（任务温度表/greedy 建议）: https://mintlify.wiki/QwenLM/Qwen3-VL/inference/generation-parameters.md
- 文档索引: https://qwenlm-qwen3-vl.mintlify.app/llms.txt （canonical: https://mintlify.com/QwenLM/Qwen3-VL/llms.txt）
- grounding 能力页（0-1 相对坐标口径，与 cookbook 0-1000 并存——以 cookbook 实测为准）: https://mintlify.wiki/QwenLM/Qwen3-VL/capabilities/grounding-2d.md
- README（vLLM quickstart temperature=0）: https://github.com/QwenLM/Qwen3-VL/blob/main/README.md
- 阿里云百炼视觉理解（无 System Message 建议 / 480-2560 鲁棒区间 / qwen3-vl 默认不思考）: https://help.aliyun.com/zh/model-studio/vision
- 阿里云国际站镜像: https://www.alibabacloud.com/help/en/model-studio/vision

**Doubao/火山方舟官方**
- 视觉定位 Grounding（bbox 标签格式/0-999/多目标 prompt 范式/示例无温度）: https://www.volcengine.com/docs/85679/1544247 （canonical https://www.volcengine.com/docs/ark/grounding）
- 图片理解: https://www.volcengine.com/docs/ark/image-understanding
- 函数调用（关 thinking 提效口径）: https://ark.volcengine.com/docs/85679/1544217

**GLM/智谱官方**
- GLM-5.3-Flash 模型页（temp 1/top_p 0.95/reasoning_effort max/强制思考/128K）: https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash.md
- 深度思考（5.3/5.3-FLASH 不支持 disabled；low 仅 5.3）: https://docs.bigmodel.cn/cn/guide/capabilities/thinking.md
- 对话补全 API 参考: https://docs.bigmodel.cn/api-reference/模型-api/对话补全.md
- 视觉 MCP（无 bbox 提示词，佐证「无 grounding 文档」）: https://docs.bigmodel.cn/cn/coding-plan/mcp/vision-mcp-server.md

**第三方**
- WBF 框融合: https://ar5iv.labs.arxiv.org/html/1910.13302 （实践指南 https://learnopencv.com/weighted-boxes-fusion）
- 小 VLM 测试时扩展（采样+多数决）: https://monurcan.github.io
- VLM 测试时扩展系统研究: https://arxiv.org/html/2604.15376v1 （Zoom Consistency，重复采样作免费置信信号）
- per-object co-teaching 伪标签训练: https://arxiv.org/abs/2511.09955
- VLM 自动标注最佳实践: https://arxiv.org/abs/2506.02359
- MissingBench-Verified（负例 few-shot 证据）: https://openreview.net/forum?id=MissingBench
- Roboflow 零样本自动标注: https://blog.roboflow.com/zero-shot-auto-labeling-with-vlms
- 检测器 TTA 框合并: https://albumentations.ai/docs/4-advanced-guides/test-time-augmentation

## 6. 汇报要点（给主会话）

**Top 发现（≤5）**
1. qwen 官方 cookbook 明示两大限制——**>40-50 实例无限循环**与**密集簇并单框**：前者指出现值 max_tokens=1500 无 finish_reason 监控是盲区（密堆类截断风险），后者证明 garbage_bag.toml 的堆/个体分级条款正是官方推荐的 prompt 级缓解。
2. **doubao 是唯一无低温官方背书的成员**（官方 grounding 示例不设 temperature）——全局 0.0 对它是未验证外推，最优先实验。
3. **GLM 官方推荐 temperature=1/top_p=0.95**（推理剖面）与现值 0.0 直接冲突；且官方确认 5.3-flash 强制思考无降档、无任何 grounding 文档——0-1000 纯标定值，官方零支持。
4. qwen 现值（temp 0、无 system msg、User 消息指令式、逐字 JSON 例、960×540 在官方鲁棒区间内）**全部与官方口径一致**，无需动。
5. 第三方共识给出一个零 API 成本的升级点：**共识框合并从代表框升级 WBF 加权平均坐标**，直接改善 GT 框位置精度。

**最值得做的 3 个实验**：E1 doubao 温度消融 → E2 glm 温度消融 → E3 WBF 融合重放（用已落盘票即可）。
