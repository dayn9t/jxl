# s4det — s4 事件域目标检测器

> **▶ 项目恢复（2026-09-25 晚）**——用户三裁定落地（taxonomy v1 冻结 / EXEMPT-S 确认 / Q1=最小先行），批 2 开工。仍待用户：T4-b1 225 帧标注窗口、doubao 控制台开通（非硬阻塞）。

为 s4 智慧城管事件域训练统一目标检测器：把 mobile/road/scene 三域全部事件相关目标检测出来，
供判定层（LLM/judge）消费。数据飞轮复用 jxl+vlabel 共识标注体系（n001/sgcc 沉淀）。

## 关键指针

- 事件数据根（弱标签源，只读）：`/var/howell/s4/v0.4/event/`（mobile 33k 事件/138k 图；road 6.7k 事件/36k 图；scene 11.6k 事件/38k 图；2026-05~09）
- 训练数据根（产物落盘）：`/mnt/data/s4det/`（已建，P0 产物在 `p0-cluster/`；README 在数据根，j-dir-readme 规范）
- s4 仓库：`/home/jiang/cc/next/s4`（prompt/枚举/事件管线单一数据源）
- 共识标注方法论：skill `j-consensus-labeling`（九阶段流水线 + 审计 + 形态词典）
- 盘点基础：`research/2026-09-25-目标分类学与数据源盘点.md`（本目录首份资产）

## 操作归属（跨仓分界，2026-09-25 用户裁决：文档交接制）

| 会话 | 职责 |
|---|---|
| **s4 会话**（/home/jiang/cc/next/s4） | 事件域知识、类目语义、需求/规格/验收文档、数据弱标签语义解读、s4 仓代码改动 |
| **jxl 会话**（/home/jiang/cc/py/jxl） | jxl/vlabel 代码改动、共识 pipeline 运行、训练、数据盘产物管理 |
| 交接面 | 本目录 `projects/s4det/`：s4 侧写需求/盘点/类目文档，jxl 侧写执行记录/工具；双向挂链 |

读操作（探查 jxl 能力/产物）两侧自由。跨仓写操作不直接进行——s4 会话只在本目录落文档。

## 状态里程碑

| 日期 | 里程碑 | 状态 |
|---|---|---|
| 2026-09-25 | 项目启动；目标分类学枚举 + 数据资产盘点 + jxl 能力盘点 + 外部标注实践调研完成 | ✅ |
| 2026-09-25 | 三项关键策略拍板（D1 小目标全标+推理过滤+ROI 灰化 / D2 堆叠混合式 / D3 mobile 先行）+ 总体设计 v1 | ✅ 设计待审 |
| 2026-09-25 | P0-1 平台码表逆向完成（CODEBOOK：11=非机动车单辆确证、12 待定三假说、码表按域分表）；P1 标注规范 v1 成文；任务书 A/B 成文 → **任务书 A 已交 jxl 会话执行** | ✅ / 🔄 jxl 执行中 |
| 2026-09-25 | **任务书 A 执行完成**（jxl）：五粗类 4,508 crop + 聚类（阈值三档存证 + k-means k=16 主档）+ 簇代表网格 6 页落 `/mnt/data/s4det/p0-cluster/`。关键发现：码 12 七簇全为路缘/地面区域框（CODEBOOK 假说③ 压倒性）；平台框噪声簇可观（石球/栅栏入框）；ET15/16/19=llm 类别锚点非 EventType。偏差与参数见 `tasks/A-execution.md`。**s4 已人审：码 12 定性 + taxonomy v1-草案命名（33 入训叶类）** | ✅ 完成 |
| 2026-09-25 | **任务书 B 执行完成**（jxl，`tasks/B-execution.md`）：B① 弱标签抽取器 56,999 行落 `preannot/`（agree 率 10.9%，与 CODEBOOK 全量交叉吻合）；B② 多类方案裁决 **(a) 类独立链+数据集层合并**（取舍记录+profiles+b_multirun.sh，ebike/garbage_bag 双类端到端冒烟过）；B③ n001 report 断链修复（13,788 图分级冒烟过）；B④ 数据根三区规划 + 台账迁移闭环（579 条已裁决回写） | ✅ 完成 |
| — | 两级类目表 v1（taxonomy v1-草案已出，**待用户审批冻结**） | ✅ **2026-09-25 用户批准定版（v1 冻结）**——brainstorming 三裁定同日落定：EXEMPT-S 同意豁免、Q1 人工窗口=最小先行（T4-b1 先行，119 兜底并入批 2 人工窗） |
| 2026-09-25 | **批 2 前置完成**（jxl，`tasks/C-execution.md`）：c_vote 类别投票工具——872 行 agree=false 自动定类 633（72.6%）/ split 239 入隔离台账高优池 / abstain 0；票据 sidecar 供批 2 挖掘回填类别先验 | ✅ |
| 2026-09-26 | **VLM 池换代**（用户新近度准则触发）：doubao-seed-2-1-pro-260915 接入（key 下已开通，旧端点关闭非阻塞）、qwen38-local 别名统一；T7' 重标定 0.808/0.741/0.724 三强票 + glm 0.482 仲裁；det_mine target 票重构（weights 可选 + target_cls 过滤 + _target_nms）——修复两类 GT 中毒 | ✅ |
| 2026-09-26 | **批 2 挖掘+仲裁+装配全链完成**（jxl）：18 链挖掘 → c_arbitrate 仲裁（735 帧：GT 677 / 人工池 58）→ c_assemble cls 保真装配 **1,268 帧 / 3,683 框**（磁盘验收 0 断链）；ebike 918/car 965/shared_bike 648/bicycle 447/tarp_covered 211…低产类按硬条件挂验证 | ✅ |
| 2026-09-26 | **人工窗口 window1 装配**（c_humanpack）：522 帧（仲裁残量 58 + 批 1 兜底 review 464）候选框预放，X-AnyLabeling 工程就绪 `/mnt/data/s4det/human_window/window1/` | 🔄 **待用户编辑**（连同 T4-b1 225 帧，两件编辑落地后自动链：回收→终版切分→M1 v0.1 重训→T4 评估） |
| — | 标注规范 v1（堆叠/小目标/遮挡细则，含用户拍板项） | ✅ 成文（P1，待用户确认） |
| — | mobile 弱标签抽取 ✅（B①）+ 共识标注种子集 P3（**等用户批准 taxonomy 冻结后开工**，jxl 会话） | 🔄 第一批执行中 |
| 2026-09-25 | **P3 第一批执行完成**（jxl，`tasks/C-execution.md`）：ebike+garbage_bag 全链跑通——投影 49,939 行 → 挖掘 1,600 帧 → VLM 仲裁消解 83% review 帧（T7 重标定 qwen-flash F1 0.875）→ 终版 **1,481 帧/1,404 框** → 8:1:1 切分；逐类 gate 达标（1,100/303 框）。双闸审计干净。**等人工兜底 119 帧**（contact sheet 已备）；doubao 端点关闭待控制台开通 | 🔄 待人工兜底 |
| 2026-09-25 | 检测器 v0-mini 训练（D0，弱标签口径）+ EVAL-BENCHMARKS 缺口清单建成（j-eval-benchmarks） | ✅（质量结论等 T4-b1 人审窗口） |
| — | 飞轮循环（mine → 重标 → 重训）常态化 + 隔离台账 | ⬜ |

## 文档地图

- `design.md` — **总体设计 v1**（D1-D4 决策表 + 类目体系 + 标注规范要点 + P0-P5 阶段计划 + 评测参照集 + 首批任务书提要）
- `annotation-guideline.md` — **标注规范 v1**（全标原则/可见框/区域类/堆叠层叠细则/信任级/ROI 灰化/负样本，P1 交付物待用户确认）
- `research/2026-09-25-目标分类学与数据源盘点.md` — 三路调研汇总 + 用户需求存档 + 外部实践 + 细分权衡 + 候选类目草案
- `tasks/` — s4→jxl 任务书（A 聚类跑批 ✅ `tasks/A-execution.md` / B 基建 ✅ `tasks/B-execution.md`）
- `data/platform-codes/` — 平台码表逆向（CODEBOOK.md 结论 + 原始计数，脚本在 s4 仓 py/tools/reveng_platform_codes.py）
- `taxonomy.md` — 两级类目表（**v0 草案骨架**，未冻结禁作标注依据；等任务书 A 网格回来看图命名→用户审批冻结）
- `lexicon.md` — 形态词典 v0（文本源草案：各叶类形态要点/203 词频直证/TEXT 直证；图证待聚类网格回填）
- `docs/EVAL-BENCHMARKS.md` — 评估参照集盘点（早期骨架：❌ 缺口警告区/逐组件分级/待建 T4-T8；**人工 GT 建成前内部指标不得当质量结论**）
