# person 全帧标注另线执行预研（提案）

> 性质：**决策材料，未执行任何标注/未动任何数据**（2026-09-27，只读预研）。
> 背景：taxonomy 附录 A 把 person 列为 D1「批 2 直入 12」之一（辅助证据类），执行上不入 18 类链、
> 走全帧标注另线（C-execution 待办行：触发点=window1/T4-b1 两件编辑完成后；裁决点=赶 v0.1 终版
> 合并切分或入 v0.2）。本文件供用户裁决，**不构成开工授权**。
> 纪律：本文件为独立新文件，未改 C-execution.md / README.md / taxonomy.md。

## 1. 现状汇总（全部只读核实，2026-09-27）

### 1.1 类目身份

- taxonomy.md L19（vehicle 表 person 行）：辅助证据类（无粗类锚），批 2 起全量标注，用户 2026-09-25 拍板；
  用途 = 违停状态判定数据源（rider_visible）+ 证据链一级锚点；COCO 底座最成熟类，零风险。
- taxonomy.md L88（附录 A）：D1 类目池 22 = 已训 2 + 批 2 直入 12（含 person）+ 挂验证 5 + 观察候补 3。
- 消费侧实证：rider_visible 是 s4 判定插件字段（`/home/jiang/cc/next/s4/deploy/event/template/mobile/ref/2_tmp.md`
  L27/L92：骑行中 riding 且 rider_visible=true 且不在机动车道 → 不判违规）——person 检出是违停判定链输入。

### 1.2 person 标注现状：零框零 GT

- 批 2 装配集 data.yaml 18 类无 person；批 1 dataset 仅 ebike/garbage_bag 两类 → **两装配集 person 框数 = 0**。
- preannot/manifest_leaf.jsonl 仅 ebike 37,515 + garbage_bag 12,424（person 无锚不投影，C-execution L64）。
- 唯一 person 痕迹：c_vote agree=false 类别投票路由 person 19 行（sidecar，非训练框）。

### 1.3 帧宇宙实测（stem 级去重，本预研磁盘清点）

| 集合 | 路径 | 帧数 | 备注 |
|---|---|---|---|
| 批 2 装配集 | `/mnt/data/s4det/p3-seedset/batch2/mining/dataset/all/images` | 1,268 | 3,683 框/18 类，0 断链 |
| 批 1 dataset | `/mnt/data/s4det/p3-seedset/batch1/mining/dataset/all/images` | 1,481 | 1,403 框/2 类；**706 无框负样本帧**（§10 裁决） |
| window1 人工窗 | `/mnt/data/s4det/human_window/window1/images` | 522 | 与两装配集零重叠；**0/522 已编辑**（等用户） |
| 批 2 负样本池 | `.../batch2/seedset/pools/negative` | 250 | 待终版合并入训 |
| 批 2 新池（pools 五池） | `.../batch2/seedset/pools/` | 3,316 | 装配集 1,268 全部 ⊆ 其中 |
| 批 1 种子池 | `.../batch1/pools/{ebike,garbage_bag}` | 1,600 | dataset 1,481 ⊆ 其中 |
| **派生宇宙** | | | |
| 批 1+2 装配合并 | 上两行并集 | **2,679** | 重叠 70（批 2 复壮链共享池） |
| v0.1 终版训练宇宙 | b1∪b2∪window1∪负样本 | **3,433** | 假设终版合并含全部四源（待终版方案确认） |
| 批 2 共享池全量（spec 口径） | b1pool∪b2pools | **4,774** | manifest 记账口径 4,916=1,600+3,316（未去重）；两批池重叠 142 |

### 1.4 预标注资产

- **四票池**（`src/jxl/bin/vlm_pool.py` L46-73，2026-09-26 换代后）：qwen-flash / doubao-vl / qwen38-local（LAN
  免费）/ glm-flash（仲裁票）——商用 3 + LAN 1。person 是 vlm_pool 原生场景（`call_person` 即通用 grounding 调用）。
- **person 域实测精度**（KB `~/.claude/kb/30-areas/vlm-vision-grounding/20260913-vlm-service-inventory.md` §三）：
  - n001 person 标定（20 图 30 框，F1@IoU0.5）：qwen3.5-35b 思考 0.9508 / qwen3-vl-flash 0.9492 /
    doubao(旧代) 0.9355 / glm 0.7458——**person 是 VLM grounding 最强类**（对比 s4det 域 ebike 0.72-0.81）。
  - 35 帧难例排序：LocateAnything-3B 0.800 > yoloe 0.763 > qwen3-vl-plus 0.661（代际不同仅作相对参考）。
- **检测器票**（det_mine `VALIDATOR_BACKENDS`，全部本地）：yoloe-11l-seg.pt（`models/`，COCO person 原生类）/
  gdino（HF grounding）/ rfdetr / la（LocateAnything-3B，`script/la-serve.sh` 本地 :18306，BF16 ~7.4G 显存）。
  **LA license 注意**：NVIDIA 非商用——KB 定位「仅内部验证/审核用途」，宜作 EV 审核票而非 GT 生产票。
- **person target 先验**：`/home/jiang/cc/py/jxl/yolo26s.pt`（COCO 底座，person 类原生）。**v0mini 不可用作
  person 票**——已两类化微调（ebike/garbage_bag），COCO person 头已失。
- GPU：本机 RTX 4060 Ti 16G（当前余 ~11.6G）可跑 yoloe+gdino（LA 需 7.4G，可排期或上 ds）。

### 1.5 关键机制约束

- 规范 §2 全标红线（annotation-guideline.md）：**禁止「留在图里但不标」——假阴性监督毒化模型**。person
  全帧标注的穷尽性要求由此而来：任何入训帧上可见而未标的人 = person 类毒样本。**负样本帧对 18 类是负、
  对 person 可能是正**（706+250 负样本帧同样必须过 person 标注）。
- 框法：可见框原则（CrowdHuman 路线，规范 §3）——person 行人遮挡/截断画法已有成文规范。
- 双闸审计每轮必跑（TRAINING-LOOP 约束 7）；隔离台账纪律（约束 6）。
- 预算张力：TRAINING-LOOP 4c-3「每类每轮新增 ≤~1,000 标注帧」——person 全帧标注天然超（宇宙 ≥3,433），
  但成本模型不同（检测器打底近零边际），需按「另线不计类链预算」解释或用户豁免（裁决点 4）。

## 2. 轴 A：帧宇宙对比

| 宇宙 | 帧数 | 四票全量 API 调用 | 混合案 API 调用* | 与 D1 训练集关系 | 评价 |
|---|---|---|---|---|---|
| A1 批 2 装配集 | 1,268 | 5,072（商用 3,801） | ~760-1,520 | 只覆盖批 2 有框帧；v0.1 若含批 1/window1/负样本则**穷尽性破缺**（其余 ~2,165 帧未标 person = 毒） | **内部不一致**：单独成立仅当 v0.1 排除批 1（与复壮链矛盾）。仅适合作冒烟/标定子集 |
| A2 批 1+2 合并 | 2,679 | 10,716（商用 8,037） | ~1,600-3,200 | v0.1 宇宙的主干，但缺 window1 522 + 负样本 250 → 终版合并后仍破缺 | 同上，不够 |
| A3 v0.1 终版训练宇宙 | 3,433 | 13,732（商用 10,299） | ~2,060-4,120 | **与 v0.1 训练集严格相等** → person 可安全入 v0.1 | 赶 v0.1 的最小充分宇宙；依赖终版合并方案确认（四源全入假设） |
| A4 批 2 共享池全量 | 4,774 | 19,096（商用 14,308） | ~2,950-5,900 | taxonomy「批 2 起全量标注」**spec 口径**；超集覆盖 A3（差 ~1,341 帧属池内未装配帧） | 一次到位、v0.2 直用；机器段大头免费，较 A3 仅多 ~30-60min |
| A5 新采样 | — | — | — | 无必要：池已按锚组建，person 无锚本就以全池为域 | **不推荐**（造轮子） |

\* 混合案 = 检测器共识全量打底（本地免费）+ 四票池仅仲裁 review 帧（按批 1/2 实测 review 率 15-30% 估）。

**轴 A 结论**：宇宙只有两个真实候选——**A3（赶 v0.1 的最小充分）与 A4（spec 全量）**；A1/A2 有穷尽性破缺
风险（毒标注），A5 无必要。A4 是 A3 的超集且增量成本极低（检测器段免费），若机器段全量跑则 A3 可从中
直接切出、零返工。

## 3. 轴 B：预标注方案对比

| 方案 | 构成 | API 成本（A4 全量） | 质量面 | 工具面 |
|---|---|---|---|---|
| B1 四票池逐帧 grounding（同 t4_preannot 模式） | 每帧 × 4 模型 grounding → consensus_boxes | **19,096 调用**（商用 14,308），wall ~2-4h | person 是 VLM 最强类（0.93-0.95）；但密集人群穷尽性弱于检测器（VLM 偏显著目标，远/边缘人易漏→ 假阴性风险）；四票含 glm 0.75 单票噪声 | t4_preannot 同款，改 TARGETS 即可，~0.2 天 |
| B2 纯检测器共识（无 VLM） | det_mine 链：target=yolo26s（target_cls=person）+ validators yoloe,gdino → L0/L1 共识 | **~0**（全本地） | COCO 双强先验召回好；但 review 帧无仲裁 → 人工量大或违「宁缺毋毒」 | person.toml + b_multirun 一行，**零新工具**（与 18 链同构） |
| **B3 混合（推荐）** | B2 检测器共识全量打底 + 四票池仲裁 review 帧（c_arbitrate 同款）+ LA 作 EV 审核票（查漏标，非 GT 生产——license 合规位） | **~2,950-5,900 调用**（商用 2,213-4,426），wall ~0.5-1.5h | 检测器保召回（穷尽性红线）+ VLM 仲裁保精度 + LA EV 补共同盲区（MV+EV 双验收，TRAINING-LOOP 4b-5）；person 域 VLM 强 → 仲裁质量高 | c_arbitrate/det_mine 全复用；新增宇宙 symlink 池装配 ~0.5 天 |

**轴 B 结论**：**B3**。B1 成本 3-6 倍于 B3 且穷尽性（红线项）反而弱；B2 省钱但 review 无裁违反既有仲裁纪律。
B3 与「局域网优先、商用按需抽检」的成本约束（TRAINING-LOOP 约束 8）一致，且 person 有 target 先验
（yolo26s+target_cls 机制正是 2026-09-26 det_mine 重构后字段），机制上与 18 链完全同构。

**先导步（无论 B 几都做）**：person 域 T7 式标定——yolo26s person 框 vs 四票池对拍 100-200 帧，测
检测器共识对 person 的 F1 与 person 出现率（密度未知，votes 路由 person 19 行是弱证据）；LA 以审核
身份同场对拍（license 合规）。标定不达标再回退 B1 加权。

## 4. 轴 C：时序对比

| 方案 | 排期 | 增量工期 | 人工触点 | 风险 |
|---|---|---|---|---|
| C1 赶 v0.1 | 两件编辑完成后 person 线插入终版合并前：先导标定 → A3 宇宙链跑 → 仲裁 → 双闸 →（人工小窗 40-120 帧）→ 终版合并切分（19 类）→ v0.1 重训 | **+1-1.5 天**（工具 0.5 天 + 机器 ~2-3h + 仲裁 ~1h + 合并改造），v0.1 发布顺延同期 | +1 个人工小窗（估 40-120 帧 ≈ 15-40min，批 2 仲裁人工率 8% 外推） | ① v0.1 的 person 指标**无 T4 覆盖**（T4-b1 只有 ebike/garbage_bag）→ 按 WARNING-1 精神只能带警示出数；② 用户在两件编辑（747 帧）之外再加窗，编辑疲劳；③ 终版合并方案需为 person 改造（cls 保真追加第 19 类） |
| C2 入 v0.2（推荐） | v0.1 按原计划（回收→合并→重训→T4 评估）；person 线 v0.1 后启动：A4 全量 → T4-b2 装配时**加 person strata**（T4-b2 触发条件「批 2 种子集完成」已满足，400 帧档）→ v0.2 合并 19 类重训 | 0（不占 v0.1 关键路径）；v0.2 前需 ~1-1.5 天同 C1 机器量 + 全量增量 | 人工小窗与 T4-b2 人审**同批搭车**（一次会话） | v0.1 无 person（违停判定链晚一轮拿到数据源）；若 v0.2 遥远则 person 交付滞后 |

**轴 C 结论**：**C2**。决定性理由是 T4 覆盖：C1 出的 v0.1 person 指标没有参照集背书（T4-b1 无 person），
与「T4 未覆盖的口径一律不作质量结论」的项目第一纪律冲突；C2 让 person 与 T4-b2 同批，指标出生即有锚。
次理由：v0.1 关键路径不被拉长、用户编辑负担不加码。C1 的真实收益只有「省一轮重训」，但 v0.2 本来就要
为挂验证类裁决重训。

## 5. 推荐案（一句话）

**B3 混合预标注 + A4 批 2 共享池全量 4,774 帧 + C2 入 v0.2（与 T4-b2 同批搭车）**；机器段（先导标定 +
检测器共识打底）属「能自动运行」范畴，裁决通过后即可后台跑，A3 子集可随时从中切出供 v0.1 应急。

## 6. 裁决点清单（留用户）

| # | 裁决点 | 选项与推荐 | 影响 |
|---|---|---|---|
| 1 | **时序**（即既有待办行的裁决点） | C2 入 v0.2（**推荐**）/ C1 赶 v0.1（+1-1.5 天 +1 人工窗，person 指标带 WARNING） | v0.1 范围与发布日 |
| 2 | **帧宇宙** | A4 全量 4,774（**推荐**，spec 口径）/ A3 v0.1 宇宙 3,433（赶 v0.1 时最小充分） | 机器时长与 v0.2 就绪度 |
| 3 | **预标注池构成** | B3 检测器打底 + 四票仲裁（**推荐**）/ B1 四票全量 / B2 纯检测器；**la 票角色**：EV 审核票（**推荐**，license 合规）vs GT 生产票（非商用限制） | API 成本 3-6 倍差、license 合规 |
| 4 | **类链预算豁免** | person 全帧标注是否豁免 TRAINING-LOOP 4c-3「每类每轮 ≤~1,000 帧」（**建议豁免**：另线模式、检测器打底边际近零，在 TRAINING-LOOP 补一行注记） | 预算纪律口径 |
| 5 | **人工兜底形态** | review 帧人工小窗（**推荐**，40-120 帧与 T4-b2 同会话）/ 纯机器 + 双闸放行（person 成熟类可论证，但违人工兜底惯例） | 人工量 vs 纪律 |
| 6 | **T4-b2 加 person strata** | 加（**推荐**，预标注 4 票 × person 帧数，量级 ~1,600 调用内）/ 不加（person 永久无 T4 锚） | person 能力账与指标背书 |

## 7. 成本估算汇总

- **API 调用**（推荐组合 B3+A4）：先导标定 ~400-800 + 仲裁 ~2,950-5,900 + T4-b2 person strata ~800-1,600
  ≈ **4,200-8,300 调用**（商用占 3/4 ≈ 3,150-6,225；对照 B1 全量单段即 19,096）。
- **机器时长**（4060 Ti 本机）：yoloe 全量 ~5-10min；gdino ~15-40min；LA EV 审核全量 ~1.5-2.5h（7.4G 显存，
  与业务共卡需查余量）；VLM 仲裁 wall ~0.5-1.5h。合计一个下午量级。
- **人工**：先导标定抽检（若做纯人工种子）+ review 小窗 40-120 帧（15-40min）+ T4-b2 person 帧
  （strata 配额内，与既有人审同场）。
- **工具**：person.toml profile + 宇宙 symlink 池装配脚本（c_seedset 模式）+ 合并层第 19 类追加
  （c_assemble 扩一位）≈ 0.5-1 天；均为既有件变体，无新依赖。

## 8. 证据索引（绝对路径）

- taxonomy person 行：`/home/jiang/cc/py/jxl/projects/s4det/taxonomy.md` L19、L88
- 另线待办与触发/裁决点：`/home/jiang/cc/py/jxl/projects/s4det/tasks/C-execution.md` L64、L92、L119、L126
- M1 类目池与预算约束：`/home/jiang/cc/py/jxl/projects/s4det/TRAINING-LOOP.md` L31、L96（4c-3）
- 全标红线与可见框：`/home/jiang/cc/py/jxl/projects/s4det/annotation-guideline.md` §2、§3
- 四票池配置：`/home/jiang/cc/py/jxl/src/jxl/bin/vlm_pool.py` L46-73；预标注模式：`projects/s4det/bin/t4_preannot.py`
- person 域 VLM/检测器实测 F1：`~/.claude/kb/30-areas/vlm-vision-grounding/20260913-vlm-service-inventory.md` §三
- 检测器后端与 license：`/home/jiang/cc/py/jxl/src/jxl/bin/det_mine.py` L50；`/home/jiang/cc/py/jxl/script/la-serve.sh`
- rider_visible 消费方：`/home/jiang/cc/next/s4/deploy/event/template/mobile/ref/2_tmp.md` L27、L92
- 帧宇宙清点（2026-09-27 实测）：`/mnt/data/s4det/p3-seedset/batch{1,2}/`、`/mnt/data/s4det/human_window/window1/`
- v0.1 宇宙 3,433 = b1(1,481)∪b2(1,268)∪w1(522)∪neg(250)，b1∩b2=70；全量 4,774 = b1pool(1,600)∪b2pools(3,316)，∩=142

## 修订日志

- 2026-09-27 v1：预研成文（只读；未执行标注、未动数据、未改三份禁改文档）。
