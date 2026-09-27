# 任务书 M0 执行记录（jxl 会话，2026-09-27 起）

> **M0 烟火独立线**——3+1 模型架构里独立的「+1」（taxonomy D2：smoke / open_flame）。
> 上游依据：taxonomy.md 附录 A（D2 两类）+ design.md §2c（烟火独立线可先行，补 ET10 空档）
> + TRAINING-LOOP.md §3/§5（M0 例外单独处理，公开消防数据 + 本域画像，业务确认已过=Q3）。
> 本文件接续 C-execution.md 的批式结构（**C-execution.md 与 projects/s4det/README.md 禁改，
> M0 记录独立落此**）。数据根：`/mnt/data/s4det/m0-fire/`。

## 硬边界（phase 1）

- 无 GPU、无 VLM API（phase 2 才做）；纯研究 + profile + 候选池装配。
- s4 仓（`~/cc/next/s4`）只读；事件库 `/var/howell/s4/v0.4/event/` 只读。
- Python：`/home/jiang/cc/py/jxl/.venv/bin/python`（带宽周 09-21~27 禁 uv/pip 触网）。

## phase 1（2026-09-27，无 GPU）✅

### 步骤 1 研究定锚 ✅

**三域事件码盘点**（s4 仓枚举 + 事件库实证，单一数据源=枚举源码注释指向）：

| 域 | 类型体系 | 烟火可用码 | 实证 |
|---|---|---|---|
| mobile | `EventType` 1-19（s4-tu `mobile_event.rs`） | **无专属码**；ET10 突发情况=空档载体（无模板无封闭类集） | event_index.jsonl 32,469 条 + 本次全库扫描，**ET10 事件 0 条**（ET15/16/19 同样 0） |
| road | `RoadObjectType` 1-8（部件：消防栓=1、消防通道=4 为**设施**非烟火）+ 部件三态 1-3 | **无**——road 域是部件域，无明火/烟事件语义 | 6,665 事件扫描 0 烟火命中；fire_ext 词族 638 全为消防设施误命中 |
| scene | `TaskType` 201/202/203（prompt 开放目标短语） | **202 动态 prompt 是唯一潜在回流口**（运营可配烟火目标短语） | 11,628 事件 prompt 全为垃圾/堆物/摊位闭环描述，0 烟火命中 |

补充盘点：`emergency2*` 三目录（142 事件）= **ET2 再标注实验**（emergency_type=2，非机动车乱停/无目标），与烟火无关不入扫。`darknet/` = 上游对拍难例池（EVAL-BENCHMARKS 已记）。

### 步骤 2 profiles ✅

`projects/s4det/targets/{fire.toml, smoke.toml}`——**D2 独立类命名空间**（M0 自有类表 0:smoke / 1:fire，与 D1 的 18 类互不相干；序=taxonomy 附录 A「smoke/open_flame」书写序）：

| 字段 | fire | smoke |
|---|---|---|
| name | fire | smoke |
| yolo_text | open flame | smoke |
| vlm_prompt | 明火/火焰本体（火苗/火舌/燃烧点）；排除单独烟雾/灯光反光/红色物体/晚霞 | 可见烟体（烟柱/烟团/冒烟点）；排除水雾/雾气/蒸汽/扬尘/尾气白汽 |
| weights / target_cls / rfdetr_cls_id | 均 None（无在役先验→L1 共识框即标注，同批 2 新类链） | 同左 |
| output_cls_id | 1 | 0 |

验证：`load_target(name, Path(...))` 双 profile 加载通过（pydantic TargetProfile 字段集全对齐）。
**命名决策（待用户否决权）**：taxonomy 叶类 EN 名为 `open_flame`，profile 文件/字段名用 `fire`——
`load_target` 按 `targets/<name>.toml` 查找（name 须=文件名），短别名 fire；别名映射记于此，
不改 taxonomy（中文名权威=taxonomy 表不受影响）。

### 步骤 3 弱标签投影 + 候选池 ✅（**0 命中，如实报数**）

工具（新增，`projects/s4det/bin/`）：
- `m0_scan.py`——三域全库文本扫描（212 日期目录，51,341 事件，~226s）：平台对象 Description + scene Prompt + llm_entries 双模型语句 + confirm.json 确认描述 → 五词族分类（fire/smoke/scene_adj/cig/fire_ext）→ `scan/matches.jsonl` + `scan_stats.json`（含三域 event_type 直方图）
- `m0_pool.py`——matches → 候选池装配（`{date}_{event_id}_r{stem}.jpg` 扁平名 symlink，c_project.py 同款约定）

**结果**（两轮独立口径互证——含/不含 confirm.json 文本，结果一致）：

| 词族 | 命中 | 处置 |
|---|---|---|
| fire（明火/火焰/燃烧/焚烧/篝火…13 词 + en） | **0** | — |
| smoke（冒烟/烟雾/浓烟/烟柱…10 词 + en） | **0** | — |
| fire_ext（消防栓/消防通道…） | 638 | 设施域误命中，仅计数 |
| cig（吸烟/烟头/烟蒂） | 15 | 垃圾/行为域，非 D2 |
| scene_adj（野炊/烧烤/露营/帐篷/钓鱼/垂钓） | 19 | 逐条复核全为**摊位遮阳「帐篷」**（占道设摊/堆物事件），无真实野炊露营场景——与 design §2c「证据链场景样本≈0」判断一致 |

候选池 `/mnt/data/s4det/m0-fire/candidates/`：**0 事件 / 0 图 / manifest 0 行**（README 含完整判据/命令/扩源建议）。
根因：ET10 无模板 → 平台不产烟火事件；scene prompt 无烟火任务配置。**空池=真实现状，非扫描缺陷**（判据可复跑验证）。

### 步骤 4 验收 ✅

- profiles：TargetProfile 加载通过（fire/smoke 各字段值见上表）。
- 候选池：`images/` 0 文件 = `manifest.jsonl` 0 行（计数一致）；0 断链（空池平凡成立）。
- README：candidates/ 目的/来源判据/产生命令/归属四要素齐备；数据根 README.md 已加 m0-fire 结构行 + 操作日志。
- EVAL-BENCHMARKS.md §2 已加「M0 烟火检测器 ❌ 无参照集」行（缺失可见，评估口径=检出率/误报率，T11 预留）。

## phase 2 计划（GPU + VLM 环节，裁决点过闸后启动）

> 前提：TRAINING-LOOP M0 行既定「公开消防数据 + 本域画像」双源。本域源当前为 0（步骤 3 实证），
> phase 2 以公开集为主轴起步，本域链保持幂等待样本。

**支 A：公开集起步（主轴，不受本域空池阻塞）**

1. 数据集选型与下载（**待裁决点 1**；带宽周 09-27 止，下载排 09-28+）：
   首选 **D-Fire**（fire+smoke 两类 ≈9.6k 图，类结构与 D2 同构）；备选 FLAME / FIgLib / wildfire-smoke。
2. 格式对齐：转 YOLO → 类表重映射至 M0 命名空间（0:smoke 1:fire）→ 验收（图-标注计数一致）。
3. `yolo_train.py` 训 M0 v0（批 2 同款骨架）→ 产 weights 先验回填 profiles 的 weights/target_cls
   （此后本域帧可走 det_mine target 票链）。
4. 评估：按 T11 口径（检出率/误报率）+ 公开集 test split；**本域参照集仍 ❌**——EVAL-BENCHMARKS
   M0 行挂着，本域样本到达后按「预标注优先」补建。

**支 B：本域画像（事件回流后，工具链全部复用批 2）**

1. `m0_scan.py` 重跑（幂等，~4 分钟）→ `m0_pool.py` 装配候选池。
2. `b_multirun.sh <out> smoke=<池>,fire=<池>`——**参数序 smoke 在前**（pairs 序=cls 位次，
   与 output_cls_id 0/1 已对齐；序错=names 表与标签 id 错位，数据集级毒）；validators=yoloe,gdino
   （支 A 产出 weights 后可加 target 票）。
3. `c_arbitrate`（VLM 池四成员）→ `c_assemble` cls 保真装配 → M0 v0.x 重训。
4. 隔离台账照旧（j-quarantine-ledger）。

**s4 侧运营提案（跨仓写文档交接，不直接改 s4 仓）**：① ET10 烟火判定模板草案；② scene 202
烟火目标短语任务建议——两者是本域烟火事件回流的前置（**待裁决点 3**）。

## 待用户裁决点

1. **公开集选型**：D-Fire 首选是否批准；获取渠道（GitHub/官方站）与下载体量（~GB 级）。
2. **命名确认**：profile 名 fire（短别名）vs taxonomy 叶类名 open_flame——维持 fire 或改 open_flame（后者需文件名+`load_target` 联动改）。
3. **跨仓提案是否提交 s4 会话**（ET10 模板 + scene 202 烟火任务）——影响本域回流时程。
4. **支 B 触发口径**：本域候选积累到多少帧启动本域链（建议 ≥200 帧，对齐批 1 池 800 的 1/4 起步档；纯提案）。

## 遗留与坑

- **命名冲突警示**：`/mnt/data/s4det/smoke-v0/` 是 M1 管道**冒烟测试**（wave1-smoke-train.md，18 类 4 epoch），与 M0 烟火线无关——M0 数据一律落 `m0-fire/`。
- lexicon.md 形态词典 v0.1 **无烟火词条**（六族均无）——phase 2 有真样本后按网格人审流程补 D2 节（本阶段样本 0，凭空写形态=违反「判断归模型」红线，故不预写）。
- mobile 域 560 事件 `EventType` 字段缺失（扫描直方图 mobile/-1；event_index 同有此象）——数据质量注记，非 M0 阻塞。
- scene_adj 19 事件已在 `scan/matches.jsonl` 留档（野炊证据链 §2c 专题若启动可直接取用，含证据片段）。

---
更新：2026-09-27（phase 1 完成记录 + phase 2 计划 + 四裁决点）。
