# wave1：负样本池 250 装配执行记录（jxl 会话 s4det，2026-09-27）

> 主链被用户两件人工编辑（T4-b1 225 / window1 522）gating 期间的不被 gating 前置件。
> 产物：`/mnt/data/s4det/p3-seedset/negpool/v1/`。本记录为独立文件（并发纪律：不改 C-execution.md / README.md）。

## 1. 规格核对（250 的来源口径）

- **250 明文出处**：`tasks/C-execution.md` L62——批 2 种子集配额「vehicle 1,000 / garbage 1,000 / vendor 666 / phenomenon 400 / **负样本 250**」（全局 ~4,916 ≈ 5,000 预算，用户定规模）；L118/L126「终版合并切分（含**负样本池 250**）→ M1 v0.1 重训」。
- **负样本原则**：`annotation-guideline.md` §10——无目标事件图保留入训练集教「没有」；删框留图；**不人为配平正负比例**。
- **口径辨析（关键）**：c_seedset 装配时的 `negative` 池 250 帧（`seedset/pools/negative/`）入选判据仅为「事件不在正样本池 ∪ 不在批 1」——**无 ET 过滤、无任何无目标检测证据**（同 ET 落选事件亦会入池，含目标风险高），且从未被任何挖掘链跑过。本装配不沿用该 250，按任务质量红线以**挖掘证据**重建（见 §2）。种子集 negative 池 250 帧现状见 §6 遗留。
- **结论**：明文口径（数量 250、用途=终版合并切分、§10 原则）与证据重建无冲突，按任务规格推进，未动既有数据。

## 2. 证据判据（多方「无目标」，非「没被链选中」）

**链覆盖实况**（比任务书预估更强）：批 2 共享池各池实际被多链全量挖掘——

| 池 | 帧数 | 挖掘链 | 每帧模型视角 |
|---|---|---|---|
| vehicle_pool | 1,000 | 5 链（ebike 复壮[v0mini target+yoloe+gdino] + bicycle/car/shared_bike/tricycle[yoloe+gdino]） | 最多 11 |
| garbage_pool | 1,000 | 7 链（garbage_bag 复壮 + carton/foam_box/garbage_pile/leaves_pile/plastic_loose/tarp_covered） | 最多 16 |
| vendor_pool | 666 | 6 链（construction_fence/piled_goods/scaffolding/shop_display/stall_cart/vendor_booth） | 最多 12 |
| phenomenon_pool | 400 | 0（Q2 VLM 后置不训，未挖掘） | 无证据 |
| negative（种子集） | 250 | 0 | 无证据 |

**帧级合格判据**（与 `c_assemble.py` 正样本装配口径严格互补，即批 1 已审计背书的 706 负样本同款 det_mine 语义）：

- 帧在其池**全部**链上均「无共识框」= 无 ≥2 模型一致的框：
  - `level==L0` 且（复壮链 target 为空）——含「孤立检出」帧：单模型检出不过共识门即不成框（gdino 单独看到 = 亚共识幻觉候选，det_mine L0 丢弃语义）；
  - `level==L1` 且该链 `labels/<stem>.txt` 为 0 字节（有争议但共识框为空）；
  - `level==review` → 不合格（仲裁 GT=正样本 / 人工池 58=待裁，两态都非负样本）。
- 排除：已装配 `batch2/mining/dataset` 帧（1,268）、仲裁人工池（`arbitration_human.jsonl` 58）、批 1 已训帧（`batch1/.../dataset/all` 1,481——与批 2 池存在 44 帧 stem 级重合，见 §5 缺陷）。

**合格数量：1,324 帧**（vehicle 111 / garbage 636 / vendor 577）——远超 250 配额，**无硬凑**。各池账目全平：vehicle 882(in_ds)+2(human)+5(b1)+111(qual)=1,000；garbage 301+27+36+636=1,000；vendor 85+4+577=666。

## 3. 采样与装配

- **工具**：`/home/jiang/cc/py/jxl/projects/s4det/bin/c_negpool.py`（新建，ruff 全绿；幂等——重跑 manifest md5 一致已验证）。
- **采样**：域配额尽量均匀 vehicle 84 / garbage 83 / vendor 83（余数给合格量最少的域）；域内按月尽量均匀（2026-05~09）；**月内按孤立检出总数升序**（宁缺毋毒：被任何模型看见东西越少越安全）+ seed=42 随机 tie-break。
- **选中分布**：月度 05:4 / 06:29 / 07:91 / 08:34 / 09:92。
- **产物** `/mnt/data/s4det/p3-seedset/negpool/v1/`：
  - `images/` 250 symlink → `batch2/seedset/images/<stem>.jpg`（原始帧单源）
  - `labels/` 250 个 0 字节 .txt（一帧一文件）
  - `classes.txt` = 批 2 dataset 18 类逐字节一致（ebike…vendor_booth）
  - `manifest.jsonl` 250 行（stem/pool/month/孤立检出数/逐链 level+raw 证据）
  - `qualified_pool.jsonl` 1,324 行全量合格帧 + selected 标记（**超额 1,074 帧为储备非弃置**，后续轮次可直接扩取）
  - `README.md`（j-dir-readme：目的/判据/命令/归属 jxl 会话 s4det/日期/消费方）

## 4. 验收证据（命令 `.venv/bin/python` 内联脚本，2026-09-27 16:20）

| # | 项 | 结果 |
|---|---|---|
| 1 | 三方清点 | images=250 = labels=250 = manifest=250 行 ✓ |
| 2 | 断链（symlink 全链 resolve） | 0 ✓ |
| 3 | 空标签逐文件 | 250/250 全部 0 字节 ✓ |
| 4 | stem 双射 | images==labels==manifest，250 唯一 ✓ |
| 5 | classes 对齐 | 与批 2 classes.txt 逐字节一致，18 行 ✓ |
| 6 | 独立性 | ∩dataset=0 ∩人工池58=0 ∩批1已训=0 ∩window1=0 ✓ |
| 7 | 储备账目 | qualified_pool 1,324 行 selected 250 == manifest ✓ |
| 8 | 抽样复验 | 随机 5 帧重跑全链判据，违例 0 ✓ |
| 9 | 幂等 | 重跑 manifest md5 一致 ✓ |

**方法论坑（记录）**：symlink 的空标签验收不可用裸 `find -size 0`（find 默认 lstat 量 symlink 自身长度恒非 0，批 1 切分目录曾因此误读）——必须 `Path.stat()`（跟随链接）或 `find -L`。

## 5. 过程发现的既有缺陷（未修，待裁决）

**c_seedset.py 批 1 事件排除是 no-op**：`exclude_events` 取 `p.stem.split("_")[2]`，但批 1 池实际文件名为 3 段 `{date}_{event_id}_r{rid}`（注释按 4 段 leaf 前缀写的）——取到的是 `r3` 段而非 event_id，排除从未命中。后果：批 2 池与批 1 已训帧存在 **41 帧 stem 级重合**（garbage 36 + vehicle 5 + vendor 0；另有批 1 池内 119 人工帧所在事件的其它帧未在 stem 级统计内），批 2 实际新增略低于 3,316 记录值。**本负池已排除全部 41 帧**（验收项 6）。修复建议：`split("_")[1]`；但**重跑 c_seedset 会改变种子集构成、使 18 链挖掘产物失配**——建议批 3 装配时修，本批不动（惜改保形不适用于带病代码，但重跑代价是数据链级的，须用户裁决）。

## 6. 遗留与建议

1. **种子集 `negative` 池 250 帧无证据悬置**：`seedset/pools/negative/` 现存 250 帧从未被挖掘，含同 ET 落选事件（潜在含目标）。建议批 3 将其并入对应池跑链取证后再论去留；在此之前不得入任何训练集。未弃置（仍在种子集原位），无需隔离台账。
2. **本装配零弃置**：超额 1,074 帧合格负样本为储备（qualified_pool.jsonl 可索引）；review 待裁 33 帧已在 window1 人工池轨道——均无声丢弃风险为零，故本任务未新增隔离台账行。
3. **已知局限（如实）**：①跨族盲区——vehicle 池帧只被车辆族 5 链查过，图内若有垃圾/摊贩族目标不可见（批 1 706 负样本同款局限，审计已接受；缓解=终版训练后 EV 审计（vlm_only 挡）+ 飞轮误报回流）；②共同盲区——全部模型都漏的目标负池会继承（§10 自然分布原则下接受，T4 对拍校准）；③孤立检出升序偏好使选中帧偏「干净」，难负样本（近失误）占比低——如需可从储备按 raw_det_total 降序补难例版 v2。
4. **消费指引**：终版合并切分时 negpool/v1 的 images+labels 直接并入（stem 与批 2 dataset 无冲突已验）；classes 18 类对齐，无需映射。
