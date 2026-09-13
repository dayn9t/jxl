# iapx → jxl/sgcc：分类器需求与样本供给

> 建立：2026-09-12 ｜ 维护方：iapx 侧（`~/cc/py/iapx`，会话代理）
> 用途：让 jxl/sgcc 了解 iapx（会话切分原型）的模型需求、已交付样本的位置与对接方式。
> 样本已就绪（28,498 张，见 §2）；两个分类器需求（§3/§4）+ 一个检测器登记项（§5）。

## 1. 背景与消费关系

iapx 是 n001 收费窗口的**会话切分原型**（检测 → 上半身分类 → ROI 在场 → ReID 关联 →
分段 → session 导出），当前消费 sgcc 项目训练的两个模型：

| 模型 | 部署物 | iapx 侧角色 |
|---|---|---|
| person 检测器 | `2026-09-13_person_n.pt/.onnx`（v4，09-13 23:19 上线） | crop640 域 person 检测；重复框 1,267→7，消费端 IoU≥0.95 防御仍建议保留 |
| upper_body 分类器 | `2026-09-10_person_upper_n.pt/.onnx`（主力） | 行走下半身过滤（0.5 阈） |

iapx 产出根：`/mnt/data/jiang/ws/iapx/n001/`；源视频（只读）：
`/var/howell/iap/v0.9/ias/sh-sgcc/n001/video/{source_id}/0/{date}/*.mkv`。

## 2. 样本供给（已就绪，2026-09-12）

**位置**：`/mnt/data/jiang/ws/iapx/n001/samples/`

- `samples/{source_id}/{date}/{HH-MM-SS.mmm}.jpg`——**原始 I 帧，1920×1080，不裁切不叠框**；
  文件名 = 帧墙钟时间（与 session 目录时间格式一致，天然唯一）
- **28,498 张 / 8.0 GB**（177 个检测 cache 的 52,343 个 I 帧中**含人**的帧；0 失败 0 缺源；
  跨源同墙钟重叠去重 -12）
- 日期覆盖：**2026-07-01 / 07-02**（主语料）+ **07-04 / 07-31**（跨日期验证日，多样性白赚）
- **manifest.jsonl**（28,498 行，行=帧）：

```json
{"file": "samples/1/2026-07-01/08-00-02.000.jpg", "source_id": 1,
 "date": "2026-07-01", "wallclock": "08-00-02.000", "stem": "08-00-02.000",
 "mkv": "2026-07-01/08-00-02.000.mkv", "pts_secs": 0.0, "n_persons": 1,
 "detections": [{"bbox": [823.9, 560.2, 1138.9, 640.0],
                 "box_norm": [0.4291, 0.5187, 0.5932, 0.5926],
                 "confidence": 0.727, "upper_body": false}]}
```

**对接惯例**：`box_norm`（/1920×1080 四位小数）+ `stem`（jpg 去扩展名）按 jxl labels
jsonl 的 `{uid, stem, box, attrs}` 设计——manifest 可直接机械转换为标注输入（uid 建议
`{stem}_{框序}`）。`bbox` 为全图像素坐标（源 cache 原值）。

**预筛入口**（jq 示例）：

```bash
# 行走下半身负样本候选（模式 D 素材，见 §3）
jq -c 'select(.detections[]?.upper_body == false)' manifest.jsonl | wc -l
# 双人帧（身份分类器素材，见 §4）
jq -c 'select(.n_persons >= 2)' manifest.jsonl | wc -l
```

**增量供给**：后续日期随时可再产（`uv run python -m iapx.bin.export_samples`，幂等重入
只补新帧）；需要更多日期/源请找 iapx 侧。

## 3. 分类器需求 A：upper_body 重训（行走下半身负样本）

- **现状缺陷（iapx 实测）**：行走下半身 crop（无头胸）置信 **0.50-0.95 波动**，0.5 阈下
  随机放行 → 「腿可见即开段」（错误 session 起点）；真上身 crop 置信恒 1.000
- **根因**：原训练负样本 = 全帧下半带裁剪，未覆盖**紧致行走人腿 crop** 分布
- **需求**：补行走下半身负样本重训。素材 = manifest 中 `upper_body=false` 的帧（crop 内
  仅腿/下半身，jxl 侧重新标注框级标签）
- **验收建议**：行走腿类负样本测试集置信显著低于部署阈；既有真上身测试集无回归
- **部署后**：iapx 侧做「重分类 pass」（对已缓存 crop 重判，不重检测）；**注意** iapx 的
  cache 指纹将补分类器模型版本字段（当前缺——换分类器不失效缓存是隐患，修复中）

## 4. 分类器需求 B：身份分类器（三分类，新族）

- **类别**（用户裁决 2026-09-12）：
  1. **男性引领员**（引导/指座类工作人员）
  2. **女性保洁员**
  3. **默认客户**（其余一律）
- **边界**：服务窗**内**的业务员不在 iapx 的 crop/检测范围（crop640 域天然排除），不考虑
- **用途**：双人帧**主目标选择**——工作人员非关注目标（辅助作用），主目标 = 非引领员
  非保洁者优先；治最大失败族 A（staff 干扰）
- **素材锚点**（内容观察层有完整描述：`/mnt/data/jiang/ws/iapx/n001/annotations/content.json`）：
  - 保洁员：src2 2026-07-01 10-2x（拖地+擦台面，两段）
  - 引领员：src2 2026-07-01 10-32（指座位引导客户）、15-24（开场即 staff）
  - staff 在座（非办理）：src2 2026-07-01 14-13、15-06

### 4.1 供给答复（2026-09-13 14:40，对应 `~/cc/next/iap-s2/docs/2026-09-13-jxl-role-classifier-sample-supply.md` §3.1）

**Phase-1 窗表（立即可用，4 日期；来源 = audit VLM 理由挖矿 + 当日用户 L1 人工确认）**：

| role | source | 日期 | 窗 | 置信来源 |
|---|---|---|---|---|
| cleaner | 1 | 2026-07-01 | 13:45-13:47 | **用户人工确认**（围椅拖地 12s）|
| cleaner | 2 | 2026-07-01 | 10:21-10:24 | v1 锚点 o14 + VLM 复述（同页）|
| leader/引导 | 1 | 2026-07-03 | 08:50-08:56 | **用户人工确认**（白衣工作人员站椅侧办理）|
| leader/引导 | 1 | 2026-07-03 | 10:00-10:08 | **用户人工确认**（引导黑衣男入座交接）|
| ~~leader?~~ ✗ | 1 | 2026-07-01 | 09:35-09:39 | ~~VLM（臂章工装）~~ **已被 jxl spark 复核推翻**（全判 customer/not_person，不入 leader）|
| staff 在座（A4 边界类）| 2 | 2026-09-03 | 10:01-10:34（三段同person：红短袖+黑马甲+左臂纹身）| VLM（同person 连续在场 33min）|
| 业务员（罕见，暂不入类）| 2 | 2026-09-03 | 11:00-11:02 | **用户人工确认**（修叉笔笔座）|

**Phase-2（今晚 T9 全量后补满）**：06-22/23、07-04、07-06、07-31 五日 ~600 新 session 的 audit VLM 理由同法挖矿 → 目标凑齐 cleaner ≥8 窗 / leader ≥8 窗、日期 ≥5（仅差 1 日即达标）。
**manifest 已扩**：`export_samples` 增量已跑（07-03/07-05/09-03 金丝雀日帧并入 samples/，含上表所有窗的快照图与 bbox）。
**注意**：§3.3 v1 锚点窗之外，上表 07-03/09-03 全部为新日期新现场多样性；staff 在座/业务员窗按词典边界判例归 uncertain/不入正类，由 jxl 侧裁决。

**jxl 侧消费注记（2026-09-13 15:1x，spark 复核后）**：
1. **manifest 缺口**：上表声称「07-03 已并入」——实测 `manifest.jsonl` 无 07-03 行（jpg 已在
   `samples/1/2026-07-03/`，缺 manifest 行即无 bbox 可 join）；09-03 的 300 帧仅覆盖
   09:20-09:27 与 11:30-11:34（金丝雀段），**staff 在座窗 10:01-10:34 零帧**。请 iapx 侧重跑
   `export_samples` 增量补 07-03 全日 + 09-03 10:01-10:34 段。
2. **Phase-1 已提 2/5 窗**（84+200 crop），spark 复核（`iapx_role_seed_review.py`，verdicts
   已并入 `seed_verdicts.jsonl`）：p1a 净增 **cleaner +8**；p1d（leader? 工装臂章窗）
   **被复核推翻**——200 帧全判 customer/not_person，无引导动作证据，不入 leader 正类
   （臂章工装若属第三工种，按词典 residual 扩展规则处理，素材留 uncertain 池）。
3. 净效果：cleaner 唯一源 44→52；leader 仍 104（押在 07-03 两窗与 Phase-2）。
4. **补窗复核（同日 15:4x，manifest 闭合后）**：p1b/p1c（07-03 leader 两窗，1,025 crop）
   净增 **leader +14 / cleaner +12**（verdict 口径）；p1e（staff 在座 1,482 crop）绝大
   多数 customer/uncertain，符合词典边界预期（其中 verdict=leader 3/cleaner 2 系 spark
   与词典分歧，v2 build 前人工抽查后定）。**累计唯一源：cleaner 64 / leader 118**。
5. **给 Phase-2 的校准**：分钟级窗粒度命中率仅 ~2-4%（窗内大部分 crop 是同场客户），
   量级达标靠窗数不现实——Phase-2 请优先用 audit VLM 理由文本挖矿（直接命中保洁/引导
   动作帧，粒度=帧而非窗）；07-03 新日期多样性价值已验证（leader 首次覆盖第三日期）。
6. **疑点 session 线丰收（同日 17:4x）**：16 条审计疑点段（`pairs/role-candidate-sessions.jsonl`）
   全量 7,150 crop 经 spark 复核——**cleaner +115 / leader +13**（新日期 06-22/23、07-04/06
   等，staff 干扰富集判断实证有效）；uncertain +472（复审池）。**累计唯一源（verdict 口径）：
   cleaner 181 / leader 134**（含 p1e 分歧 5 待抽查）。剩余缺口走全 manifest 双人帧标注线
   （新 6 日期 n_persons≥2 ∧ upper_body，est ~40k crop，spark 过夜扫）。
7. **全 manifest 标注线完成（同日 21:4x）**：55,011 crop（新 7 日期双人帧∧upper_body）全部
   spark 复核——净增 **cleaner +142 / leader +111**。**全局累计（68,093 verdict 行）：
   cleaner 295（≈达标 300）/ leader 183（缺口 117；两线 crop 有重叠，全局累计为去重后值）**；uncertain 池 5,541（人工复审富矿，
   可再捞稀有类）。v2 决策：cleaner 已达标；leader 183 较 v1(104) +76%，先训 v2 实证
   recall 提升幅度再定是否继续扩窗（假设-实证循环）。

**iapx 侧回填（2026-09-13 15:20）**：①p1d 推翻接受——Phase-1 表 09:35 臂章工装窗**划除**（下表中已标 ✗）；②manifest 缺口已闭——增量运行完成后 `manifest.jsonl` = **101,503 行**（07-03 全日 19,239 ✓；09-03 7,307 含 10:0x 段（staff 窗现值 1,569 帧 ✓；2,708 为 mid-run 快照口径）；jxl 所见空窗为运行中快照）；③**07-31 仅 320 行**（与检测预跑竞态）——预跑完成后 iapx 补跑一次增量，届时 Phase-2 挖矿一并交付；④06-22/23、07-04、07-06 已提前入 manifest（15k/8.7k/12k 行）——Phase-2 素材池现在就大于承诺。

**Phase-2 终报（2026-09-13 17:10，T9 全量后）**：①新日（06-22/23、07-04、07-06、07-31）audit 理由**零直接 role 命中**——首尾帧采样对动作类角色天然低概率，窗口挖矿到此为数据上限；②**替代供给 = 全量 10 日 manifest**（top-up 后 ~130k 帧行，07-31 补满）+ **新日 30 条审计疑点 session 名单**（same=false 27 + null 3，staff/干扰富集段——jxl 侧优先对这些 session 的 crop 跑 VLM 标注，命中率远高于全扫）；③T9 分段 322 sessions 全 10 日就绪、GT F1 0.9677 逐位复现——session 边界数据可直接用于时段定位。结论：≥300 唯一源的正路 = jxl 全 manifest 标注线 + 新日期多样性，窗口表已完成历史使命。
- **流程**：照 jxl 既有（标注 → VLM 审核 → 训练）；素材 = §2 全集重标注（单人双人帧都要）

## 5. 检测器章节：无需补样本 + 一个登记缺陷

> **【已裁决 2026-09-12】** 归因=训练集近重复 GT 教坏 one-to-one 头（非导出/非推理
> 封装/非 NMS——YOLO26 无 NMS）。证据：3 证据帧 PT/ONNX 复现一致；dataset_v3 train
> 含 73 对 IoU≥0.99 + 264 对 0.95-0.99（集中于 cam1 2026-06-22 亚像素对，共识融合层
> 缺帧内近重复守卫）。处置：iapx 消费端防御维持；person 下次重训前以
> `jxl.bin.dedup_gt_boxes` 清洗 GT，重训后用重复对清单回归预期清零（09-13 已按全量重扫清单 1,267 帧执行，见归因报告 §6）；
> `consensus_dataset` 补帧内近重复守卫防再发。全文：
> `research/2026-09-12-检测器重复框归因.md`
> **附带发现（同日 VLM 审计）**：空场景帧存在检测 FP（如 src1 08-00-02 台面边沿
> conf 0.727，GT 15 窗未覆盖的时段）——iapx 会话守卫建议对 ub=false 且低置信框叠加
> 时域持续性过滤；分类层 not_person 选项见词典 §4。

**主结论：person 检测器在 n001 窗口域无样本不足类问题**，证据：

| 证据 | 数字 |
|---|---|
| GT 召回（15 visits 精标） | R = 1.0，零漏检 |
| 误检（双负窗 VLM 逐帧比对） | 100% 对齐，零幻影框 |
| 构建稳定性 | 177/177 cache 成功（零推理失败） |
| 跨日期泛化 | 09-03 / 07-04 / 07-31 正窗全一致 |
| 边界精度 | session 端点误差 p50 = 1.9s |
| 困难形态 | 椅后站立/横躺/行走半身/双人紧贴/弯腰拖地全部正常检出 |

**登记缺陷：同帧重复检测（来源未定，请 jxl 侧排查）**

- 现象：同帧两条**亚像素级重合** bbox 的重复检测（IoU≈1.0，conf 如 0.773/0.506）。
  全量扫描：**52 帧 exact 对（IoU≥0.99，占含人帧 0.18%）+ 194 对 near（0.95-0.99）**，
  共 60/177 个 cache 受染
- **归因（重要）**：YOLO26 为 **NMS-free 端到端架构（one-to-one head），没有 NMS**——
  「逃过 NMS」的说法不成立。候选来源：one-to-one 头偶发重复发射 / ONNX 导出 /
  usls 推理封装后处理。**非样本不足**
- 请求：排查训练/导出链路，裁决后处理归属（部署图内置去重 or 消费端各自防御）
- 证据帧（cache JSON，det 双条同 bbox）：
  - `/mnt/data/jiang/ws/iapx/n001/cache/2/2026-07-02/11-30-02.000.json` 帧 `11-30-40.010`
  - `/mnt/data/jiang/ws/iapx/n001/cache/1/2026-07-01/08-40-03.000.json` 帧 `08-43-43.079`、`08-45-57.109`
- iapx 侧已消费端去重防御（IoU≥0.95 保高 conf，commit `12b2b96`）；危害实证：重复框杀
  单人流守卫 → 同人被切两段（已修复）

## 6. 模型回流契约

- 部署线照旧：训练 → `embed_contract` → ONNX + symlink 切现网（回滚 symlink 还原）
- iapx 侧验收：GT（15 窗）+ benchmark（版本化用户裁决）双口径回归；OSNet 强 embedding
  接入后 ReID 阈值会重定标（见 §7，2026-09-13 裁决 B）
- 对接人：iapx 会话代理（Claude Code，dayn9t 的会话）；样本/证据增量随时可请求

## 7. 需求 C：OSNet 域微调（ReID 治本，2026-09-13 用户裁决 B）

> 背景：通用 MSMT17 权重在 n001 窗口域定标门**七格全败**（2 权重 × 3 测量法，最优
> gap −0.016；same/diff 余弦在 0.54-0.75 带重叠不可逾越）——通用权重判别上限不足，
> 治本 = 域微调。证据全套：`/mnt/data/jiang/ws/iapx/n001/gt/osnet-calibration*.md`
> （4 份）+ `osnet-evidence-run{,2}.md`。期间生产/金丝雀继续 handcrafted HSV@0.8 基线
> （GT F1 0.9677）作过渡。

### 7.1 基座契约（勿选错变体）

- **基座权重**：`osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`
  （`/mnt/data/jiang/ws/sgcc/person/osnet_weights/`）
- **架构 = BN 版 OSNet x0_75，非 AIN**（实测钦定权重无 InstanceNorm 键，AIN 定义
  strict load 440 missing；BN 定义 0/0 直载）。参考实现：`~/cc/py/iapx/src/iapx/vendor/osnet.py`（MIT vendored BN 定义）
- 推理契约（微调后不变）：输入 `N×3×256×128`、ImageNet mean/std 归一、无 TTA、
  512-d 未归一化输出（运行时 L2 归一化）；ONNX 导出脚本 `iap-s2/script/export_osnet_onnx.py`
  （剥 fc 头、动态 batch、自检）——Rust 侧现存 AIN x1_0 ONNX 与本基座非同模型，回移须重导

### 7.2 训练数据构造（iapx 供 pair list，jxl 侧切图）

**crop 来源 = §2 samples/ 全帧 + manifest bbox**（`box_norm` 直接换算），无需另行截图。
**✅ pair list 已交付（2026-09-13 11:41，iapx `540b26e`）**：`/mnt/data/jiang/ws/iapx/n001/samples/pairs/`
——`train.jsonl` 418 对（session-anchor 正对 204 / session-distant 远距负对 207 /
person-change 换人硬负对 7）+ `eval.jsonl` **106 对 held-out**（same 77 / diff 29，v2c
交叉核验清洗后逐对一致）+ README；行 schema `{label, origin, a, b}`，a/b =
`{mkv, time, stem, det_index}`，join manifest（键 mkv+wallclock）取 `detections[det_index].box_norm`
——40 行抽验 join 零缺失。L0 采信 69/76 session（37+32）。构造规则（契约化）：

| 集合 | 构造规则 | 规模（现语料） | 用途 |
|---|---|---|---|
| **正对（same）** | 同 session 内取首/中/尾检测对；**仅 L0 审计采信的 session**（VLM same=true 且 conf≥0.95 且不与归档裁决矛盾）——剔除模式 C 残余标签噪声 | ~70 session × 3 对 ≈ 200+ 对 | 训练 |
| **负对（diff）** | **跨 session 且时间间隔 >1h 或跨日**——严禁相邻 session 对（benchmark 6 个合并组 = 同人被切成相邻 session，相邻负对必错标）；从 GT 换人 6 处 + audit same=false 真疑点 4 处补充边界硬负对 | 数百对起 | 训练 |
| **裁决对（held-out）** | v2c 定标的 106 对（qwen3.5-35b 异源 VLM 交叉核验清洗后；缓存 `gt/calibrate-v2-checks/` 115 键） | 106 对 | **验收专用，不进训练**（防评估污染） |

随 T8/T9 全量跑完，session 数 75→800+，正对规模免费扩大一个量级（pair list 增量重产）。

### 7.3 训练建议（jxl 域，仅供参考）

- 微调起点 = §7.1 基座（非 from-scratch——域数据量级撑不起）；triplet/circle loss +
  小学习率（基座名内 amsgrad/labelsmooth/flip/jitter 即原配方线索）
- **类别 = person identity（每 session 一 id）**，不是二分类——ReID 要的是度量空间
- 防过拟合：裁决对 held-out（见上）+ 早停看 held-out same_p5/diff_p95 间隔

### 7.4 验收门（两段，iapx 侧执行）

1. **定标门（硬）**：v2c 非对称端点测量法复测（旧端 = 窗内≤5帧主 det 均值 L2 gallery、
   新端 = 单帧；工具 `iapx/gt/calibrate.py`）——**same_p5 > diff_p95**（分布完全分离；
   七格全败时最优 gap −0.016，微调目标把它翻正）
2. **管道门**：GT 15 窗 F1 ≥ 0.9677 且换人 0、过切 ≤2；benchmark 六合并组 4/6 → 5-6/6
   （G3 黄昏区应拆开）；v3 审计 4 真疑点逐窗验证

### 7.5 交付与回流

- 交付物：微调后 `.pth` + ONNX（§7.1 导出脚本）+ 训练指纹（数据日期集/LR/epoch）
- iapx 侧：`reembed_cache` 全量换嵌入（分量校验通道，~20min）→ 重定标 → 验收门 →
  cfg 轴 tag 建议命名 `osnet-x0_75-ft-v{n}`
- 时序：不阻塞金丝雀/全量（A 过渡先行）；微调权重到达后作为独立飞轮圈次验收

**jxl 侧 OSNet v1 结论（2026-09-13 19:0x）**：`osnet_ft.py` v1 已训（332 对/59 ids/
150 crops，circle loss，早停 patience=10）——**best_gap = −0.565 vs 未调基线 −0.4995
（恶化）**，早停正确兜底。实证结论：pair list 现量级撑不起度量学习（§7.3 风险预判
命中）。**v2 触发条件 = iapx 用 T9 全量 322 sessions 增量重产 pair list**（正对 204→
~966 量级）后重训；脚本/管线已验证可复用（build→train→eval 全通，远端路径契约
crops.jsonl 需带路径替换——已修）。期间生产维持 HSV@0.8 过渡基线（§7 原决议不变）。

**jxl 侧 role v2 交付（2026-09-13 23:5x）**：素材扩至唯一源 cleaner 295 / leader 183 后重训
（train 1,792/2,120/1,602/2,384，过采样对齐 customer 量级）。test（含副本口径）top1 0.8414，
分类别 recall 对比 v1——**leader 0.496→0.719（+22pt）**、cleaner 0.667→0.686、customer
0.942→0.971、not_person 1.000→0.997。主混淆=稀有类→customer（residual 吸走，素材多样性
不足）。**部署候选已 stage：`2026-09-13_person_role_n.pt/.onnx`**（新族无旧 symlink，
iapx 对接时直接指向；**names 契约（字母序权威）**：0=cleaner, 1=customer, 2=leader,
3=not_person——注意 customer 在 leader 之前）。词典线 0.80 未达（leader 差 8pt），
路径=uncertain 5,541 人工复审捞底 + 新日期增量 → v3。分类用途为主目标软选择（降权），
v2 已显著优于 v1，建议 iapx 先行接入。

**jxl 侧 role v3 交付（2026-09-14 03:2x，七分类）**：词典 §6 扩类后重训（uncertain 池
5,541 零人工复审——四模型投票 5,000 张 + GLM 视觉仲裁 285 张 split 残量）。test top1
**0.8926**（v2 0.8414）。分类别 recall：**cleaner 0.879（+19.3pt 首次达 0.80 线）**、
**teller 0.875（新类首训达线）**、leader 0.775（差 2.5pt）、customer 0.967、not_person
0.991；manager/security 素材不足（test n=3/1；train 唯一源 ~10/~7 扩采中）仅供参考。**部署候选已 stage：
`2026-09-14_person_role_n_v3.pt/.onnx`（七类 names 字母序：cleaner/customer/leader/
manager/not_person/security/teller）**。建议 iapx 直接对接 v3（跳过 v2——v2 的 customer
类被制服人员污染 ~90% 已由七分类纠正）。leader 最后 2.5pt 缺口路径：security/manager
采集窗顺带挖引导动作帧 + uncertain 头顶层放弃池不再捞。
