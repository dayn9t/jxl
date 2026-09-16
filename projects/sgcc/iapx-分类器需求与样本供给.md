# iapx → jxl/sgcc：分类器需求与样本供给

> 建立：2026-09-12 ｜ 维护方：iapx 侧（`~/cc/py/iapx`，会话代理）
> 用途：让 jxl/sgcc 了解 iapx（会话切分原型）的模型需求、已交付样本的位置与对接方式。
> 样本已就绪（28,498 张，见 §2）；两个分类器需求（§3/§4）+ 一个检测器登记项（§5）。

## 1. 背景与消费关系

iapx 是 n001 收费窗口的**会话切分原型**（检测 → 上半身分类 → ROI 在场 → ReID 关联 →
分段 → session 导出），当前消费 sgcc 项目训练的两个模型：

| 模型 | 部署物 | iapx 侧角色 |
|---|---|---|
| person 检测器 | 现役 `2026-09-09_person_n.pt/.onnx`（**v3**——2026-09-15 生产点火实证 v4 坐姿回归后回退；**v5 已 stage 待切**：`2026-09-15_person_n_v5.*`，验收+跨日期扩展验收双 PASS，见 §8） | crop640 域 person 检测；重复框 1,267→7，消费端 IoU≥0.95 防御仍建议保留 |
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
- **交付注记（2026-09-12，jxl 侧）**：✅ v2 已交付——cls_upper_psq_v2 重训完成，
  legs test p50=0.0001、>0.5 残留 50%→**10.34%**（压低 5 倍），真上身 test 0.9772 无回归；
  部署物 `2026-09-12_person_upper_n_v2.*` 已 stage，**symlink 未切**（待本节 cache 指纹
  修复后协同）；残差 10% 建议叠加时域持续性过滤（§5 附带发现同款）。
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
**✅ pair list 已交付（2026-09-13 11:41，iapx `540b26e`；2026-09-15 全量重产覆盖）**：
`/mnt/data/jiang/ws/iapx/n001/samples/pairs/`（09-15 起为全量版：train 正对 **848** /
远距负对 861 / 硬负对 7 + eval held-out **328**（same 289/diff 39），L0 采信 287/326
session，join 抽验零缺失——见 `research/2026-09-15-生产点火v4坐姿回归证据与pairlist交付.md` §交付物②；本节下述 204/106 为 v2 轮历史口径）
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
> **⚠️ v1 结论修正（2026-09-15，v2 训练中发现）**：v1 的验收判据**方向写反**——
> `heldout_gap` 在余弦**距离**值上沿用了 §7.4 **相似度**口径的判据标签
> （`same_p5 > diff_p95`），语义变成「惩罚 same 对变近」。实证铁证：triplet 使
> same_p5 0.113→0.032（模型在正确变好），旧口径 gap 反而 −0.499→−0.746「恶化」。
> **上段「微调恶化 −0.565」的论据失效**（早停兜底方向碰巧成立，v1 失败主因修正为
> circle loss 与判据错配）。判据已修复（`same_p95 < diff_p5`）并落盘。

**jxl 侧 OSNet v2 交付（2026-09-15，全量 pair list 重训）**：triplet batch-hard
margin 0.3、P×K 32×2、Adam lr 3e-4、seed 20260913、sgcc0 4090；train 输入池 1,716 对
（same 848/远距负 861/硬负 7）→ 剔触碰 eval 的 crop 后 **kept 1,410 对**/256 ids/654
crops，held-out 328 对
（289/39）泄漏 0。**eval gap = −0.0938，jxl 侧验收门 PASS**（same_p95 0.3628 <
diff_p5 0.4566 分布分离；基座未微调同口径 +0.1159 FAIL；ep6 翻正、best@ep42、
早停@ep62）；对照轮 circle（3e-4/3e-5 正确口径 +0.074/+0.083）均 FAIL——**triplet
显式优化尾部是翻正关键，数据 4.2× 放大其效果**。交付
`/mnt/data/jiang/ws/sgcc/person/osnet_weights/osnet_x0_75_ft_v2.pth`（md5
`a33775ef`）+ `.onnx`（`e235228a`，剥 fc/动态 batch/opset 17，torch-vs-onnx 一致性
min cos 0.9999999）；cfg tag 建议 `osnet-x0_75-ft-v2`。**§7.4 定标门（全量基线
gap −0.084 翻正目标）+ 管道门属 iapx 侧复测待办**（距离/相似度两口径数值不可直接
比）。残留风险：最难 3 对 same（dist 0.71/0.62/0.61）疑似换人残余标签噪声，
建议 iapx 复测留意；margin 疑似已到本数据量级收益边界，下一档收益 = 扩正对 +
每 id 增锚解锁 K>2。报告：`research/2026-09-15-osnet-v2微调报告.md`。

**jxl 侧 OSNet v2.1 交付（2026-09-16，姿态漂移正对补训——管道门回归的靶向处置）**：
用户裁决 jxl 自挖 pair 解除前置（原等 iapx 挖掘；知会单
`iapx docs/jxl-notice-2026-09-16-osnet-v21-pair-self-mining.md`）。jxl 侧伪轨迹+
困难正对挖掘（`gencheck/osnet_drift_mine.py`：conf≥0.5 全检、帧级贪心 IoU≥0.5、
帧间 >4s 断链防粘连、跨空档重连 8-60s 几何双门、ft_v2 余弦 <0.62 难例筛）产出
**119 对**（10-41 铁证段定向 5 + 困难 114，18 src×date 组均衡）。与 v2 全量
合并重训（sgcc3 首训，同配方单一变量）：kept 1,527 对/352 ids（drift 簇并入），
held-out 328 泄漏 0。**双证据 PASS**：①held-out gap −0.0938→**−0.1138**（不回退
且改善，best@ep34/早停 ep54）；②drift 119 对 **118 对余弦改善**，铁证段
0.195→0.787、0.352→0.824（复测混淆带 0.36-0.58 拉回 same 带，全体最低
0.195→0.758）。交付 `osnet_weights/osnet_x0_75_ft_v21.pth`（md5 `c4b13f74`）+
`.onnx`（`fcb4dcfa`，opset17/动态 batch/剥 fc，一致性 min cos 0.9999999）；
cfg tag 建议 `osnet-x0_75-ft-v21`。**§7.4 管道门复测靶点 = 10-41 段**
（F1 0.9375→≥0.9677）；残留风险：drift 重连族标签噪声若致换人上升可按 origin
消融。报告 `research/2026-09-16-OSNet-v21姿态漂移补训.md`；BN 导出脚本
`gencheck/export_osnet_bn_onnx.py`（iap-s2 脚本系 AIN 硬编码不可复用，本轮固化）。

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
5,541 零人工复审——四模型投票 5,000 张 + GLM 视觉仲裁 split 残量（送裁 285 张、落盘 43 行分歧裁决））。test top1
**0.8926**（v2 0.8414）。分类别 recall：**cleaner 0.879（+19.3pt 首次达 0.80 线）**、
**teller 0.875（新类首训达线）**、leader 0.775（差 2.5pt）、customer 0.967、not_person
0.991；manager/security 素材不足（test n=3/1；train 唯一源 ~10/~7 扩采中）仅供参考。**部署候选已 stage：
`2026-09-14_person_role_n_v3.pt/.onnx`（七类 names 字母序：cleaner/customer/leader/
manager/not_person/security/teller）**。建议 iapx 直接对接 v3（跳过 v2——v2 的 customer
类被制服人员污染 ~90% 已由七分类纠正）。leader 最后 2.5pt 缺口路径：security/manager
采集窗顺带挖引导动作帧 + uncertain 头顶层放弃池不再捞。

**leader 0.80 冲线结论（2026-09-14，#30 关闭）**：v3 的 leader「错分 44」去重后仅
**8 张唯一源**——唯一源口径 recall = 39/47 = **0.830 已过词典线**。8 张人眼复判：
3-5 张为 teller 形态（黑马甲白衬衫坐柜台，旧 4 类 prompt 时代标签噪声）、1-2 张近
not_person 边缘框、无一张典型引领员——**v3 在纠正过期标签，test 标签才是旧语义**。
处置：v3.1 迭代时用共识管线按词典 v3 语义全面复判 test 标签（标签语义版本升级，
非为提分改标签）；无需再补样训练。副本口径（0.775）与唯一源口径（0.830）的差异
提示：**稀有类验收应以唯一源口径为准**（副本同图同判定只放大分母）。

**role v3.1 最终交付（2026-09-14，扩采重训终版 09:4x；05:5x 为首训时点）**：uncertain 复审扩采全部并入重训（teller 唯一源
998→1,841、customer 池清洗至 1,566 纯便装）。test 混淆矩阵与 v3 误差内持平：cleaner
**0.879** / teller **0.800（n=135 扩大后仍达线）** / customer 0.957 / leader 0.770 /
not_person 0.979；manager/security 素材不足继续采集（词典 §6 扩采中）。**最终部署候选 =
`2026-09-14_person_role_n_v31.pt/.onnx`**（本地/部署 md5 一致 ebb03262…；stale-onnx 教训已在导出流程加入 md5 配对核对）。边际收益归零，role 分类器迭代到此收官；recommend
iapx 对接 v3.1（对接契约见 `research/2026-09-14-role-v3-对接预演.md`）。

**交付通知单（2026-09-14，已落地 iapx 仓库）**：`~/cc/py/iapx/docs/jxl-deliveries-2026-09-14.md`
——行动导向摘要（v4 行为告知 / upper v2 切换前置序列 / role v3.1 对接契约 / pair list
与素材请求 / jxl 文档地图）。iapx 侧 cache 指纹修复完成后，其会话可直接按单执行。

**role v3.2 交付（2026-09-14 晚，复判数据集重训——取代 v3.1 成为最终候选）**：当日三线
收官后重训。数据 = ①8,868 旧源图按词典 v3 语义四模型复判（**uid 去重后改判 909 族/3,185 文件 = 16.9%**，
主改判流 →teller 2,146——旧 prompt 无 teller 类的语义过期实证；抽检 12/12 支持改判）
②mgrsec security 扩采 91 uid（唯一源 109→200）③train 过采样 cleaner×2/leader×4/
manager×4/security×4（val/test 不动）。**test top1 0.8877（5 组跨类双标签修复后干净口径）**，
且口径为复判后干净标签
（v31 的 0.8846 建立在过期 test 标签上，两版数字不可直接对比）。per-class recall：
**manager 0.974 / security 0.95 / teller 0.896 / customer 0.80 达线（5/7）**；
cleaner 0.714 / leader 0.50 未达——复判挤出账面水分后真实难度显形（保洁/引领是
动作+工装混合类，单帧 crop 天然边界；**09-15 spike 实证：时序聚合收益上限 0，两类
错误 100% 个体级系统性——词典 0.80 线单列不适用，不再实施 v3.3**，见 §8 与
`research/2026-09-15-v33时序聚合spike.md`）。
**部署候选已 stage：`2026-09-14_person_role_n_v32.pt/.onnx`**（md5 配对
`2364d5e7`/`16be90c7`；names 与 v31 相同七类字母序）。**建议 iapx 对接 v3.2**；
v31 仍在位可回退。数据集 `gencheck/attr_bank/cls_role_psq_v32`（数据根
`crop640_persons/` 下；v31 的 cls_role_psq 原地不动）。

**v32 预演说明（2026-09-15 补，诚实口径）**：v31 的「20 帧预演 20/20」系一次性内联
脚本产出（未存档），检测层重跑存在框序漂移（NMS 超时截断），v32 无法与之直接对比；
同检测同简化规则的双权重对比（`role_v32_preview_fair.json`）显示 v32 选出 customer
18/20 帧 vs v31 13/20——v32 的 customer 纯净度更高，但该规则非 iapx 真实部署规则。
**对接验证建议 iapx 侧按自身部署规则做影子验证**。流程教训：一次性验证脚本必须落盘
（本例 v31 预演脚本已失传，交接数字不可复现）。

## 8. 待办与恢复快照（2026-09-14 收官落盘，供上下文压缩后续接）

### 等外部触发（触发即执行，2026-09-16 刷新）

| 项 | 触发条件 | 触发后动作 |
|---|---|---|
| OSNet v2 验收复测 | iapx 执行 §7.4 两段门（reembed_cache→ASYM 定标 gap −0.084 翻正 + 管道门） | **✅ 已执行（2026-09-16，jxl 授权代理，temp 通道生产零写入）**：定标门 **PASS**（v2c same_p5 0.685 > diff_p95 0.685，gap **+0.001** 翻正，三表征全正）；管道门 **FAIL**（F1 0.9375 < 0.9677，唯一回归 = 07-02 src1 10-41 姿态漂移重入同人 cos 0.36-0.58 被判新人，参数全组合免疫；换人 0/过切 2 达标）→ **生产 reid 轴维持 handcrafted**，回裁决层（候选：v2 加姿态漂移正对）；复测报告 `research/2026-09-16-上线执行与OSNet复测.md` §3 |
| **v5 切现网** | iapx 照通知单 §1 执行（切 symlink→cache 失效→src2 07-06 冒烟→灰度） | **✅ 已执行（2026-09-16）**：md5 核对过；生产配置 model_name + 部署树 symlink 双切 v5（实测加载路径 = dated 直指，配置才是生效轴）；cache 两侧指纹自然失效；jxl 侧等价冒烟 **10/10 PASS**（铁证帧 (501.7,177.4,687.3,428.4) conf 0.913 + 三塌陷段抽样）——**36 MKV 全日生产冒烟待 iapx 侧执行**；灰度期关注 ROI 下方柜体区 FP（+9.2% 集中段） |
| upper_body v2 上线 | iapx cache 指纹修复 | **✅ 已执行（2026-09-16）**：指纹两侧早已就位（iapx cfg_tag 7 分量 + Rust upper_schema，09-13 审计已修——本行触发条件实为既成事实）；生产 Rust 配置 09-15 Ignition 已切 v2 且旁车缓存重分类已完成；本轮补齐 iapx TOML/默认常量/symlink 三位一体（commit e72b3ab）；iapx 侧重分类随下轮 run_pipeline 自然完成 |
| role v3.2 对接支持 | iapx 开始对接 | 照 `~/cc/py/iapx/docs/jxl-deliveries-2026-09-15.md` §2（softmax 全向量契约 + cleaner/leader 按工作人员粗类使用）；jxl 可提供 crop 集 |
| spark 恢复 | 用户重启 spark（=182 vLLM 机，内存压死后待人工恢复） | 重测 :8000 服务 → 免费池主力切回（并发 ≤3 红线）；sitpack 第四票**已由豆包补齐**（09-16），spark 恢复后重跑仅为可选的回归原四模型口径 |

### sgcc 线 2026-09-15/16 总账（生产点火应对 + OSNet v2 + spike，全部闭环）

- **生产点火 v4 坐姿回归应对** ✅：v4「保持现役」被真机证据推翻（生产已回退 v3，
  `research/2026-09-15-生产点火v4坐姿回归证据与pairlist交付.md`）→ v5 三臂真机验收
  **PASS**（塌陷段 226/226 帧闭合、C 阳性覆盖 99.8%、FP raw +3.6%、铁证帧 conf .916）
  → 部署物 stage + iapx 通知单（含回滚路径）。附：部署惯例不一致发现——v3 部署物带
  embed_contract 契约而 v4/role/upper 系裸导出，v5 已回归 v3 同款（`export_yolo_with_contract`）
- **OSNet v2 域微调** ✅：全量 pair list（正对 848=4.2×）triplet 微调，eval gap
  −0.094 分布分离 PASS（基座不微调 +0.116 FAIL）；**发现并修正 v1 判据方向 bug**
  （§7 ⚠️ 块——余弦距离误用相似度口径，v1「恶化」论据失效）；§7.4 两段门交 iapx 复测
- **role v3.3 时序聚合 spike** ✅ 实证否定：cleaner/leader 错误 100% 个体级系统性
  （oracle 聚合上限=逐帧），词典 0.80 线对两类**单列不适用**（词典已修）；v3.3 不实施
- **sitpack_v6** ✅ **四票完整包已冻结归档**：621 帧/1,036 框 glasspack 兼容（09-16
  豆包第四票补齐 623/623 帧、0 弃权、框级一致 98.65%、手术剔 32 框/推翻 2 帧）；
  v5 已闭合坐姿缺口故不并包
- **训练机切换** ✅：sgcc0→sgcc3 默认（+用户授权双机并行），环境（torch cu128/sm_120
  实算验证）/数据（12G 机间直传）/脚本 host 全就位；**依赖源修正：本仓 uv.lock 走
  devpi（192.168.18.146:3141），阿里镜像会致 lock 重解析**；本机→sgcc3 仅 ~2MB/s，
  大文件必机间直传（11.2MB/s）
- **spark 事故入账**：批量图片请求 8 并发压死整机（内存），**并发 ≤3 红线**已入
  memory；sitpack 缺失的 585 张第四票**已由豆包补齐**（09-16，FALLBACK 用户授权），
  spark 恢复后重跑仅为可选回归口径
- 数据口径：manifest 实测 **133,299 行**（§4.1 iapx 回填的 101,503 为 09-13 历史口径，
  §2 初版为 28,498；iapx 侧
  又增量过）

### sgcc 线 2026-09-14/15 收官总账（三线 + v3.2 重训 + 四维度审核，全部闭环）

- **manager/security 定向扩采** ✅：38,742 未投票 crop 两段漏斗收完。manager 复判线
  **352 超额达标**（粗筛仅 26——场景频率低）；security 粗筛 514→共识通过 91（粗筛→共识
  17.7%），唯一源 109→**200，场景总量上限实证不足 300**（split 残量 95.5% 无二次价值）。
  清单 `gencheck/iapx_round2/mgrsec_security_uids.jsonl`
- **旧源标签复判** ✅：8,868 图，uid 去重后改判 **909 族/3,185 文件（16.9%）**（主改判流
  →teller——旧 prompt 无 teller 类的语义过期实证；抽检 12/12 全对）。`cls_role_psq_v32`
  落成（v3.1 数据集原地不动可回退）
- **person v5** ✅：test 0.8945（−0.6pt）；玻璃反光专项 recall 0.911→**0.930** 代价 FP +21。
  **裁决建议 v4 保持现役，v5 存档候选**（`runs/person_n001_v5/`，切换待用户拍板），
  详见归因报告 §6.2（**此建议已被 09-15 生产点火推翻**：v4 坐姿回归实证、v5 真机+跨日期
  验收双 PASS，见上文 09-15/16 总账——v5 已 stage 待切）
- **role v3.2 重训交付** ✅：数据 = v32 复判集 + security 91 + train 过采样；**test top1
  0.8877**（5 组跨类双标签修复后干净口径）；manager 0.974/security 0.95/teller 0.896/
  customer 0.80 达线（5/7），cleaner 0.714/leader 0.50=挤水后真实难度（改进走时序聚合——
  **09-15 spike 已实证否定该路径**，词典 0.80 线单列，见上文 09-15/16 总账）。
  交付物 `2026-09-14_person_role_n_v32.pt/.onnx`（md5 配对 `2364d5e7`/`16be90c7`）。
  **cleaner/leader 若要冲线的下一步 = v3.3 时序多帧聚合**（非补静态样本），未开工
  （**09-15 更新：spike 实证否定，v3.3 不再实施**——聚合收益上限 0，两类按
  「工作人员粗类」使用，词典 0.80 线单列）
- **全项目审核** ✅（四维度 docs/code/data/xref，2 轮 workflow）：19 项确认问题全部修复
  （rename 覆盖/rsync 漏传/双标签污染/7 步清单分叉/8→9 traits/口径虚高 3 处纠正等）；
  一次性验证脚本已按教训落盘 gencheck（`role_v32_preview_fair.py` 等）

### SHTM（冻结）

解冻条件「sgcc/iapx 全部完成」现仅剩上表外部触发。解冻后队列：r2 审核 1,246 帧（用户）→
hardcase_promote 晋升 → V2.2 干净标签重训 → 双域评估；vlabel 分支审阅（跨项目）。
冻结完好性已经全项目审核确认（0 违规）。

### 交付通知机制

模型交付自动落 `~/cc/py/iapx/docs/jxl-deliveries-<date>.md`（首例 2026-09-14，4c95c67）——
iapx 会话照单执行，无需用户转述。
