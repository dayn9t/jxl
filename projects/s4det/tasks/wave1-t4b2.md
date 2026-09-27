# T4-b2 执行记录：其余类 400 帧抽样 + 四票池预标注（2026-09-27）

> 任务：T4-b2 评估集装配（tasks/T4-eval-gt.md §5 滚动计划；触发条件「批 2 种子集完成」已满足）。
> 执行会话：jxl wave1 并行 agent（本文件即执行记录正本——C-execution.md 并发冲突不入）。
> **执行中用户裁决插入（2026-09-27 下午，两次）**——分辨率门控 + prompt 排除项 +
> 执行重排（T4-b1 重标优先），本记录 §6 起为裁决后口径。

## 0. 触发与范围

- 批 2 种子集挖掘正本 1,268 帧 / 3,683 框已验收（C-execution 二次装配节）→ T4-b2 触发。
- 覆盖批 2 其余 15 类（任务书清单 16 类中 plastic_loose 存量 0 无评估源）；
  person 另线（全帧标注）不属本任务；ebike/garbage_bag 已由 T4-b1 覆盖。

## 1. 矛盾上报（执行前核查，决策质疑原则）

**任务书指定抽样源 = 批 2 种子集本体（p3-seedset/batch2/mining/dataset/all），
与 T4-eval-gt §2「种子集以外抽样」协议冲突**——该协议的目的是防 eval 帧入训练
（n001 0.895 虚高教训）。核查证据链：

1. 替代源不存在：preannot/manifest_leaf.jsonl（批 1 弱标签投影）leaf 仅
   ebike/garbage_bag/空；批 2 挖掘链 merge_input/*.jsonl 已被消费清空
   （除批 1 类残留 179 行）。**种子集是批 2 类唯一有框可定位源**。
2. 任务书给了与 dataset/all 框数完全吻合的逐类统计（car 965 框/shared_bike 648
   框/…/stall_cart 47 框/plastic_loose 0），说明路径选择是主会话核查过的知情决策。
3. T4-b2 触发条件本身 = 「批 2 种子集完成后」——评估源与种子集相关是计划内语义。

**执行裁决：按任务书从种子集抽，泄漏风险以机制显式落地**——
t4-b2 manifest 即训练排除清单，README 置顶警示「终版合并切分（vlabel-jxl-split）
时必须按 manifest 的 event 清单排除」；本记录再次上报，供用户复核否决。
（训练侧尚未切分，一切可逆。）

## 2. 工具改动（扩类复制 T4-b1 模式）

- `bin/t4_sample.py`：+`--batch`（默认 t4-b1 行为不变）；新增 `sample_b2`/
  `assemble_b2`——源 = dataset/all YOLO 标签（逐帧逐类最大框对角线定尺寸段），
  源族 = 挖掘池（seedset manifest pool 字段；比批 1 的 ET 族对跨五池的批 2 帧
  更有分层信息量），(日×机×3h)≤2 去密 + 低产类豁免补齐（如实计数）；
  labels.txt（正类+`?` 候选两形式，对齐 window1 修复后实践）；t4-b1 帧互斥排除
  （首跑实测事件级巧合重合 3 帧 → 源级排除 + 硬断言）；幂等含孤儿 symlink 清理。
- `bin/t4_preannot.py`：+`--batch`（run/score 同步）；`TARGETS_B2` 从
  `targets/<类>.toml` 的 `yolo_text` 动态读取——**与批 2 仲裁 c_arbitrate 同款
  描述，单一数据源**（任务书「中文目标描述」与「风格与既有两类一致」冲突：
  既有两类 T4-b1 TARGETS 为英文短语、池 prompt 模板英文、toml yolo_text 为批 2
  实战描述 → 取英文短语；中文形态要点在 toml vlm_prompt 已存，见 §5 对照表）；
  `render_labels`/`_load_gt`/`score` 全参数化（目录/类集按批次）。
  **〔§6 修订〕** yolo_text 短语方案后又被用户裁决 2（排除项）取代——终态 =
  toml vlm_prompt（含排除行）转池契约，见 §6.1。
- ruff check + format 全绿；导入冒烟过（QUOTA_B2 15 类合计 400；B2 目标表加载）。

### diff 摘要

- t4_sample.py：+254/-65 行（b1 逻辑原样函数化为 `sample_b1`，零行为变更；
  新增 b2 全链）
- t4_preannot.py：+203/-65 行（batch 参数化 + toml 目标表；b1 默认路径不变）

## 3. 抽样配额（400 (帧,类) 对）

存量 = 种子集逐类**有框帧数**（任务书数字为框数口径，本表用帧数口径，对照见下）：

| 类 | 选中/存量(帧) | 说明 |
|---|---|---|
| car | 44/475 | 均摊 43 +1（补 garbage_pile 互斥缺口） |
| shared_bike | 43/475 | 均摊 |
| bicycle | 43/303 | 均摊 |
| tarp_covered | 42/184 | 均摊 |
| tricycle | 42/75 | 均摊 |
| carton | 41/64 | 均摊 |
| stall_cart | 47/47 | 全取 |
| garbage_pile | 39/39 | 名义 40，t4-b1 互斥排除占 1 → 实取 39（不硬凑） |
| construction_fence | 29/29 | 全取 |
| piled_goods | 14/14 | 全取 |
| shop_display | 5/5 | 全取 |
| leaves_pile | 3/3 | 全取 |
| vendor_booth | 3/3 | 全取 |
| scaffolding | 3/3 | 全取 |
| foam_box | 2/2 | 全取 |
| plastic_loose | —/0 | **本批无评估源**（种子集 0 框，如实记录） |

（存量帧数与任务书框数口径换算：car 965 框=475 帧、shared_bike 648=475、
bicycle 447=303、tarp_covered 211=184、carton 146=64、tricycle 77=75、
stall_cart 47=47——两口径均已核吻合；互斥排除后存量较全量少 3 帧分布。）

- 总量 400 行 = (帧,类) 对；唯一帧 359（同帧多类 41 行——manifest 语义与
  t4-b1 一致：一行 = 一个被抽样叶类；批 1 帧单类属源特性巧合）。
- 尺寸段 large 382 / medium 18 / small 0（§6 small 段豁免——弱标签源结构性偏大）。
- 源族（池）：vehicle 172 / garbage 127 / vendor 101（无 phenomenon——18 类链
  无 phenomenon 类，符合预期）。
- 去密：23 帧豁免补齐（低产类全取超出组限），日志如实记录。
- 负样本 0：eval 专用负样本池另 agent 在建，后补（任务书许可）。

## 4. 预标注（四票池）

- 命令：`python t4_preannot.py run --batch t4-b2`（默认池 qwen-flash/doubao-vl/
  qwen38-local/glm-flash，concurrency 6；断点重入；已有 json 跳过=尊重人编辑）。
- 冒烟 2 帧：四票全出、consensus trusted 6 + low_agreement 3、labels 渲染正常。
- 全量 398 票任务（400 对 - 冒烟 2）后台跑，日志 /tmp/t4b2_preannot.log。

## 5. 目标描述对照表（yolo_text 中间方案留档；终态 = vlm_prompt 转池契约，见 §6.1）

| leaf | yolo_text（池消费） | toml vlm_prompt 中文形态要点 |
|---|---|---|
| car | car | 汽车（含面包车/货车/小货车，均为 car 内形态不细分） |
| shared_bike | shared bike | 共享单车（统一涂装+车筐） |
| bicycle | bicycle | 私有自行车（细轮，非统一涂装） |
| tricycle | tricycle | 三轮车（载货斗/封闭棚） |
| carton | cardboard boxes | 纸箱/纸板 |
| foam_box | foam box | 白色泡沫箱 |
| garbage_pile | garbage pile | 混合垃圾堆（aggregate 整体一框） |
| leaves_pile | pile of leaves | 落叶/枯枝堆 |
| tarp_covered | tarp covered object | 篷布覆盖物（hardcase：车/堆歧义待人工核实） |
| construction_fence | construction fence hoarding | 施工围挡（黄/蓝白铁皮） |
| piled_goods | piled goods | 整齐堆物（区分于垃圾） |
| scaffolding | scaffolding | 脚手架（+防护网） |
| shop_display | shop display goods outside store | 店外经营陈列（货架/商品/外摆） |
| stall_cart | street vendor stall cart | 流动摊（遮阳伞摊/三轮售卖车/地摊布） |
| vendor_booth | vendor booth stall | 棚/货架固定摊位 |

## 6. 用户裁决插入与全面重制（2026-09-27 下午）

**裁决 1（分辨率门控）**：用户在 T4-b1 编辑中发现 VLM 把 ~10px 小目标幻觉识别成垃圾袋
→ 检测层专用低分辨率图，**上限 960×540**（原图永不动——下游 crop/VLM 细判用高分辨率）；
样本=检测专用缩图实体文件（非 symlink）。主会话已物理重制 t4-b1（225 帧 960×540 LANCZOS
q88、labels 清空、旧票归档 `preannot_votes_orig-res-20260927.jsonl`、用户已编辑帧备份
`labels-edited-backup/`）。vlm_pool c9277f6：`image_data_url(downscale=N)` +
parse_boxes 同帧重复框去重（qwen38 曾 38 连发退化框）。

**裁决 2（prompt 排除项）**：每类目标描述必带「排除: …」行（garbage_bag 背包事故，
用户原话「保证下次不要犯」）。

**执行重排**：T4-b1 重标最优先（用户停标注在等）→ T4-b2 全切新口径。

### 6.1 工具二次改造

- `t4_preannot.py`：① run 加 `--downscale`（默认 1——语义=样本图未达标再缩，
  960×540 达标即 1）；② TARGETS 重构——废弃内联英文短语与 yolo_text 短语两版，
  统一从 `targets/<类>.toml` 的 **vlm_prompt**（检测语义+排除行）转换池契约
  （剥 det_mine 的 [0,1]/label 契约行，换 0-1000+persons 键，`pool_prompt()`）——
  单一数据源真正落 toml；b1/b2 同路径。
- `t4_sample.py`：assemble_b2 的 images 改 `write_det_frame()`——宽 >960 等比
  LANCZOS 缩 + q88 实体落盘，达标字节拷贝；幂等含旧 symlink 清除。
- **15 个 b2 类 toml 的 vlm_prompt 各加一行「- 排除: …」**（锚「- 若无」前插入，
  TOML 重解析验证）——主会话修 garbage_bag.toml 的同型操作。排除要点（lexicon/
  taxonomy 易混边界）：carton 排除装货/搬运使用中的纸箱与待售商品包装；
  tarp_covered 排除遮阳棚/雨棚/车衣覆盖在用车辆/施工安全网；garbage_pile 排除
  整齐货物堆与可辨单体（各自类）；shared_bike/bicycle 互排涂装特征；
  stall_cart/vendor_booth 互排流动/固定；等（全文见 targets/*.toml）。

### 6.2 t4-b2 重制

- 旧口径产物作废归档：`preannot_votes.jsonl`（296 行原图口径票）→
  `preannot_votes_orig-res-20260927.jsonl`；labels 清空（纯机器渲染，票档在可重建）。
- `t4_sample.py --batch t4-b2` 重跑：images 重制为 **369 实体缩图**
  （365×960×540 + 2×960×552——源有 1920×1104 少量，宽达标等比）。
- **抽样确定性悬案（如实记录）**：16:29 跑（豁免 23/359 唯一帧）与 17:05 跑
  （豁免 19/367）差异部分源于配额 car 43→44 的代码改动（可解释）；但 17:05 与
  17:05 后同代码重跑曾现一次 manifest 漂移（367→369 唯一帧，原因未定位——
  cand 构建/抽样段隔离双跑均稳定，连续两次终态跑逐字节一致）。终态 manifest
  sha256 已录 README；若未来重跑再漂移，优先排查 t4-b1/manifest.json 读取时点
  与并发会话写盘。

### 6.3 t4-b1 重标（最优先主线）

- 命令：`t4_preannot.py run --batch t4-b1`（图已 960×540 → downscale 1；
  garbage_bag prompt 带排除行）。
- **遗留（交主会话裁决）**：labels-edited-backup/ 内用户已编辑 4 帧为 1920×1080
  域坐标（旧三票池 qwen35b-local 渲染底稿 + 09-26 换代后票档未重渲染——两版
  票档均无法完整重现 backup 内容，4 帧甄别未完成）；若需续用须坐标 ÷2 转换
  到 960×540 域。本轮重标未还原（主会话指令未要求）。

## 7. 验收（2026-09-27 18:10 终态）

**t4-b1 重标**（/mnt/data/s4det/eval-gt/t4-b1/）：
- labels/*.json = **200**（= 正样本帧数），全部 960×540 域，X-AnyLabeling v4.0.6
  rectangle 4 角点绝对像素、description 带来源票；越域点仅 14/1131 框（glm 越界
  票经共识上屏，既有已知行为，人审可调）。
- preannot_votes.jsonl 200 行：四票署名 195 + 3 票 5（3×qwen 429 / glm ReadTimeout
  与 500 各 1——弃权票语义）；consensus trusted 538 / low_agreement 844 / arbited 2。
- 门控效果：garbage_bag 每帧均框 2.85（旧口径 3.59，-20%）。
- 已即时报告主会话（用户续标解锁）。

**t4-b2 全链**（/mnt/data/s4det/eval-gt/t4-b2/）：
- images **369 实体**（365×960×540 + 2×960×552）＝ manifest 唯一帧数；0 孤儿 0 断链。
- manifest **400 (帧,类) 对**＝逐类配额精确命中（§3 表）；sha256 录 README。
- preannot_votes.jsonl **400 行**，(basename, cls) 集与 manifest 精确对齐（脚本核验
  True）；四票署名 393 行 + 3 票 7 行（qwen 429 偶发——AIMD 协议正常震荡）。
- labels/*.json **369**（= 正样本唯一帧；空卷帧 0——全部帧有上屏框）。
- 抽查 3 json：v4.0.6 / 960×540 / rectangle 4 角点绝对像素 / description 带来源票
  （「共识: doubao-vl+glm-flash+qwen-flash…」「候选: doubao-vl」）全合规。
- consensus 全量：trusted 535 / arbited 19 / low_agreement 909（候选多——15 类
  语义比批 1 两类难，人审裁决量预期大，符合 hardcase 分布）。
- labels.txt 30 行（15 正类 + 15 `?` 候选）覆盖实际标注的全部 15 类（核验 True）。
- 隔离台账：**本任务零弃置样本**（全取类无弃置；均摊类落选帧仍在种子集服务
  训练，非门拒绝样本；预标注弱单票/空票留 votes 台账不上屏——台账本身即记录）。

## 8. 遗留问题

1. **训练排除机制（P0，随本任务产生）**：t4-b2 帧源 = 批 2 种子集本体——终版
   合并切分（vlabel-jxl-split）必须按 t4-b2/manifest.jsonl 的 event 清单排除，
   否则 eval 入训练（README 已置顶警示；切分工具侧未接线——需在切分任务落地）。
2. **t4-b1 用户已编辑 4 帧**（labels-edited-backup/）：1920×1080 域 + 旧三票池
   底稿，两版票档均无法完整重现（09-26 换代重跑票后 labels 未重渲染——旧 json
   为 09-25 池渲染）；续用需坐标 ÷2 + 逐帧甄别，交主会话/用户裁决。
3. **t4-b1 排除行版本差**：b1 本次重标用 garbage_bag.toml 排除行 v1（背包/搬运
   中麻袋）；主会话随后扩充 v2（+车罩/防尘罩/挡风被/篷布/苫布 + 垃圾桶）。
   b2 15 类不受影响（不含 garbage_bag）；b1 是否以 v2 重跑由主会话裁决
   （用户已在标注中，重跑会覆盖未保护帧——render_labels 只跳过已有 json，
   人审进行中重跑是安全的：已编辑帧有 json 会被跳过，仅空帧/未编辑帧重渲染）。
4. **vlm_pool 两处待主会话收口**：① glm spec 的 max_tokens=768 override 只接了
   verdict 路径（L141），call_person（L263）写死 1500 未读 spec——本次 grounding
   实际用 1500（比 768 宽松，偶发 length 截断已告警，不阻塞）；② 我在
   call_person 加了 content=None → 弃权票防御（2026-09-27 t4-b2 实炸 TypeError，
   修复一行，与池「失败不当空票」原则一致）。
5. **抽样确定性悬案**（§6.2）：终态连续两跑逐字节一致、cand/抽样段隔离双跑稳定；
   一次未解释漂移已记录。根因候选：t4-b1 manifest 读取时点与并发会话写盘。
6. 负样本 0：eval 专用负池另 agent 在建，建成后按 T4 规范 §3 补 10-15% 配额。
7. plastic_loose 本批无评估源（种子集 0 框）——EVAL-BENCHMARKS 盘点应记缺口。


