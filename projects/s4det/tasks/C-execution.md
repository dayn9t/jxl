# 任务书 C 执行记录（jxl 会话，2026-09-25 起）

> 对应 [C-seedset.md](./C-seedset.md)。本文件按批回写；第一批 = ebike + garbage_bag（s4 会话 2026-09-25 开工通知，taxonomy v1-暂定冻结生效）。

## 第一批：ebike + garbage_bag（2026-09-25）

### 步骤 1 叶类投影 ✅

- `preannot/manifest_leaf.jsonl`：投影 **49,939/56,999 行**（ebike 37,515 + garbage_bag 12,424），**双族锚歧义 0**
- 锚点：ebike = ET2 ∨ llm「非机动车乱停」；garbage_bag = ET3（taxonomy v1-暂定冻结，llm 优先）
- 直通候选（agree ∧ conf≠lo ∧ 非 ambig_g12）：帧池各 800（超限按种子随机取子集）
- 坑：帧池 symlink 初版用原始 basename（事件图恒名 `{rid}.jpg`）互相覆盖只剩 2-3 帧——改 `{date}_{event_id}_r{rid}.jpg` 唯一扁平名（n001 同款教训，已写死在 c_project.py）

### 步骤 2 共识挖掘 ✅（b_multirun 类独立链，validators=yoloe,gdino，替身在役模型 yolov8m.pt）

- ebike：L0 21 / L1 546 / review 233（800 帧）
- garbage_bag：L0 30 / L1 539 / review 231（800 帧）

### 步骤 3 VLM 仲裁 ✅（vlm_consensus_gt，池=qwen-flash+qwen35b-local+glm-flash）

- **T7 刻度重标定**（s4 域 GT=批 1 L0，21 帧/23 框）：qwen-flash **F1 0.875** / qwen35b-local 0.76（强票双达标）/ glm-flash 0.33（仲裁票）；刻度 0-1000 实测确认（`t7_calib_report.json`）
- review 仲裁：ebike 194/233 帧消解 → 688 框（trusted 580 + arbited 108），人工池 39 帧；garbage_bag 151/231 → 197 框（trusted 167 + arbited 30），人工池 80 帧。**人工负载降 ~83%**
- agree=false 行（ebike 4 / garbage_bag 666）对象级类别投票：**未做**（无现成工具，批 2 前补 vlm 类别投票脚本；不阻塞种子集——种子框来自共识挖掘而非 preannot 行）
- ⚠️ **外部阻塞**：doubao 端点被方舟关闭（`doubao-seed-1-6-vision-250815` ClosedEndpoint；替代模型「未开通/不存在」，需控制台开通）。池以三成员运行（强票 2/3 仍可达）

### 步骤 4 人工兜底 ⏸ 等人工（材料已备）

- 低共识帧 contact sheet：`/mnt/data/s4det/p3-seedset/batch1/mining/review_human/`（ebike 39 帧 1 页 + garbage_bag 80 帧 2 页，stem 可回溯池帧）
- 求证类入池：tarp_covered（覆盖车 vs 覆盖堆）随本批 hardcase 一并人工裁决

### 步骤 5 双闸审计 ✅（初查 + 终查）

- 闸门（L0 k≥2∧strong≥1）：挡 1/22 框（ebike，已入隔离台账 `gate:` 前缀，priority=high）
- 位置卫兵：终版 1,404 框 **0 嫌疑簇**（移动目标+车载巡查，位置签名分散；n001 静态物风险在本域天然低）

### 步骤 6 合并切分 ✅

- 六层互斥合并（L0 dump / L1 / arbitration trusted+arbited）：**1,481 帧 / 1,404 框**，近重复 1 框清洗
- 坑两枚（已固化进 `c_val_to_yolo.py` / `c_arb_to_yolo.py`）：det_mine labels/ 混含 review 帧（与仲裁层 stem 冲突）；validators 的 target 字段是原始检测不是自动标注（L1 须用 labels/ 或按层精确导出）
- 切分：train/val/test 有框帧 627/71/77（≈8:1:1，symlink）
- **逐类 gate**：ebike 1,100 框 ✓ / garbage_bag 303 框 ✓（≥50 达标；garbage_bag 偏弱因替身在役模型无垃圾类，专项权重后复壮）
- 产物：`/mnt/data/s4det/p3-seedset/batch1/mining/dataset/`（data.yaml names 0:ebike 1:garbage_bag 与 profiles output_cls_id 一致）

## 批 1 遗留（批 2 前处理）

1. 人工兜底 119 帧（低共识 contact sheet）——**已裁定（2026-09-25 晚）：并入批 2 人工窗口**
2. doubao 池成员恢复（控制台开通替代模型）
3. ~~agree=false 对象级类别投票工具~~ ✅ 已建（`c_vote.py`，见「项目恢复」节）
4. 垃圾族弱（303 框）：批 2 扩池或专项权重后重挖

## D0 v0-mini 训练（2026-09-25，干线验证 + 专项权重）✅

- 训练：yolo26n 官方 COCO 底座，100 epochs，batch auto（`p3-seedset/batch1/train/v0mini/`，exit 0）
- val（弱标签共识切分，**WARNING-1 未解除不得当质量结论**）：all mAP50 0.438 / mAP50-95 0.293；ebike 0.496/0.313；garbage_bag 0.380/0.273
- 规模适配检查（skill ⑨ 双向）：末段 val 仍小幅上升（epoch91 0.269 → best 0.293）+ train/val 无过拟合分叉 → nano 容量当前足够，瓶颈在数据量而非参数规模（无需升级；待数据扩批后复判）
- 产物换装：`/mnt/data/s4det/weights/s4det-v0mini.pt` → targets/{ebike,garbage_bag}.toml weights ✓（garbage_bag 复壮重挖留批 2）
- 指标已入 `docs/EVAL-BENCHMARKS.md` §2（❌ 级 + WARNING-1 标注）

## 批 2 种子集装配（2026-09-25，项目暂停前完成）

- 工具：`c_seedset.py`（锚组配额 + 双重去重）。产物：`/mnt/data/s4det/p3-seedset/batch2/seedset/`（symlink 池 + manifest + stats）
- 结果：**新增 3,316 帧**（vehicle 1,000 / garbage 1,000 / vendor 666[数据量截断] / phenomenon 400 / 负样本 250）+ 批 1 已入 1,600 = **全局 ~4,916 ≈ 5,000 预算**（用户定的规模）
- ⚠️ **当日晚校正**：原跑在 manifest 写盘前崩溃（`build_candidates` 帧字典缺 `pool` 键 → KeyError），分池数字来自崩溃前逐池日志、**产物从未落盘**（仅 1 symlink + 4 个去重 npy）——批 2 开工预检时发现，修复后重跑（seed=42 确定性复现，数字与原记录吻合），**磁盘验收通过**（manifest 3,316 行 = images 3,316 文件 = stats 分池，0 断链）。教训：**日志数字≠落盘产物，装配类任务必须清点文件才算完成**；commit 只含工具时尤其要区分「结果已记录」与「结果已落盘」
- 设计要点：按**锚组（粗类族）**配帧而非细类——弱标签粒度不可细分 ET2 子类，细类归属由批 2 挖掘类专用校验器完成；person 无锚（COCO 预训练 + 全帧标注覆盖）
- 关键教训：**mobile 巡查编组制**——ET1/2 全部 14,049 事件仅落 143 个（日×机×3h）组，组 cap 必须放宽（=2 时全局容量仅 286 干涸）；组内是沿途不同目标，真重复由语义去重（DINO cos≥0.95）兜底
- 流程教训（用户指出）：新工具实施前设计须先过目确认（superpowers 流程），不能直接开跑——已固化为执行纪律

## 项目暂停状态（2026-09-25 用户指令）

值守循环已停、两会话队列清空待命。恢复触发（任一）：① 人工窗口（119 兜底 + T4-b1）② doubao 控制台开通 ③ taxonomy 终态批准 → 批 2 开工（种子集已就绪）④ EXEMPT-S 确认。

> → **已恢复（2026-09-25 晚）**：③④ 落地（taxonomy 冻结 + EXEMPT-S 确认），批 2 开工；② doubao 仍待控制台开通（池三成员运行）。

## 项目恢复与三裁定（2026-09-25 晚，用户 brainstorming 菜单裁决）

- **taxonomy v1 冻结**（批准定版）→ 批 2 解锁开工
- **EXEMPT-S 同意豁免**（EVAL-BENCHMARKS 转确认态；v1 质量口径=中大型目标）
- **Q1 = 最小先行**：T4-b1 225 帧先行开窗（解锁 v0.1 质量结论）；119 兜底并入批 2 人工窗口

## 批 2 挖掘开工（2026-09-26 凌晨）

- **16 个新类 profile**（cls_id 2–17，语义对齐冻结 taxonomy；8abed90）：vehicle 池→bicycle/car/shared_bike/tricycle；garbage 池→carton/foam_box/garbage_pile/leaves_pile/plastic_loose/tarp_covered；vendor 池→construction_fence/piled_goods/scaffolding/shop_display/stall_cart/vendor_booth；target 弱票=yolo26s.pt（COCO 底座，激进策略），ebike/garbage_bag 复壮链保持 v0mini
- **c_seedset 修复重跑**：原装配死于 `build_candidates` 缺 `pool` 键（KeyError，manifest 写盘前）——「4,916 帧就绪」曾只存在于日志；修复+seed=42 复现重跑，**磁盘验收通过**（manifest 3,316=images 3,316=stats，0 断链；15347b2）。教训：**日志数字≠落盘产物，装配任务必须清点文件才算完成**
- **b_multirun 顺序毒修复**（bbd1feb）：assoc 数组任意迭代序 → >2 类合并 --classes 与标签 id 错位风险；改 PAIRS 序 + det_mine --force（编排器拥有 out 子树，链重跑=重做）
- **挖掘链运行**：18 链 × 666–1,000 帧（`batch2/mining/<cls>/consensus`），后台运行中；person 走全帧标注另线不入类链；phenomenon 池不入本轮挖掘（Q2 VLM 后置）

## 模型池换代 + det_mine target 票重构（2026-09-26，用户新近度准则触发）

- **触发**：用户指出池成员版本古老——KB 0925 校正注（qwen3-vl 系过时）挂了没 acted on，违 0925「模型选型新近度准则」
- **四网关 /models 实测**：doubao-seed-2-1-pro-260915 **key 下已开通**（恢复触发②就此解除，无需控制台操作）；spark 本地=qwen3.8-flash-next 别名（与 3.5-35b 同权重根）；dashscope 3.8 代仅文本/omni 无 VL（VL 最新仍 3-vl-flash，出 VL 即换）；glm-5.3 当前代
- **换代**：doubao-vl → `doubao-seed-2-1-pro-260915`（思考模型，`thinking disabled` extra_payload——关思考 6s/图，开思考 >120s 超时；杂键 bbox_2nd 与 bbox_2d 并存，共识聚类无害）；qwen35b-local → **qwen38-local**（全代码引用同步改名）
- **重标定**（T7 GT；首跑 F1 全零系漏传 --target-text，修正后）：qwen-flash 0.808 / qwen38-local 0.741 / doubao-2.1 0.724（三强票）/ glm 0.482（仲裁）；doubao 2.1 刻度 0-1000 确认（`t7_calib_report_20260926.json`）；**池回四成员**
- **det_mine target 票重构**（合并近重复 fail-fast 挖出的真毒）：领域约定「L0 GT=target 框」下 ①无在役先验的 16 新类链挂 yolo26s 作 target——其全类检出会串类入 GT（stall_cart 的 COCO car 框=毒）→ `TargetProfile.weights` 改可选，无先验链无 target 票、L1 校验器共识框即标注；②v0mini 复壮链双头串扰（ebike 链 20 行 L0 塞进 garbage 头同物框 → merge fail-fast 拦下）→ 新字段 `target_cls` 类过滤（ebike=0 / garbage_bag=1）
- **18 链全部重跑**（首跑产物作废）；顺手清偿 vlm_consensus_gt / vlm_grounding_calibrate / vlm_role_vote 存量 lint
- **T4-b1 预标注以四票新池重跑**（旧三票票档归档 `preannot_votes_pre-upgrade-20260925.jsonl`，留作换代前后对拍）

## T4-b1 预标注模式（2026-09-26 用户准则「预标注优先」）

- **准则**：人工标注一律多模型预标 + 人编辑，禁止凭空画框；模型署名保留，编辑结果即真值 + 各模型 IoU/P/R/F1 对拍数据源。正本已入全局 CLAUDE.md（j-cc 509c78a）；T4-eval-gt §3 / TRAINING-LOOP §7 同步修订（原 SAM2「纯人工」铁律被用户裁决替代）
- **工具**：`bin/t4_preannot.py`——run=三模型逐帧逐类 grounding（vlm_pool，每帧只预标被抽样叶类；负样本帧不预标=空卷）→ consensus_boxes 分级（trusted/arbited=共识底稿；强模型单票=候选框带 `?`，glm 0.33 单票不上屏但留台账）→ X-AnyLabeling v4 JSON（schema 抄上游 assets/demo.json）；score=用户编辑后按 per-model+池共识对拍 P/R/F1
- **防误覆盖**：labels/ 为交互层，`preannot_votes.jsonl` 为机器侧单一数据源；已有 json 的帧跳过渲染（尊重人编辑）
- 已知局限（如实记录）：①正样本帧内偶见的跨类目标不预标（噪声源），靠用户手动补；②共同盲区目标（三模型全漏）预标注不可见，GT 会继承——互补对拍/纯人工抽检校准待 GT 建成后执行

### 批 2 前置：agree=false 类别投票（批 1 遗留 #3）✅

- 工具：`bin/c_vote.py`（bbox±10% crop → vlm_pool 菜单分类投票 → role_consensus；split 入隔离台账 priority=high，断点重入按行键）；`vlm_pool.call_role_vote/parse_verdict` 参数化 prompt/valid（role 默认行为不变）；菜单导出件 `projects/s4det/taxonomy_leaves.json`（taxonomy 附录 A 46 类，taxonomy 演进时重导出）
- 结果（872 行）：**auto 633（trusted 607 / arbited 26）/ split 239 入隔离台账 / abstain 0**；sidecar `preannot/votes_agree_false.jsonl`（不改写 manifest，训练框仍以共识挖掘为准）
- 分布要点：garbage_bag 196 / tarp_covered 135 / carton 100 / garbage_pile 26 / construction_fence 25 / foam_box 22 / water_puddle 22 / person 19 / piled_goods 15——苫布覆盖硬类大量路由浮现；批 2 挖掘装配按 verdict 回填类别先验
- 坑：parse_verdict 参数化时**签名加了 valid 参数但函数体仍比较旧常量 ROLE_VALID**（冒烟 6/6 全 abstain 暴露）→ 已修；教训=参数化改造必须 grep 函数体内全部旧引用点，冒烟必须真跑不能只看 exit 0
