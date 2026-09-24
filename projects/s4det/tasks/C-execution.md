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

1. 人工兜底 119 帧（低共识 contact sheet）——用户/s4 裁决后回填
2. doubao 池成员恢复（控制台开通替代模型）
3. agree=false 对象级类别投票工具（vlm 分类票，批 2 建）
4. 垃圾族弱（303 框）：批 2 扩池或专项权重后重挖
