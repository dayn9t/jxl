# T4 规范：人工裁决级 eval 参照集（s4 会话成文 2026-09-25，随批滚动建设）

> 依据：design §6 + docs/EVAL-BENCHMARKS.md WARNING-1（n001 test 虚高 0.895 教训）。
> 执行：jxl 会话（抽样/装配），人审由用户。**评测专用，禁止任何形式入训练**（含经由 isolation 失效的间接泄漏——目录物理隔离 + README 警示）。

## 1. 分层框架（strata）

维度 = 叶类 × 尺寸段 × 源族：

- **尺寸段**（对角线像素，crop 外接框换算原图）：small <32×32 / medium 32-96 / large >96——对应 COCO 惯例，推理侧尺寸过滤阈值将来按此段出报告。
- **源族**：mobile 域内按事件粗类（ET2 族 / ET3 族 / 其他），防单一来源偏。
- 分配：目标总量 **600 帧**（首批先做 batch1 两类 200 帧：ebike 120 + garbage_bag 80）；叶类配额按 taxonomy 入训类均摊，观察类不入。

## 2. 抽样协议

1. 从**种子集以外**的事件池抽（seedset split 的 train/val/test 全部排除——按 event_id 排查；用 preannot manifest 未入池行）。
2. 分层比例抽样 + 层内随机；同 (日×机×3h) ≤2（去过密）。
3. 每帧记录：strata 标签 + 源 event 路径（可回溯）。

## 3. 标注协议（人工，唯一真值层）

- 空标帧保留为负样本（占 10-15% 配额，防检测器全图正偏）。
- 框规范 = annotation-guideline.md v1（可见框/堆叠混合式/区域类边界）；有分歧的帧由用户终裁，分歧记录留档（为规范 v2 提供素材）。
- 工具：vlabel review_workbench；双人独立标 10% 子集算一致性（IoU≥0.7 ∧ 类一致为 agree），一致率 <85% → 规范歧义信号，回 s4 修订规范。

## 4. 产物与用法

- `{data_root}/eval-gt/{batch}/`：images symlink + labels + manifest（strata 元数据）+ README（禁止入训练警示）。
- 用法：检测器评估唯一锚（叶类 mAP + 尺寸段分项）；每次重训后跑同一集；数字入 EVAL-BENCHMARKS §2 时移除 WARNING-1 前缀（仅对该集口径）。

## 5. 滚动计划

| 批 | 覆盖 | 触发 |
|---|---|---|
| T4-b1 | ebike + garbage_bag（200 帧） | 用户 20-30 分钟人审窗口（可与 119 帧兜底同场做） |
| T4-b2 | 其余 15 类（400 帧） | 批 2 种子集完成后 |
