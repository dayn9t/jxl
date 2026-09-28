# 任务书 C：P3 mobile 种子集（s4 会话 → jxl 会话，2026-09-25 成文，待冻结通知开工）

> 前置：taxonomy.md v1 **用户批准冻结**（未批不得开工本任务书）+ annotation-guideline.md v1。
> 执行：jxl 会话。执行完回写 `C-execution.md` + README 里程碑打点。

## 目标

产出 mobile 首批 17 叶类（taxonomy v1 mobile 块）的共识标注种子训练集，经防毒标注双闸审计。

> **执行校正（2026-09-26）**：批 2 实际按冻结附录 A 口径执行——16 个新类 profile + ebike/garbage_bag 复壮 = 18 链，见 C-execution 批 2 节；本文 17/15 为成文时 v1-草案口径，保留作历史。

## 输入（全部就绪）

- `/mnt/data/s4det/preannot/manifest.jsonl`：56,999 object 行（agree=true 6,211 直通候选；conf hi 20,277/mid 15,832/lo 20,890；ambig_g12 23,629 单独标记）
- 机制件：`targets/*.toml` profiles + `b_multirun.sh`（B② 已冒烟）+ guideline 例图指针
- 数据根三区（B④）

## 步骤

1. **叶类投影步**（冻结后第一动作）：preannot 行按 taxonomy v1 映射（EventType 码族 + llm 类别串族双锚点）投影出 `leaf` 字段，另存 `preannot/manifest_leaf.jsonl`（不覆写 B① 产物）。agree=true ∧ conf≠lo ∧ 非 ambig_g12 → 种子直通候选池。
2. **共识标注**（b_multirun 类独立链）：第一批 ebike + garbage_bag（机制已冒烟）全量跑通 → 第二批扩其余 15 类（执行实为 16 新类，见上注记）。候选帧池 = 直通候选 + 各类 conf hi 抽样。
3. **VLM 仲裁**：agree=false 与 review 分歧行按优先级分批（先 conf hi 非 ambig → conf mid → ambig_g12 抽验）；vlm_pool 池成员先过 T7 刻度重标定（s4det 域 GT 用第一批 L0 样本组建）。~51k 行分批，成本/进度每批回写。
4. **人工兜底**：review_pack → review_workbench；覆盖全部 hardcase 类（tarp_covered/charging_bike 形态求证、pile 边界、层叠 ≥25% 判定）。
5. **双闸审计**（skill j-consensus-labeling ⑩）：label_audit gate（k≥2∧strong≥1）+ cluster 位置卫兵；毒簇清洗删框留图。
6. **合并切分**：consensus_dataset（多类 --classes）→ jxl_split 8:1:1（symlink）。

## Gate（s4 会话验收）

- 双闸审计报告通过（闸门拦截率+嫌疑簇处置记录）
- 每入训叶类样本 ≥50（不足者按 taxonomy 硬条件降观察类并记录）
- 负样本保留（规范 §10）、隔离台账全程记账
- data.yaml names 与 profiles output_cls_id 逐类一致

## 纪律

GPU 共卡查余量；VLM key 只走 cfg/env；中间产物全落盘；叶类投影发现的映射歧义（一 object 多叶类候选）不硬投影——入隔离台账转 s4 会话裁决语义。
