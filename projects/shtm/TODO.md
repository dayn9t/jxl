# SHTM 待办存档（2026-09-12 插入新任务前快照；2026-09-13 增补 sgcc v4）

> 恢复上下文锚点：读 `projects/shtm/README.md` 状态表 +
> `projects/shtm/research/2026-09-11-S1难例自动削减.md`（削减全案）。
> **插入任务 iapx 三件套已完成**：见 `projects/sgcc/research/2026-09-12-iapx三件套执行报告.md`
> （A 重训完成待协同切换 / B v1 可行性+词典 / C 归因闭环；**person v4 重训断点续跑手册见其 §6**）。
> 本文件是挂起任务唯一清单，插队任务完成后按此续推。

## 等用户动作（4 件）

1. **r2 残留人工审核**：`vlabel-gui /home/jiang/ws/trash/s1_hardcase_review_r2`
   （1,246 帧；晋升后另有 ~78 帧否决恢复帧并入）。对照
   `projects/shtm/s1/hardcase_residual_list.jsonl`（已对齐重编号）。
   审完拷回 `s1_relabel_v1` → S3 导出训练
2. **削减晋升批准**：抽查 `research/2026-09-10-SHTM自动删框拼图-{1,2}.jpg`
   （可选，92 条已预否决）→ 说「晋升」即跑
   `uv run --project . python projects/shtm/s1/hardcase_promote.py`
   （隔离池 527 帧入正式集；92 否决自动恢复并退回 r2；幂等有 --force）
3. **类别草案终审**：`projects/shtm/类别严格定义-draft.md` 3 开放问题
   （O1-O3/L1-L3/D1-D3）
4. **vlabel 分支审阅合并**：`/home/jiang/cc/next/vlabel` feature/convert-pipeline
   8 commits + BRANCH_REVIEW.md（跨项目，非 SHTM）

## 我接续的下游（等上述解锁）

- 晋升跑完后：核对 r2 帧数、把否决帧并入 r2、汇报
- r2 审完：拷回 `s1_relabel_v1`（r2 vlabels 覆盖对应帧）→ S1 终版导出 → V2.2
  from-scratch 干净标签重训（sgcc0，`~/ws/trash` 对等路径）→ 双域评估 → 部署决策
  （链路见 `research/2026-09-09-检测器改进流程方案.md`）

## 挂起改进（下轮重标前）

- **prompt v4**：误检例删「杂物」，显式写明「散装垃圾堆是目标(dump)」——92 条误删根因
- backlog 154 条 AUTO_REJECT 同法文本挖掘复审（低风险：从未入 GT）
- can→direction 属性缺口（meta 权威 schema 有、`s1_vlabel_merge.ATTRS_BY_CLASS` 未实现）
- （可选）jxl 仓库 push 远端备份（领先 origin 100+ commits）
