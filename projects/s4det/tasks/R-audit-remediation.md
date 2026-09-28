# R 任务书：系统审计修复——jxl 侧剩余项交接（s4 会话成文 2026-09-28）

> 背景：2026-09-26 s4 会话对 s4det 全项目系统审计（5 维度 + 逐维对抗复核，57 条确认，0 误报）；09-27 对账时 jxl 侧已自行修复一批（`28824c6` 文档批量 / `3f713d8` 人工池台账 / `29b4116` c_humanpack / `cc241c8` c_vote 池 / `33ae655` docstring / T4-b2 装配 / `c734ecd` 能力账 / `771ea86` c_recycle）。
> 本文只列**仍开放**的 jxl 侧项（代码+数据+执行记录）；文档口径修复由 s4 会话另行提交。行号为 09-27 审计时点，执行时按内容定位。

## 0. 用户裁定需落地（数据侧）

- **window1 分层标注（2026-09-27 用户裁定）**：不重打包，manifest 加 `layer` 字段——**必编 177**（仲裁残量 58 + 批 1 低共识 119）/ **复检 345**（已 VLM 仲裁入终版帧，编辑时快速过目纠明显错）。119 低共识子集映射自批 1 仲裁人工池清单。文档侧已落 TRAINING-LOOP §6-Q1 与 README（s4 会话 `5b3caef`）。

## 1. P0（阻塞正确性）

1. **c_seedset.py L152-155**：`split("_")[2]` 取到 `r{rid}` 而非 event_id（池文件名实为 `{date}_{eid}_r{rid}`，leaf 在目录名），batch1 排除失效 → **batch1∩batch2 仍 142 事件重复**（09-27 重算确认）。修法：改 `split("_")[1]` + picked 装配前 fail-fast 断言 + 142 重复盘存处置（剔除重装配；或用户显式豁免落档——s4 侧意见：剔除重装配，质量优先不计重装配成本）。
2. **t4_sample.py 抽样端排除解析核验**：L115-117 构造 `{date}_{eid}` 但 L123/L177 用纯 eid 比对，疑似仍永不匹配（`exclusion-events.jsonl` 合并侧排除已救 b1/b2 存量，但 t4-b3/批 3 抽样会再漏）。若确未修：构造改取 `split("_")[1]` + 抽样落盘前断言。
3. **t4_preannot.py score() L214 只遍历 `gt.items()`**：负样本帧/无 GT 帧上假阳性漏计 → P 虚高。`c734ecd` 能力账已接线，score 口径失真会直接进能力账——**先修再跑对拍**。修法：遍历 votes_idx 全部键、`gt.get((stem,cls), [])` 取框（GT 空=全部预测计 FP）。

## 2. P1

4. **c_assemble.py 空标签负样本帧整帧丢失**（L90 `images.setdefault` 旁加 `frames.setdefault(lf.stem, [])` + 负帧计数入日志；磁盘佐证：assemble 相对非 review 空 txt 缺 79 个仅有空标签的 stem）。
5. **c_vote.py abstain/img_error 入 sidecar 后被断点重入永久跳过**（done 集合按 `status != "abstain"` 过滤 + 消费侧 last-wins 语义注记）。
6. **c_humanpack.py `_load_jsonl` 源缺失静默返回空**（去存在性兜底自然抛 FileNotFoundError，或 run 入口必需文件逐个 BadParameter）。
7. **t4-b1 labels.txt 补 `?` 变体**（应 4 行：ebike/garbage_bag/ebike?/garbage_bag?）——⚠ 用户编辑中，重生成时机避开编辑会话。

## 3. P2

8. quarantine 579 条 confirmed-discard 行 `recoverable=true` 人机矛盾（一次性原子回写 false + 模型 validator 强制）。
9. c_seedset semantic_dedup embedding 缓存无 files 指纹旁车（候选集变化后静默错位）。
10. a_manifest llm_ok 死标志（`_parse_llm` 的 bool 返回值被丢弃，统计恒 0）。
11. CHAIN_SRC 双份上提 a_common 单源；MINING/B1/B2 硬编码路径改 `DATA_ROOT` 派生。
12. 溯源机制：c_vote/t4_preannot/c_arbitrate 首写台账落 run_config（git HEAD + 脚本 sha256）。
13. 4 个破百行函数拆分（c_project.main / t4_sample.main / a_manifest.build_family / c_seedset.main，下次触碰顺手拆）。
14. C-execution 口径两处（:70 人工窗口 119→464 补注；:115 「GT 677」加「链帧（唯一 514，129 帧被多链重复仲裁）」限定）。
15. 数据根 `/mnt/data/s4det/README.md` 更新（结构表缺 batch2/t4-b1/t4-b2/window1/weights；台账行 579→819 现存量）。
16. jxl 仓 TODO.md(sgcc) 未提交 + main 领先 origin ~100 未 push，择机收口。

## 4. 已完成侧（对照，勿重复）

- s4 仓：CODE_MAP/TODO 指针刷新（`90b83d2`）、reveng_platform_codes 损坏计数（`6d6bb47`）。
- s4 会话文档修复 pass 2（taxonomy/TRAINING-LOOP/design/annotation-guideline/research×3/盘点/A-cluster-batch/C-seedset/README/T4-eval-gt/EVAL-BENCHMARKS/CODEBOOK）：随本任务书同批提交。
- 审计证据存档：s4 会话 tool-results/bcvfht57b.txt（57 条摘要）。

## 5. 执行记录

- [ ] jxl 侧按 P0→P2 顺序执行，完成逐项勾选并在本节记 commit hash。
