# jxl 代办（todos.md）

> j-todo 单一归宿。状态：`[ ]` 待办 / `[x]` 完成 / `[~]` 进行中。

## 今天（2026-09-19）用户要参与的

- [ ] **隔离台账人工批量复核（290 条，高优先 121）——★坐姿正样本回收是 v7.1 前置**：`gencheck/quarantine/2026-09-19-sedimentation.jsonl`
  ——重点高优先类（模型分歧/不可判定）：upper 不一致 7、osnet 重连异人对 114（含不可判定）。
  每条三选一：回收（人工验证后入池）/确认丢弃/升级入参照集。看图入口参考 visual-adjudication skill。
- [ ] **ASR 评估集 5 个决策点**（方案 `research/2026-09-19-ASR评估集设计方案.md`）：
  抽样量 24/30/36、时长边界、A/B 判定门、**裁决人资质（你本人裁沪语 verbatim 是否可行）**、LLM 辅助边界。
- [ ] **role v3.3 切换前 50 张人工抽检**（清单 `gencheck/rolepool_20260919/rolepool_spotsample_checklist.md`，C 类 12 张优先）。
- [ ] **v6 切换执行跟进**：切换建议已正式发 iap（通知单 2026-09-19 补遗：v7 撤回、改推 v6）——等 iap 执行 `ln -sfn 2026-09-18_person_n_v6.*` 后确认。
- [ ] **v7.1 回炉**（依赖台账复核）：从隔离台账 100 帧 bad_fit 弃帧回收低头坐姿正样本 → 重训 v7.1 → 重跑预冒烟（重点坐姿）。
- [x] **sgcc3 升级后验证（零负载部分完成，烤机取消）**：升级完成（内核 7.0.0-1012-aws/驱动 **615.71.09 已载**——SB 问题已解）；零负载核验 PASS（venv torch 2.9.1+cu128 CUDA 实算/本次开机零 NVIDIA Xid）。**烤机监控按用户 2026-09-19 深夜裁决取消**，不再跟踪。
- [ ] cleaner 稀有类正样本人工裁（1 张，role_conf 0.97 窗前交互——稀有类样本矿）。

## iapx 解散对 jxl 的影响（2026-09-19 发现，spec `iap docs/superpowers/specs/2026-09-19-polyglot-restructure-design.md`）

- [x] **osnet vendor 移交（09-19 午后完成）**：jxl.vdt 侧移交已就位（`reid_osnet.py`+`vendored/osnet.py`）；gencheck 侧 `osnet_ft.py` VENDOR_CANDIDATES + 两个 export 脚本 VENDOR 已切 jxl.vdt 路径（原 iapx 路径随解散已删、实际已断）。验证：vendored 加载 v2.1 权重 strict 全匹配 + 前向 PASS（352 类）。
- [ ] **reid 复测通道将失效**：iapx pipeline+影子测试被删（Rust 已镜像+GT 门绿）——v2.1 正式复测须改走 iap Rust 重放（monitor_replay_corpus），reid-retest skill 的 temp 通道流程届时更新。
- [ ] **数据资产去向关注**：`/mnt/data/jiang/ws/iapx/n001/`（samples/eval.jsonl held-out 328/benchmark）——export-samples/export-reid-pairs 工具进 iap py/tools 接口不变，但存量数据迁移方案需确认。09-19 午后核实：**数据原地完好未迁**（annotations/benchmark/cache 全在），role 训练链（iapx_role_train_v33）与 osnet eval 数据依赖暂无恙。
- [x]（无需动）spec 明确「jxl 除 osnet 外零改动」；通知单 4 份归档 iap docs/jxl-deliveries/，在办事项移交 iap TODO。

## 飞轮例行

- [ ] 下批 iap 增量到 → 信号盘点+全模型沉淀（skill data-flywheel 自动触发路径，含第七步台账）
- [ ] SHTM 下周解冻：r2 审核 1,246 帧（用户人工）→ hardcase → V2.2 重训
