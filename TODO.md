# TODO（SGCC/SHTM 双项目）

> j-project-docs 骨架。**防止被循环输出淹没的显式标记区**——升级队列复审、
> 稀有样本、跨模型分歧等长周期条目在这里，不在对话流里。

## 🔁 升级队列（escalation queue——定期人工批量复审）

- [ ] **P1 跨模型分歧样本**（cross_model.disagreement）——spark vs 4.5v 边缘
      判定分歧逐条补录+复审；首条：v6 预冒烟 bad_fit 边缘截断框（2026-09-19）
- [ ] **P1 cleaner 窗前交互稀有样本 ×1**（rolepool excluded 内，role_conf 0.97）
      ——人工裁后若为真，入 role 稀有类正样本
- [ ] **P2 person bad_fit 弃帧复审**（~100 帧含 119 个 pass 框）——分离
      「可回收边缘截断」vs「真不可信框」；回收帧入 pospool
- [ ] **P2 osnet rejected 114 对复审**——分离「实证异人」（可作 diff 负对
      素材！）vs「不可判定」（重审或弃）；实证异人对是 v2.4 负对方向的现成素材
- [ ] **P2 role 归因失败 7 复审** + **P3 upper removed 7 / person not_person 57**
- 触发阈值：任一 P1/P2 源 ≥100 条或每周五；队列实体
      `gencheck/escalation_queue/INDEX.jsonl`（各轮飞轮 rejected 同步登记）

## ⏳ 等用户/外部

- [ ] role v3.3 切换前 50 张人工抽检（清单
      `gencheck/rolepool_20260919/rolepool_spotsample_checklist.md`）
- [ ] ASR 评估集 5 决策点（方案
      `projects/sgcc/research/2026-09-19-ASR评估集设计方案.md`）
- [ ] v6/v7 检测器切换裁决（双 staged，iapx/iap 侧；通知单
      `~/cc/py/iapx/docs/jxl-deliveries-2026-09-1{8,9}.md`）
- [ ] **RAP 拆解 iapx 的范围确认**（若涉及 WeightSpec/REID_KINDS 注册接口
      或通知单惯例，需知会 jxl——2026-09-19 用户提出，jxl 侧暂无影响实证）
- [ ] sgcc3 升级完成 → 栈验证+轻量烤机（jxl 执行，等用户告知完成）
- [ ] SHTM 解冻（下周）：r2 审核 1,246 帧（用户人工）→ hardcase → V2.2 重训

## 📋 已知技术债

- [ ] iapx `consts.VIDEO_ROOT` 仍指 v0.9（jxl 已两次提醒，待 iapx 会话修）
- [ ] iap 增量批例行化（cron/手动触发词）未落地——飞轮下一循环的数据前提
