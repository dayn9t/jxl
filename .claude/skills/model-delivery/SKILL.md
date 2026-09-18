---
name: model-delivery
description: SGCC n001 模型交付流程（jxl → iapx/iap）。当模型训练完成且验收 PASS、准备交付/部署/staged 新版本权重、导出 ONNX、写交付通知单、或交付后发现问题需勘误时使用——双证据验收→导出 md5→注册→通知单→收尾五步走全，不靠记忆装载。
---

# 模型交付流程（jxl → iapx/iap）

> 前身：v5 / upper_v2 / role_v31 / OSNet v2 / v2.1 五次交付的经验固化。
> 知识索引：memory `iapx-delivery-notice`（通知单惯例 Why）；实例详报见各
> `projects/sgcc/research/` 交付文档。本 skill = 步骤序列，参数查指针。

## 交付五步（顺序不可倒）

### 1. 双证据验收（没有证据不交付）

- 通用门：eval/held-out **不回退**（held-out 冻结集，防自我强化）+ 专项指标改善
- 检测器加：多 seed 刻度——**单次 mAP 差 <0.006 不可判优劣**（2026-09-18 烤机
  结论），跨版本结论必须带 seed 带宽说明
- 任何一条证据被事后推翻 → 勘误流程（见第 5 步）

### 2. 导出与指纹

- `.pth`（unwrap state_dict）+ `.onnx`（BN 版用 `gencheck/export_osnet_bn_onnx.py`
  模式：dynamo=False 防动态 batch 固化；检测器导出链查既有脚本）
- **md5 双记录**（交付表 + 通知单），ONNX 附 torch-vs-onnx 一致性数字
- 权重落 `~/ws/sgcc/person/<族>_weights/`（数据盘规范位置）

### 3. 注册/部署物

- reid 类：iapx `src/iapx/pipeline/osnet.py` WeightSpec + `src/iapx/cfg.py`
  REID_KINDS（同 commit 双点）
- 检测器/分类器：部署物 dated 目录 + `person.*` symlink（生产在 iap 侧
  `/opt/howell/iap/v0.10/`——jxl 只 stage，切换由 iap/iapx 裁决）

### 4. 通知单（交付即写，不等用户要求）

- 位置：`~/cc/py/iapx/docs/`（reid/检测器）或 `~/cc/next/iap/docs/`（素材/知识交付）
- 必含：交付物+md5 表 / 接收方动作序列 / 契约（tag/输入/规则）/ **复测靶点**
  （哪个门、什么基线、目标值）/ 消融提示（出问题先怀疑什么、按什么字段分组）
- 通知单是**滚动文档**：预复测结果、勘误都在同一文件追加段落，不另起

### 5. 收尾与勘误

- `projects/sgcc/README.md` 状态表加行 + memory 快照更新
- **勘误惯例**（v2.1 案例）：发现证据缺陷（如铁证对错标）→ research 文档加
  ⚠️ 勘误块 + 通知单同文件追加勘误段 + memory 更正——**影响面说明三件套**
  （什么作废/什么仍成立/下游要不要动）

## 检查清单（交付前过一遍）

- [ ] 双证据齐且未被勘误
- [ ] md5 双记录、ONNX 一致性验证过
- [ ] 注册点/部署物实际落位（grep 验证，不凭记忆）
- [ ] 通知单含复测靶点与消融提示
- [ ] README + memory 已更新
