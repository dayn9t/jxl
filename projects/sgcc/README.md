# SGCC — n001 收费窗口 person 检测与属性分类（与 SHTM 并列项目）

> 数据根：`/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/crop640_persons/`（本机数据盘）
> 训练机：sgcc0（对等路径 `~/ws/sgcc/...`，见 memory training-on-sgcc0）

## 项目状态（2026-09-09）

| 里程碑 | 状态 |
|---|---|
| 毒标注手术（1,508 框）→ v3 重训 | ✅ 闭环，test mAP50-95 **0.8855**（干净口径） |
| v3.1 实验（微小框+负样本 finetune） | ✅ 增益微弱不采纳，v3 保持部署候选 |
| 属性库 VLM 全量标注（21,744 框） | ✅ 99.97% 成功；uncertain 5,061 → doubao 二审中 |
| 属性分类器族（yolo26l-cls） | ✅ V1 双训完成（2026-09-09）：**upper test 0.974 达标入循环**；coarse test 0.826（val/test 落差 16pp，混淆矩阵分析→V2 已排） |
| 部署（ONNX+symlink 切现网） | ✅ 2026-09-09 13:34 上线（`2026-09-09_person_n.pt/.onnx` 实体 + symlink 切换；回滚：`ln -sfn 2026-07-09_person_n.* person.*`） |

## 文档地图

- `archive/`——历史存档按日期（数据集构建/训练/事故/裁决全记录）。
  关键：`2026-09-08-泛化验证毒标注发现存档.md`（手术全案+架构裁决）、
  `2026-09-08-多属性分类器体系构想存档.md`（属性词典 V2+age 补样方案）、
  `2026-09-07-person-n001-模型训练存档.md`
- `research/`——07 份调研（数据飞轮/共识阈值/PAR/VLM 标注/分类器训练实践/综述）
- 现场：数据根 `gencheck/`（手术/审计/跑批全部脚本与产物）

## 关联

- 通用工具：`src/jxl/bin/`；流程 skill：`~/.claude/skills/consensus-labeling/`
- 并列项目：`projects/shtm/`；VLM 资源：`docs/infra/spark-vlm.md`
