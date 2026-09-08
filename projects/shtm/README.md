# SHTM — 垃圾桶检测与属性分类（与 SGCC 并列项目）

> 建立于 2026-09-08。项目目标：垃圾桶检测（det）+ 垃圾桶属性分类（cls）。
> 数据源：`/home/jiang/ws/trash`（全部数据）。

## 定位

- **与 SGCC 并列**：SGCC = n001 收费窗口 person 检测（数据在
  `/mnt/data/jiang/ws/sgcc/...`）；SHTM = 垃圾桶。两者共用 jxl 的通用工具
  （det_mine / consensus_dataset / label_audit / vlm_ensemble …）与流程经验
  （共识标注 → 审计 → 手术 → 重训 → 泛化验证），**各自演化、互不耦合**。
- **放在 jxl 的目的**（用户裁决）：与本项目一起演化——借助越来越自动化的
  标注流程与审计能力（尤其 2026-09-08 毒标注事故沉淀的 label_audit 体系）
  提升 SHTM 标注准确度；jxl 的组织结构为此调整（`projects/` 装配层）。

## 目录

```
projects/shtm/
  README.md      # 本章程（裁决/状态/入口）
  research/      # 调研与盘点报告（trash 数据盘点等）
```

数据不进 repo（体积），只在此登记指针与结论。

## 状态（2026-09-08）

| 项 | 状态 |
|---|---|
| trash 数据盘点 | 🔄 后台 agent 进行中 → `research/2026-09-08-trash数据盘点.md` |
| 流程复用方案 | ⏳ 等盘点结果定稿（共识标注直接复用；属性分类接多属性分类器体系） |
| 组织结构调整 | 🔄 `projects/{shtm,sgcc}` 骨架已建；sgcc 存量资产（gencheck/ 等）是否迁入待裁决 |

## 关联

- jxl 通用工具：`src/jxl/bin/`；共识流程 skill：`~/.claude/skills/consensus-labeling/`
- 多属性分类器体系（SHTM 属性分类的模板）：`docs/2026-09-08-多属性分类器体系构想存档.md`
- 训练机规则：训练一律 ssh sgcc0（memory: training-on-sgcc0）
