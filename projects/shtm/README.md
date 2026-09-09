# SHTM — 垃圾桶检测与属性分类（与 SGCC 并列项目）

> 建立于 2026-09-08。项目目标：垃圾桶检测（det）+ 垃圾桶属性分类（cls）。
> 数据源：`/home/jiang/ws/trash`（全部数据）。

## 定位（2026-09-08 盘点后修正）

**复活 + 数据飞轮，不是冷启动**：trash 是 2021-2025 已投产 4 年的上海垃圾管理项目
（~105G / 90 万+ jpg / 6,499 摄像头快照流），已有 cabin 检测（YOLOv8n mAP50-95=0.883，
5 类 opening/lid/dump/person/can）+ sort/amount/direction/side 四属性分类器
（3.7 万+样本，9 个权重，2025-12 仍在迭代），标注工具 jxl_label 即 VLabel 前身
（m31.json schema）。生产部署在本机 ias（`/opt/ias/project/shtm`）。

- **与 SGCC 并列**：共用 jxl 通用工具与流程经验（共识标注/label_audit 审计/泛化验证/
  VLM 批量标注），各自演化、互不耦合。
- **放在 jxl 的目的**（用户裁决）：借助本项目越来越自动化的流程提升 SHTM 标注准确度。
- **SHTM 独有资产**：属性分类轨道（4 分类器实操经验 + 分类样本集管理）、
  已部署模型的增量重训机制（CHANGE.md/get.sh）、误报回流池。

## 目录

```
projects/shtm/
  README.md      # 本章程（裁决/状态/入口）
  research/      # 调研与盘点报告
    2026-09-08-trash数据盘点.md   # ★ 先读这份
```

数据不进 repo（体积），只在此登记指针与结论。

## 检测样本集（2026-09-09 用户澄清 + hash 实锤）

- **主体 = `trash/cabin/samples/`**（images+labels 规范结构，8,899 对；清 110 组内部重复后唯一内容 8,349）
- **`cabin/dates/` 是主体的按日期导入副本，hash 实测 0 张独有**——纯冗余（自带 715 张内部重复），处置待裁（删或降级为工作区）
- **补充源**：新厢房 725 张（无标注，域外候选 test，S0 已体检）+ 误报 7 事件（飞轮回流入口）——需并入主体标注流程
- 干净基准切分（五类均衡 6,730/808/811）基于 8,349 唯一内容，stems 需映射回 samples 路径

## 状态（2026-09-08）

| 项 | 状态 |
|---|---|
| trash 数据盘点 | ✅ 完成（见 research/）——定位修正为「复活+飞轮」 |
| **待裁决** | ① outside(62G)/_arch(31G) 弃留（占 90% 体积基本未标）② illegal 属性是否启动（README 中 TODO 未做，SHTM 目标最大缺口）③ 2025-12 批次 properties 空洞是否 VLM 批量补标（接 spark/182 qwen 免费资源）④ 4 年老标注是否跑 label_audit 质量审计（防 n001 型毒标注） |
| 安全 | ⚠️ dates/README.md 含 howell 平台明文密码（admin/***），纳入项目前须脱敏 |
| m31.json 隐患 | illegal 与 side 属性 id 同为 4（schema bug 待修） |

## 关联

- jxl 通用工具：`src/jxl/bin/`；共识流程 skill：`~/.claude/skills/consensus-labeling/`
- 多属性分类器体系（SHTM 属性分类的方法论模板）：`docs/2026-09-08-多属性分类器体系构想存档.md`
- 训练机规则：训练一律 ssh sgcc0（memory: training-on-sgcc0）
- 免费 VLM 资源：`docs/infra/`（spark + 182 qwen）
