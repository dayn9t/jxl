# SGCC — n001 收费窗口 person 检测与属性分类（与 SHTM 并列项目）

> 数据根：`/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/crop640_persons/`（本机数据盘）
> 训练机：sgcc0（对等路径 `~/ws/sgcc/...`，见 memory training-on-sgcc0）

## 项目状态（2026-09-09）

| 里程碑 | 状态 |
|---|---|
| 毒标注手术（1,508 框）→ v3 重训 | ✅ 闭环，test mAP50-95 **0.8855**（干净口径） |
| v3.1 实验（微小框+负样本 finetune） | ✅ 增益微弱不采纳，v3 保持部署候选 |
| 属性库 VLM 全量标注（21,744 框） | ✅ 99.97% 成功；uncertain 5,061 → doubao 二审中 |
| 属性库回写 VLabel 基准 | ✅ 2026-09-09：gt 17,435 全中（匈牙利 IoU≥0.999），56 手术框无属性；verify-roundtrip PASS——`research/2026-09-09-属性库回写VLabel基准.md` |
| 属性分类器族（yolo26l-cls） | ✅ V1 双训（2026-09-09）；**2026-09-10 用户裁决：upper_body V1（test 0.974）为主力分类器；coarse V1（test 0.817）与 age V1 均冻结**——coarse V2 doubao 重标与 age V1 训练取消（省商用费用与队列），未来业务需要时再解冻（数据集 cls_coarse_psq / cls_age_psq 已备存档，含 COCO 补样的 child/teen 452/116）。错误分析存档 `research/2026-09-09-coarse错误分析与V2方案.md` |
| age 补样（COCO val2017 + spark 伪标） | ✅ 2026-09-09：child 448 / teen 93（conf>0.8；teen 属源上限，134 例悬在 conf(0.6,0.8]）入 `attr_bank/age_supplement/`；四档 `cls_age_psq`（child 452/teen 116/adult 12,798/senior 2,805）建成待训——`data/age_supplement_report.md` |
| 部署（ONNX+symlink 切现网） | ✅ 2026-09-09 13:34 上线（`2026-09-09_person_n.pt/.onnx` 实体 + symlink 切换；回滚：`ln -sfn 2026-07-09_person_n.* person.*`） |
| **upper_body 分类器部署** | ✅ 2026-09-10 同目录上线（用户裁决主力分类器）：`2026-09-10_person_upper_s.pt/.onnx`（s-cls 6.6M，规模扫描实测膝点档）+ `person_upper.pt/.onnx` symlink。**加载须显式 `task="classify"`**（ultralytics 对分类 ONNX 不自动识别 task；.pt 可自动）|

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
