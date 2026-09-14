# SGCC — n001 收费窗口 person 检测与属性分类（与 SHTM 并列项目）

> 数据根：`/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/crop640_persons/`（本机数据盘）
> 训练机：sgcc0（对等路径 `~/ws/sgcc/...`，见 memory training-on-sgcc0）

## 项目状态（2026-09-14 增补：v4 上线 + role v3 七分类）

| 里程碑 | 状态 |
|---|---|
| iapx 三件套（A 重训/B 身份分类器/C 重复框归因） | ✅ 全部落地——**先读 `research/2026-09-12-iapx三件套执行报告.md`**；A 部署物已 stage（`2026-09-12_person_upper_n_v2.*`，symlink 未切待 iapx cache 指纹协同）；B **已到 v3.1 七分类收官**（cleaner 0.879 / teller 0.800 达 0.80 线，最终候选 `2026-09-14_person_role_n_v31.*` 已 stage 待 iapx 对接，词典 `身份分类器词典.md` §6/§7）；C 工具+守卫+v4 回归闭环 |
| person v4 重训（重复 GT 清洗版） | ✅ **已上线**（2026-09-13 23:19 symlink 切换）：test mAP50-95 **0.9005**（v3 0.8855，+1.5pt）、重复框 1,267→7（清零 99.45%）、新日期共识 GT 评估召回 +1.3~8.7pt；回滚 `ln -sfn 2026-09-09_person_n.* person.*` |
| person v5（玻璃反光专项重训，2026-09-14） | ✅ 已训（`runs/person_n001_v5/`，v4 数据+glasspack 426 帧唯一差异）：专项 recall 0.911→**0.930**（miss 14→11），代价 FP +21、test 0.9005→0.8945；**建议 v4 保持现役、v5 存档候选**，切换待用户拍板——详见 `research/2026-09-12-检测器重复框归因.md` §6.2 |
| 身份分类器 v3.1（七分类收官，2026-09-14） | ✅ 词典扩类（teller/manager/security，用户裁决按制服细分）+ uncertain 池 5,541 零人工复审（四模型投票+GLM 视觉仲裁）+ p1 窗扩采（teller 唯一源 1,841）；**最终候选 `2026-09-14_person_role_n_v31.*`**（与 v3 误差内持平、customer 桶污染清洗）；manager/security 素材采集中 |

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
| **upper_body 分类器部署** | ✅ 2026-09-10 同目录上线（用户裁决主力分类器）；**终版=n-cls 2.5M**（n/s/m/l 四档全扫描 0.9782/0.9794/0.9800/0.9774——噪声带内，膝点实测在 n，等精度取最小）：`2026-09-10_person_upper_n.pt/.onnx` + `person_upper.pt/.onnx` symlink。**加载须显式 `task="classify"`**（分类 ONNX 不自动识别 task；.pt 可自动）|

## 文档地图

- `archive/`——历史存档按日期（数据集构建/训练/事故/裁决全记录）。
  关键：`2026-09-08-泛化验证毒标注发现存档.md`（手术全案+架构裁决）、
  `2026-09-08-多属性分类器体系构想存档.md`（属性词典 V2+age 补样方案）、
  `2026-09-07-person-n001-模型训练存档.md`
- `research/`——12 份调研与盘点（数据飞轮/共识阈值/PAR/VLM 标注/属性库回写/归因/执行报告/对接预演等）
- 现场：数据根 `gencheck/`（手术/审计/跑批全部脚本与产物）

## 关联

- 通用工具：`src/jxl/bin/`；流程 skill：`~/.claude/skills/consensus-labeling/`
- 并列项目：`projects/shtm/`；VLM 资源：`docs/infra/spark-vlm.md`
- LLM 统一网关：`~/cc/llmux`（Rust，9 providers × 9 capability traits；
  imgmatch 背景相似度匹配 doubao-seed-2-0-mini top-1 94.9%，可复用于场景/机位匹配，
  bench 见其 `docs/benchmark_report.md`——逐张多图 94.9% vs montage 拼图 59%）
