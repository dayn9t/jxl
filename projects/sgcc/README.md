# SGCC — n001 收费窗口 person 检测与属性分类（与 SHTM 并列项目）

> 数据根：`/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/crop640_persons/`（本机数据盘）
> 训练机：**sgcc3**（默认，2026-09-15 起；RTX 5060 Ti 16G，环境就绪 torch cu128/sm_120 实算验证——
> 赶时间可按用户授权用 sgcc0 或双机并行；对等路径 `~/ws/sgcc/...`，见 memory training-on-sgcc3）

## 项目状态（2026-09-18 增补：iap 夜间全链批交接）

| 里程碑 | 状态 |
|---|---|
| **飞轮第一轮全模型（2026-09-19 晨）** | ⚠️ **v7 预冒烟 FAIL 已撤回（坐姿边缘回归 2 框，归因=pospool bad_fit 弃帧丢坐姿形态）——正式改推 v6 切换**；v7.1=台账复核回收坐姿正样本后重训。v7 原验收数据（mAP 0.8981/FP 探针 0/保真 <0.003px；upper v3 回归门过/legs 门 174 张 +2 张判条件过；role v3.3 0.8858 门内（代理级标签待人工抽检）；**OSNet v2.3 三档 sweep 全档证伪——v2.2/v2.3 双证伪加固 v2.1 终态**。md5 v7 `71165d39`/`91ba15c9`（v6/v7 双 staged 未切）；通知单 `~/cc/py/iapx/docs/jxl-deliveries-2026-09-19.md` |
| **person v6（座椅区静物 FP 修复，飞轮首转）** | ✅ **验收 PASS、已 stage 待 iapx/iap 切换**：FP 探针（neg_2026-07-05 独立 35 张）v5 1 检出→**v6 0 检出**；test mAP50-95 0.8979（v5 seed 带内上段，无回归）；ONNX 保真 <0.005px。v5 数据+66 张 neg0906 负样本（spark VLM 66/66 全检无人）单变量，val/test 逐字节冻结；seed1 主轮（sgcc0，sgcc3 seed0 因升级中止 @ep17）。部署物 `2026-09-18_person_n_v6.pt/.onnx`（md5 `109af4b7`/`f7762723`，sgcc0 /opt/howell v0.10 stage）；通知单 `~/cc/py/iapx/docs/jxl-deliveries-2026-09-18.md`。报告 `research/2026-09-18-person-v6交付报告.md` |
| iap 夜间批交接（09-05~09-17 新数据全链） | 📥 **已入库**（`research/2026-09-18-iap夜间全链批交接报告.md`）：424 段 v5 四指纹缓存落 n001（v5 上线后首批全域生产数据）；★09-06 误检框群（10,121 恒定尺寸框，员工座椅区、ROI 外）= person_n 下一轮 hard-negative 候选；09-05+ 新日期域表现评估素材。待办：难例提取 + 新日期域评估 |
| **09-06 误检簇处置 + v6 素材** | ✅ **看图门+提取+全检+入池全链当日完成**：spark VLM 三时段裁决=静物非人（排除留守人员，09-08 教训门）→ 10,121 框聚 35 簇 → **66 张 640×640 crop 全检无人零剔除**入池就绪 `gencheck/neg0906/`（形态同 v5_glass）；知会 iap `~/cc/next/iap/docs/jxl-notice-2026-09-18-neg0906-hard-negatives.md`（含 FP 探针参照集建议） |
| **sgcc3 烤机 + v5 seed 带归因** | ✅ **17.7h 五 seed 满载零故障**（61°C/零降频/零 Xid/耗时 ±0.1%）——sgcc3 长训资格实证；**v4→v5 mAP 降幅 0.006 == 5-seed spread 0.006 → v5「代价」是 seed 噪声，上线决策无需翻案**；方法学：该量级数据集 <0.006 单次 mAP 差异不可判优劣。报告 `research/2026-09-18-sgcc3烤机报告与v5-seed带归因.md` |
| **v5 新日期域画像** | ✅ **无退化迹象**：正常营业新段（09-05/07/08/17）vs 06-22 框/h -2%、conf 稳 0.9/1.0、误检扁带占比反降（25.9→14.8%）——**不支持紧急域适配**；新发现 07-05（训练域内）同闭馆静场模式 → 静物误检为跨域稳定弱点非新域问题；src2 框高 +34% 同向漂移列待复核。⚠️ 前提修正：历史段仅 06-22 有 v5 缓存（余 pre-v5）——建议 iap 历史段 v5 重检扩基线。报告 `research/2026-09-18-v5新日期域画像.md` |
| **v6 立项判断（三证据合流）** | 🟡 **建议立项、非紧急**：烤机归因（无翻案动机）+ 09-06/07-05 静物 FP 簇（真动机：座椅区静物负样本缺失，但生产被 ROI 门隔离）+ 域画像（无退化、不加域素材）→ **v6 = v5 + neg0906 负样本**（07-05 同模式素材可选并入），验收加「座椅区 FP 探针」指标（v5 基线 1/帧 → 目标 0）；可与下个迭代合流，训练机 sgcc3 |

## 项目状态（2026-09-15/16：v5 真机验收上线 + OSNet v2 + 训练机切换）

| 里程碑 | 状态 |
|---|---|
| **v4 坐姿召回回归（生产实证）** | ⚠️ **v4「保持现役」结论被推翻**（2026-09-15 生产点火）：src2 07-06 全日语料 v4 对隔玻璃+反光坐姿人物整段漏检（v3 同域对照检出），生产已回退 v3。证据 `research/2026-09-15-生产点火v4坐姿回归证据与pairlist交付.md` |
| **person v5 真机验收 + 上线** | ✅ **验收 PASS、已 stage 待 iapx 切换**：塌陷段 226/226 帧闭合、C 阳性帧覆盖 99.8%、FP raw 仅 +3.6%（离线 FP +21 未在生产放大）、铁证帧 conf .916。**09-16 跨日期扩展验收再 PASS**（06-22/07-31/09-03 三日期 5,328 帧：零分钟塌陷、raw +0.67%、帧覆盖 99.92%、独有检出 77% 真人——未覆盖仅剩 src1）。部署物 `2026-09-15_person_n_v5.pt/.onnx`（md5 `a135025a`/`ee129ec3`，ONNX 铁证帧保真 <0.11px）；通知单 `~/cc/py/iapx/docs/jxl-deliveries-2026-09-15.md`。报告 `research/2026-09-15-v5真机验收报告.md` + `research/2026-09-16-v5跨日期扩展验收.md` |
| sitpack_v6（坐姿盲区数据包） | ✅ **四票完整包已冻结归档**（621 帧/1,036 框，glasspack 兼容；豆包第四票 09-16 补齐 623/623、0 弃权、框级一致 98.65%，手术剔 32 框/推翻 2 帧）：v5 已闭合坐姿缺口故**归档不并包**；spark 恢复后重跑仅为可选的回归原四模型口径。`research/2026-09-15-v6坐姿数据包预备.md` |
| **role v3.3 时序聚合 spike** | ✅ **实证否定**：cleaner/leader 错误 100% 个体级系统性（12/12 错分轨零对帧，oracle 聚合上限=逐帧），**词典 0.80 线对两类单列不适用**（已修词典）；softmax 全向量契约保留作消费侧平滑自由度。`research/2026-09-15-v33时序聚合spike.md` |
| **OSNet v2 域微调（#26）** | ✅ **jxl 侧验收 PASS**：triplet 微调后 eval gap −0.094 分布分离（基座 +0.116 FAIL）；**并发现 v1 判据方向写反**（余弦距离误用相似度口径，v1「恶化」论据失效——需求文档 §7 已修）。交付 `osnet_x0_75_ft_v2.pth/.onnx`（md5 `a33775ef`/`e235228a`）；§7.4 定标门+管道门待 iapx 复测。`research/2026-09-15-osnet-v2微调报告.md` |
| 训练机切换 | sgcc0→**sgcc3**（默认，环境就绪）；数据 12G/36 万文件已机间直传；活跃脚本 host 已切（osnet_ft REMOTE_HOST 归位 sgcc3）；**依赖源修正：本仓 uv.lock 用 devpi（192.168.18.146:3141），勿用阿里镜像**。spark（=182 vLLM 机）并发 ≤3 红线（8 并发曾压死整机） |

## 项目状态（2026-09-14 增补：v4 上线 + role v3 七分类）

| 里程碑 | 状态 |
|---|---|
| iapx 三件套（A 重训/B 身份分类器/C 重复框归因） | ✅ 全部落地——**先读 `research/2026-09-12-iapx三件套执行报告.md`**；A 部署物已 stage（`2026-09-12_person_upper_n_v2.*`，symlink 未切待 iapx cache 指纹协同）；B **已到 v3.1 七分类收官**（cleaner 0.879 / teller 0.800 达 0.80 线，最终候选 `2026-09-14_person_role_n_v31.*` 已 stage 待 iapx 对接，词典 `身份分类器词典.md` §6/§7）；C 工具+守卫+v4 回归闭环 |
| person v4 重训（重复 GT 清洗版） | ✅ **已上线**（2026-09-13 23:19 symlink 切换）：test mAP50-95 **0.9005**（v3 0.8855，+1.5pt）、重复框 1,267→7（清零 99.45%）、新日期共识 GT 评估召回 +1.3~8.7pt；回滚 `ln -sfn 2026-09-09_person_n.* person.*` |
| person v5（玻璃反光专项重训，2026-09-14） | ✅ 已训（`runs/person_n001_v5/`，v4 数据+glasspack 426 帧唯一差异）：专项 recall 0.911→**0.930**（miss 14→11），代价 FP +21、test 0.9005→0.8945；**建议 v4 保持现役、v5 存档候选**，切换待用户拍板——详见 `research/2026-09-12-检测器重复框归因.md` §6.2 |
| 身份分类器 v3.2（复判数据集重训，2026-09-14） | ✅ **新交付候选已 stage**（`2026-09-14_person_role_n_v32.*`，pt/onnx md5 配对 `2364d5e7`/`16be90c7`）：数据=8,868 旧源图四模型复判（改判 **909 族/3,185 文件，占 16.9%**——uid 去重口径；抽检 12/12）+mgrsec security 91。test top1 **0.8877**（5 组跨类双标签修复后干净口径）；**manager 0.974 / security 0.95 / teller 0.896 / customer 0.80 达线**（5/7），cleaner 0.714 / leader 0.50 现出真实难度（动作/工装在单帧 crop 的天然边界，后续需时序聚合）。v3.1 行保留作历史：其 0.8846 建立在污染 test 上 |
| 身份分类器 v3.1（七分类，已被 v3.2 取代） | 词典扩类（teller/manager/security）+ uncertain 池 5,541 零人工复审 + p1 窗扩采；候选 `2026-09-14_person_role_n_v31.*` 仍 stage 在位可回退 |

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
- `research/`——18 份调研与盘点（数据飞轮/共识阈值/PAR/VLM 标注/属性库回写/归因/执行报告/对接预演/v5 验收/OSNet 微调/spike 等）
- 现场：数据根 `gencheck/`（手术/审计/跑批全部脚本与产物）

## 关联

- 通用工具：`src/jxl/bin/`；流程 skill：`~/.claude/skills/consensus-labeling/`
- 并列项目：`projects/shtm/`；VLM 资源：`docs/infra/spark-vlm.md`
- LLM 统一网关：`~/cc/llmux`（Rust，9 providers × 9 capability traits；
  imgmatch 背景相似度匹配 doubao-seed-2-0-mini top-1 94.9%，可复用于场景/机位匹配，
  bench 见其 `docs/benchmark_report.md`——逐张多图 94.9% vs montage 拼图 59%）
