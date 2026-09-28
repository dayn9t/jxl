# 任务书 A：P0 类目聚类跑批（s4 会话 → jxl 会话，2026-09-25）

> 依据 [../design.md](../design.md) §2/§5-P0 与 [../research/2026-09-25-目标分类学与数据源盘点.md](../research/2026-09-25-目标分类学与数据源盘点.md) §3。
> 执行会话：jxl（/home/jiang/cc/py/jxl）。s4 会话只答辩语义/命名，不代跑。
> 执行完回写本目录 `A-execution.md`（跑批参数/簇数/异常）并在 README 里程碑打点。

## 目标

对 mobile 域五大粗类各产出**形态聚类报告 + 簇代表网格图**，供 s4 会话人工命名细分叶类、冻结两级类目表 v1（taxonomy.md）。

## 输入

- 数据源（只读）：`/var/howell/s4/v0.4/event/mobile/{date}/{id}/`（event.json 含平台框；llm_entries.json 含双模型类别）
- 抽样清单（s4 会话已定，见下表；数量为目标上限，实际按数据量截断）

| 粗类 | 事件来源（EventType） | 样本形态 | 目标数量 |
|---|---|---|---|
| vehicle | ET2 非机动车乱停(13.6k)、ET1 机动车乱停(467) | 平台框 object crop | 1,500 |
| garbage | ET3 丢弃垃圾(12k)、ET11 占道堆物(497)、ET19 散落垃圾 | 平台框 object crop | 1,500 |
| vendor/occupy | ET12 占道设摊(1.1k)、ET16 占道经营、ET6 施工(64)、ET7 店铺装修(721) | 平台框 crop（无框事件用原图） | 1,000 |
| phenomenon | ET13 油污(12)、ET14 积水(3.3k)、ET17 井盖(615)、ET18 道路破损(83) | **原图**（区域类无点框；小类全量） | 1,000 |
| state-carrier | ET15 违规充电（仅 1 事件，数据极缺） | 原图+crop | 全量（预期很少，如实报告） |

抽样纪律：跨日期均匀（每月都抽到）；同一摄像头/同时段去过密时降采样；Conf<50 的框**保留入聚类**（聚类不怕噪，只怕偏）但需记录分档统计。

## 步骤

1. 建数据根（若未建）：数据盘路径由 jxl 会话按 j-dir-readme 选定建仓（README 含目的/产出脚本/操作归属=s4det）。
2. manifest 生成：按上表抽事件 → 产 `manifest.jsonl`（event 路径、rid、object_id、框、Conf、llm_entries 类别、粗类标签）。
3. crop 生成：平台 polygon → 外接矩形 crop，**外延遵循 skill `bbox-crop-expansion`（小幅外延原则，禁固定窗口）**；原图样本直接拷贝/链接。
4. 去重：dedup 链先跑（embed_dino → dedup_sem），防止同一目标多帧重复压簇。
5. 聚类：DINOv3 embedding → Faiss 余弦 ~0.85 档（可试 0.80/0.85/0.90 三档，报告簇数曲线）；每簇保留代表 + 规模。
6. 网格：review_pack 出每粗类簇代表拼图（簇 id + 张数标注），落数据根。
7. 弃置样本（无法读图/空框/解析失败）入隔离台账 `data/quarantine-ledger.jsonl`（后迁 /mnt/data/s4det/quarantine/quarantine-ledger.jsonl 为单一数据源，见 A-execution「异常与弃置」及 B-execution B④）（j-quarantine-ledger）。

## 产出（全部落数据根 + 本目录）

- `{data_root}/s4det/p0-cluster/{族}/`：crops/、clusters.json（簇-成员-代表）、grids/ 网格图
- `A-execution.md`：参数、每族簇数、Conf 分档统计、异常与弃置量

## 验收（s4 会话核）

- 五粗类全覆盖；每族簇数合理（预期 5-30）；网格人眼可辨（能看出形态分组）；manifest 可追溯到事件目录。

## 纪律

- GPU：本机 4060Ti 共卡，训练/推理前 `nvidia-smi` 查余量，绝不杀业务进程；不依赖 ds。
- 中间结果全部落盘（调试性硬约束）；脚本带批处理指纹守卫。
- 不改 s4 仓、不改 jxl/vlabel 公共工具代码——需要新工具时放 `projects/s4det/bin/`（多类 profile 改造属任务书 B，勿混入）。
