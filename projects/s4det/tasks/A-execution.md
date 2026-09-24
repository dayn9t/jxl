# 任务书 A 执行记录（jxl 会话，2026-09-25）

> 对应 [A-cluster-batch.md](./A-cluster-batch.md)。数据根：`/mnt/data/s4det/p0-cluster/`（1.2G，README 在数据根）。
> 工具链：`projects/s4det/bin/{a_common,a_manifest,a_crop,a_cluster,a_grid}.py`（ruff clean）。
> 管线：扫描建索引 → 五粗类抽样 → crop（bbox 各边外延 10%，vlabel.crop 标准函数）→ embed_dino → dedup_sem(0.95) → 聚类 → 簇代表网格。

## 产出清单

| 族 | manifest | crops | 去重后代表 | k=16 簇数 | 网格 | 码 12 列 |
|---|---|---|---|---|---|---|
| vehicle | 1,015（ET2 950 + ET1 65） | 1,015 | 1,012 | 16 | 2 页 | 7 簇右列 |
| garbage | 1,500（ET3 1,161 + ET11 339） | 1,500 | 1,493 | 16 | 1 页 | — |
| vendor_occupy | 1,000（ET12 586 + ET7 350 + ET6 64） | 1,000 | 997 | 16 | 1 页 | — |
| phenomenon | 991 原图（ET17 615 + ET14 281 + ET18 83 + ET13 12） | 991 | 987 | 16 | 1 页 | — |
| state_carrier | 2（原图+crop，全量） | 2 | 2 | 2 | 1 页 | — |

另：`event_index.jsonl`（32,469 事件全索引）、`sampling_stats.json`、每族 `clusters.json`（簇-成员-代表-码直方）、`threshold_curve.json` + `kmeans_curve.json`（簇数曲线）、隔离台账 579 条。

## 抽样纪律执行

- **跨月均匀**：逐月轮转抽样，五月仅 garbage/vendor 有数据（vehicle ET2 六月始有）；实现为逐月 round-robin。
- **去过密**：(日期, 设备, 3h 桶) 上限 4。**注意：cap=2 时 ET2 容量实测仅 283 事件**——mobile 为车载移动巡查相机（全库仅 8 台，单日单机可爆发 979 事件），同组内是巡查沿途不同目标而非同目标重复；cap=4 + 下游 SemDeDup 语义去重双保险。指标真实瓶颈是 143 个 (日×机×时段) 组的组容量，非数据量。
- **事件内选框**：conf 降序 top-3；Conf<50 全部保留入聚类（分档：vehicle <50 占 22%、vendor <50 占 45%）。
- **无框事件**：vendor 族无实际发生（覆盖率 98.3%），fallback 未触发。

## 与任务书的偏差（均经实证，已与 s4 会话对齐）

1. **ET15/16/19 在 event.json 中 0 事件**——三类是本管线 LLM 分类产出、平台不以此下发 EventType（s4 会话确认并落档盘点 §8.1）。按 llm_entries doubao∧qwen 一致类别从全库补样（源标 `source=llm_class:*`）：违规充电 1、占道经营 121、散落垃圾 30。
2. **DINOv3 → dinov2-small**：DINOv3 为 HF gated repo 无法拉取；embed_dino 默认 dinov2-small（384d，n001 同款）。s4 会话确认无异议。
3. **聚类主档由余弦阈值法改为 k-means**：任务书预设的 Faiss 单链阈值法在此数据两端失效——0.50 链成巨簇（vehicle top954）/ 0.85 全碎（940 簇近全孤），三档曲线存于 `threshold_curve.json`。主簇表改用 dedup_sem stage-2 同款 MiniBatchKMeans（k 扫 8/12/16/24/32 存 `kmeans_curve.json`，主档 k=16）。**命名侧如需更细/更粗粒度，`a_cluster.py --k <n>` 重跑即可。**

## 数据质量发现（已通报 s4 会话，任务书 B 需处理）

- **平台 object Id 重复下发**：965/32,469 事件（3%）同事件 Objects[] 内同 Id 重复（超额 1,079 个，抽验逐字节雷同）。解析层已去重（全等丢弃+计数，冲突双保留 #n 后缀；本次冲突=0）。
- **平台框噪声可观**：garbage k=16 后半簇多为噪声形态（石球被码 1 框入、木栅栏入码 2、路缘石入码 4）——misc.md 明确排除物被平台框收录。**P3 弱标签必须走共识+仲裁，不能直接信平台框**。
- **码 12 定性证据（CODEBOOK 待定码人审素材）**：vehicle 族码 12 代表簇 7 个全部为路面/路缘/盲道区域框，三假说中「泊位·路缘地面区域」压倒性成立（网格右列即看）。

## 异常与弃置

- 隔离台账 579 条（`projects/s4det/data/quarantine-ledger.jsonl`）：event.json 缺失/解析失败 578、原图缺失 1。crop 阶段零弃置、零缺失。达 j-quarantine-ledger 阈值（50），按规需记 j-todo 待人工批量处理——本批为「源目录结构性缺失」（date/{id} 目录无 event.json），非模型门拒绝，建议人工批量确认为「确认丢弃」类。

## 复跑指引

```bash
# 抽样（索引已存在时秒级重算；--rescan 强制重扫 ~9-40 分钟）
.venv/bin/python projects/s4det/bin/a_manifest.py
# 裁切（--force 重跑）→ embed/dedup → 聚类 → 网格
.venv/bin/python projects/s4det/bin/a_crop.py --root /mnt/data/s4det/p0-cluster
.venv/bin/python -m jxl.bin.embed_dino <族>/crops <族>/embeddings.npy --device cuda
.venv/bin/python -m jxl.bin.dedup_sem <族>/embeddings.npy <族>/crops <族>/dedup --sem-threshold 0.95 --target 100000
.venv/bin/python projects/s4det/bin/a_cluster.py <族> [--k 16]
.venv/bin/python projects/s4det/bin/a_grid.py <族>
```

批指纹：每族 `run_config.json`（manifest sha256）+ `crop_report.json`；下游脚本拒绝跨批操作。
