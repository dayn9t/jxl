# 任务书 B：P2 基建（s4 会话 → jxl 会话，2026-09-25）

> 依据 [../design.md](../design.md) §5-P2、§8。执行：jxl 会话。可与任务书 A 并行（不同数据面），亦可 A 后接续。
> 执行完回写 `B-execution.md`（方案取舍/冒烟证据/异常）+ README 里程碑打点。

## B① 弱标签抽取器（P3 种子集的进料机）

**输入**：`/var/howell/s4/v0.4/event/mobile/{date}/{id}/`（event.json + llm_entries.json；目录布局单源 `rs/s4-event-fs/src/event_folder.rs`，只读）。
**输出**：预标注 `manifest.jsonl`，每 object 一行：

```json
{"event": "<date>/<id>", "rid": "...", "object_id": "...", "box_norm": [x0,y0,x1,y1],
 "conf": 82.0, "conf_band": "hi|mid|lo", "code": "11", "code_verif": "ebike_single",
 "doubao_cls": "非机动车乱停", "qwen_cls": "非机动车乱停", "agree": true,
 "img_path": "...", "crop_path": null}
```

**组装规则**（CODEBOOK 已定）：
- 位置=平台 Polygon 外接矩形（归一化，已实测 [0,1] 口径）；语义=`agree=true` 行的粗类（doubao∧qwen 一致）；不一致行 `agree=false` 入仲裁队列不入种子。
- 码分档：`11` 高优先（code_verif=ebike_single）；`12` 降权（code_verif=ambig_g12，单独标记供聚类网格列示与后续 VLM 核验）；其余按 CODEBOOK 码表填。
- conf_band：≥80 hi / 50-79 mid / <50 lo（lo 只入候选，不入种子直通）。
- `crop_path` 惰性：抽取时不裁图，P3 按需裁（裁图规范走任务书 A 同款外延原则）。
- 19 粗类映射：llm_entries class 中文串直接用（与 `mobile_names()` 同表）；**叶类映射等 taxonomy.md 冻结后另跑一步投影**，抽取器不负责。

**验收**：抽 20 事件冒烟（含 ET2/ET3/ET14 各 ≥5），字段完整率 100%；agree 率与 conf_band 分布打印；manifest 可回溯到事件文件。

## B② 多类 TargetProfile 方案评估与实施

- 现状：`src/jxl/target.py` TargetProfile 单类假设贯穿 det_mine/la_relabel/doubao_arbitrate；shtm 5 类走独立链路（`projects/shtm/`）。
- 两案对比后择一实施（**方案对比须在同一家族内展开**）：(a) 每叶类一 toml 按类并行跑 det_mine，consensus_dataset 多类合并（改动最小）；(b) TargetProfile 扩多类 spec（一处定义，改 3 工具）。
- 产出：取舍记录（含被否案理由）+ 实施代码 + 以 s4det 任一粗类冒烟。

## B③ n001-pipeline.sh 断链修复

`script/n001-pipeline.sh` 行 331-333 report 阶段仍引用已迁 vlabel 的 consensus_report/review_pack——改用 `vlabel-consensus-report` / `vlabel-review-pack`，跑一次 report 冒烟验证。

## B④ 数据根建仓

- 按 j-dir-readme 选数据盘路径建 `{data_root}/s4det/`，README 含：目的/产出脚本/操作归属（s4det，jxl 会话主导）/运行状态。
- 子目录规划：`p0-cluster/`（任务书 A 产）、`preannot/`（B① 产）、`quarantine/`（隔离台账）。

## 纪律（同任务书 A）

GPU 共卡先查 `nvidia-smi`；中间产物全部落盘；不改 s4 仓；jxl/vlabel 公共工具改动最小化、取舍记录落档。
