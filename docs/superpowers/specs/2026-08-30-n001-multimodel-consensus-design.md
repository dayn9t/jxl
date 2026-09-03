# n001 视频多模型共识标注设计（2026-08-30）

> `/var/howell/iap/v0.9/ias/sh-sgcc/n001/video`（806 mkv × 600s = 134h，两摄像头 1/+2/）
> → I 帧 4:1 抽取 → 前景感知去重 → det_mine 五模型共识标注 → 可信度分级 + 模型能力矩阵
> + 人工审核材料。**停在豆包之前**（规模出来用户再定下一步）。

## 1. 背景与目标

n001 部署点存有 134h 营业厅监控视频（已有部署 person.pt 的逐秒 json 检测记录，多数空）。
目标：多模型共同标注产出**尽可能可信**的人员检测标注，按可信度分级，
高争议进人工审核；同时产出**各模型准确度与擅长领域矩阵**（历史空白：从未统一横评）。

## 2. 已确认决策（用户裁决）

| 决策点 | 选择 |
|---|---|
| 关键帧提取 | 视频 I 帧 4:1 下采样（GOP=2s → 有效 8s 间隔，~6 万帧）|
| 去重 | 复用 SemDeDup 前景感知管线（可靠性 Phase A 验证）|
| 规模控制 | **所有标注工具都无目标的图直接删除**（det_mine L0-drop 语义）|
| 豆包 | 不进管线，review 集停在豆包之前，规模出来再定 |
| 人工审核形态 | PIL 预览网格（每模型一色框）+ manifest.jsonl |
| 产出用途 | 先保留不入池，做完再决定 |
| target 模型 | `person_yolo26n/weights/best.pt`（sgcc0 从头训 mAP 0.862；旧 person.pt 本机已不存在）|
| 编排 | 新编排 bin 串 5 stage，各 stage 独立可重跑（分批断点）|
| **分布式标注** | **本机(4060Ti) + s4(RTX 5080 16G) 双机**：帧对半分片，各自独立跑全套 det_mine（含各自 la 服务），产物合并。sgcc0/6/1 不可达，不参与。s4 按「同步到相同目录」方式部署（rsync jxl repo + 模型 + 帧分片）|

## 3. 总流程

```
/var/howell/.../n001/video/*.mkv (806×10min, 24.2万 I 帧)
   │ Stage 1 抽帧: ffmpeg skip_frame nokey → 4:1 下采样(留 index%4==0)
   ▼
raw_frames/ (~6万帧, ~18G)
   │ Stage 2 去重: person_crop(YOLO) → DINOv2 embed → SemDeDup cos≥0.95
   ▼
frames_dedup/
   │ Stage 3 共识: det_mine --target-model person.pt
   │             --validators yoloe,gdino,rfdetr,la --consensus 2
   ├─ L0 全一致(含全员空图) → 丢弃（规模控制指令）
   ├─ L1 低争议共识 → 自动标注(RF-DETR 优先框)
   └─ review 高争议 → 待人工/豆包
   ▼
Stage 4 分级+矩阵:  T1 五模型同检 / T2 多数共识 / T3 分歧
   │                + 模型两两一致性 + 框尺寸/密度分桶擅长领域
   ▼
Stage 5 审核材料:  review/ 网格预览 + manifest(逐模型输出)
   ▼
datasets/sgcc-n001/（不入池）
```

## 4. Stage 细节

### Stage 1 抽帧（新 bin `video_keyframe.py`）
- `ffmpeg -skip_frame nokey -i in.mkv -vsync vfr frames/%06d.jpg` 提取全部 I 帧
- 按解码序号 `idx % 4 == 0` 保留（4:1）；命名 `{mkv_stem}_{seq:04d}.jpg` 保源可溯
- 处理状态落 jsonl（断点续跑：已处理 mkv 跳过）；ffmpeg 单进程 ~10s/mkv，可 `--jobs N` 并行

### Stage 2 去重（现成管线）
- 复用 2026-06-25 前景感知管线：`person_crop`（YOLO bbox 前景）→ `person_embed`（DINOv2 384d）
  → `person_dedup`（Faiss cos≥0.95 + 并查集）→ 整图姿态簇指纹去重
- 负样本保留语义：无 crop 图原样保留（07-09 修复后行为）——全员空删交给 Stage 3 L0
- **Phase A 可靠性验证**：抽样对比去重前后，统计「有人图被去重删除」数（应为 0）

### Stage 3 共识标注（现成 det_mine + 编排分批 + 双机分片）
- 校验器：yoloe + gdino + rfdetr + la（la 服务 :18306 须常驻）
- 分批调用（每批 ~2000 图，断点粒度=批；服务中断续跑），批产物合并
- 参数：`--iou 0.3 --consensus 2 --review-top 0.3`（沿用 07-08 校准）
- **双机分片**（标注时长减半）：
  - 前置：s4 部署（Phase A 完成）——rsync jxl repo → `~/cc/py/jxl`、模型权重 →
    相同路径、`uv sync` 主环境、`la-setup.sh` 重建 la-venv、la-serve.sh 起服务
  - frames_dedup 按 stem 排序奇偶分两片：`rsync` 偶数片 → s4 相同数据路径
  - 本机/s4 各自跑分片内 det_mine（各自 la 服务，`--la-url` 默认本机即可）
  - 产物按片合并回 `consensus/`（stem 无碰撞，直接拼）
- **s4 部署风险**：RTX 5080 = Blackwell sm_120，本机 la-venv（torch 2.6 cu12 +
  sm89 flash-attn wheel）**不可复用**——s4 重建 la-venv 需 torch ≥2.7 cu128；
  flash-attn 无 sm_120 预编译 wheel 时 `--attn sdpa` 显式降级（质量等价、速度略降，
  降级需在 s4 报告中标注）。Phase A 首项验证

### Stage 4 分级 + 模型能力矩阵（新 bin `consensus_report.py`）
- 汇总各批 review manifest + L1 labels + 各模型逐图输出（det_mine 需扩展：全量图记录
  validators 输出到 jsonl，不止 review 集——小改 det_mine 或编排层捕获）
- 可信度三级：T1 = 5 模型（target+4 校验器）全同检；T2 = 共识成立（L1）；
  T3 = review（分歧）
- 模型矩阵：模型两两 IoU≥0.5 一致率；每模型 vs 共识的 P/R；按框高（<40px 远小 /
  40-150 / >150）、图内框数（密集>8）分桶——产出各模型擅长领域结论
- 输出 `accuracy_report.md` + `model_matrix.json`

### Stage 5 人工审核材料（新 bin `review_pack.py`）
- review 集逐图 PIL 网格：原图 + 各模型一色框叠加（target 黑/yoloe 蓝/gdino 黄/
  rfdetr 绿/la 红）+ 顶部 stem 与分歧摘要
- `manifest.jsonl`（沿用 det_mine 字段）+ 网格分片（20 图/张）
- 人工操作指引：看图标注「采信哪个模型的框/修正」——形式从简（看网格图改 labels 文件）

## 5. 两阶段推进

- **Phase A 校准**（8 个 mkv 随机，~600 帧去重前）：
  1. **s4 部署验证**（首项）：repo/模型/环境同步 → la 服务起（sdpa 降级路径可接受）
     → 20 图冒烟对比本机（框 IoU 一致性）
  2. 全链路跑通（本机），产出去重率 / 空图率 / L1:review 比 / 模型矩阵初版
  3. 外推全量（帧数、双机 GPU 时、审核量）→ **用户确认后再跑 Phase B**
- **Phase B 全量**：806 mkv（本机抽帧 ~1h 并行 / 去重 ~1h / **双机标注 ~5-10h** /
  rsync 分片 ~分钟级 / 报告分钟级）

## 6. 产出目录

```
datasets/sgcc-n001/
├── raw_frames/        # Stage1 产物(中间量, Phase B 后可清)
├── frames_dedup/      # Stage2 产物
├── consensus/         # det_mine 产物(images+labels+review+manifests)
├── review_pack/       # 网格预览 + manifest(人工审核入口)
├── accuracy_report.md # 模型能力矩阵(用户新增需求)
└── pipeline_report.json # 全链路统计
```

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| la 服务 20h+ 长跑中断 | 分批断点续跑（批粒度）；la_relabel 同模式已验证 |
| s4 sm_120 环境不兼容 | Phase A 首项验证；sdpa 显式降级保底（质量等价）|
| 双机产物 stem 碰撞 | stem 含 mkv 源名+序号全局唯一；合并按片拼接 |
| s4 中途失联 | 分片独立断点，失联不影响本机片；恢复后续跑 |
| GPU 13.3G/16.4G 贴边 | 业务服务保持停止；分批间检查显存 |
| 去重误删有人图 | Phase A 抽样验证删除对（07-09 实战 0 误删）|
| det_mine 全量图模型输出缺失 | 扩展记录 validators 逐图输出（Stage4 依赖）|
| 中间量 18G 磁盘 | /mnt/data 余 1.3T，充裕 |

## 8. 范围与依赖

- **复用**：det_mine（la 校验器）、SemDeDup 管线（person_crop/embed_dedup）、la 服务
- **新增**：`video_keyframe.py`（I 帧 4:1）、`consensus_report.py`（矩阵+分级）、
  `review_pack.py`（审核材料）、det_mine 全量输出扩展、编排脚本 `script/n001-pipeline.sh`
- **许可**：la 非商用——产出若入训练池需剥离 la 独有贡献框或仅作评估参考（Phase B 后
  与用户确认；本次先保留）

## 9. 关联

- det_mine: `src/jxl/bin/det_mine.py`；sgcc 重标注先例: [[2026-07-08-sgcc-relabel-存档]]
- 去重管线: [[2026-06-25-数据去重存档]]；la 评估: [[2026-08-30-sgcc-la重标注与验证存档]]
- 视频抽帧先例: `ff_extract.py` / `frame_sample.py`
