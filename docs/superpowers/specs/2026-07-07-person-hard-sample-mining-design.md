# Person 检测难例挖掘设计（2026-07-07）

> Cross-detector 难例挖掘 pipeline：从监控 mkv 录像抽关键帧 → person.pt（部署模型）与 YOLOE-11l（通用强模型）双检测交叉比对 → 分歧样本自动标注回灌训练集，迭代提升 person.pt 在小目标/遮挡/密集场景的召回。

## 1. 背景与目标

部署中的 person 检测器 `/opt/howell/iap/current/ias/model/person.pt`（YOLO26n，单类 person，sgcc+MOT17/MOT20+COCO 混合训练，test mAP50=0.9545，**Recall=0.8892**）在监控场景存在系统弱点：小目标/遮挡/密集场景约 11% 漏检（详见 `docs/2026-06-27-person模型训练部署.md`）。

本 pipeline 从真实监控录像自动发现该模型检测不可靠的样本（难例），自动标注后并入训练集，迭代提升召回。

**核心思路**：用更强的通用检测器 **YOLOE-11l**（COCO 预训练）作为独立第二检测器，与 person.pt 对同一批关键帧交叉检测。两者分歧即"难例"——分歧样本保留，标注采用更可信的 YOLOE 输出。

**为什么是 YOLOE 而非 VLM（豆包）**：监控 mkv 关键帧量级达数万张。按帧调 VLM API 在成本（token 费）与时间（数万次网络往返，小时级）上不可行。YOLOE 本地 GPU 推理零 API 成本、可批量流式，整体十几分钟完成；且 11l 容量大于 26n，对小目标/遮挡覆盖更强，天然适合当裁判。

## 2. 已确认决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 第二检测器 | YOLOE-11l（本地 GPU） | 量大，VLM API 不可行；11l 通用强模型补 26n 弱点 |
| 第二检测器类别来源 | YOLOE `set_classes(["person"], get_text_pe)` 开放词汇 | YOLOE 为 prompt-based 模型，不 set_classes 不输出目标（实测默认 COCO 类无效）；set_classes 后稳定检出。参考 `d2d_yoloe.py`、`rmb_yolo_ground.py`。原设计"默认 COCO class 0"实测不工作，已修正 |
| 抽帧方式 | 编码关键帧 I-frame（ffmpeg 无损全量） | 贴合"解压每一幅 key frame"原意，不做二次抽样 |
| 比对方式 | 框级 IoU 贪心匹配 | "偏离较大"指位置/大小偏差，需框级而非计数 |
| IoU 匹配阈值 | 默认 0.3（`--iou` 可调） | "放宽精度"——轻微偏差不算分歧 |
| 难例标注来源 | YOLOE 全部框 | YOLOE 更可信；person.pt 在难例上已不可信 |
| 误检图处理 | 负样本（图片 + **空 .txt**） | YOLO/ultralytics background image 标准约定，降误检 |
| 一致/空帧 | 丢弃 | 只保留难例 |
| 输出格式 | YOLO `images/`+`labels/` | 直接可训练；`data.yaml` 待并入 `samples/` 时生成 |

## 3. 端到端数据流

```
mkv 目录                              person.pt (YOLO26n, 监控专精)
/var/howell/.../sh-sgcc/n001/video       │ ultralytics YOLO, stream predict
  │ ffmpeg select=eq(pict_type,I)         │
  ▼                                       ▼
候选帧目录 ──────────────────► 双检测(GPU, 逐图两模型) ──► 框级比对(纯函数)
/tmp/person_frames/              │                              │
{video_stem}_{idx:06d}.jpg       │                              ▼
                                 │                     难例分类(决策表 §5)
                                 │                              │
                                 ▼                              ▼
                   (一致/空帧 → 丢弃)          正样本(YOLOE框) / 负样本(空txt)
                                                          │
                                                          ▼
                                       /home/jiang/ws/sgcc/person/dates/2025-07-07/
                                         images/   labels/   mining_report.json
```

两阶段 CLI：抽帧独立（通用工具，可复用）；挖掘 bin 内部一次运行完成双检测 + 比对 + 输出（本地推理快，不做中间 NDJSON 落盘）。

## 4. 组件设计

> **实现注记**：Functional Core（纯函数）实际提取到 lib `src/jxl/det/hardmine.py`（获 mypy strict 类型保障 + 充分单测），`person_mine.py` 只留 Imperative Shell（bin 被 mypy exclude）。spec 的壳/核分离意图不变，物理位置见 plan File Structure。

### Bin 1：`src/jxl/bin/mkv_keyframes.py`（通用抽帧，~80 行）

- **职责**：递归找 mkv → ffmpeg 无损提取所有编码 I 帧 → 扁平 jpg
- **ffmpeg 核心**：`ffmpeg -i <in> -vf "select=eq(pict_type\,I)" -vsync vfr -q:v 2 <out>/%06d.jpg`
- **输出命名**：`{video_stem}_{frame_idx:06d}.jpg`（保留视频内时序，便于排查与去重）
- **接口**：typer CLI，`mkv_keyframes <src_dir> <dst_dir>`
- **错误处理**：ffmpeg 未安装 / 无 mkv → 报错退出（No Silent Degradation）

### Bin 2：`src/jxl/bin/person_mine.py`（核心，~250 行，壳/核分离）

#### Imperative Shell（副作用：GPU/IO，不单测）

| 函数 | 职责 |
|------|------|
| `detect_person(paths, model_path, conf, iou, device) -> dict[str, list[Box]]` | `ultralytics.YOLO(model).predict(stream=True)`，取 `boxes.xyxyn` 归一化 xyxy + conf |
| `detect_yoloe(paths, model_path, conf, iou, device) -> dict[str, list[Box]]` | `ultralytics.YOLOE(model).predict(stream=True)`，筛 COCO class 0(person) |
| `write_yolo_sample(dst_dir, img_path, boxes_or_none) -> None` | 复制图 + 写 txt：`None`→空文件（负样本）；`list[Box]`→YOLO `cx cy w h` 行 |
| `run(...)` | typer CLI 入口，串 shell + core |

#### Functional Core（**纯函数，充分单测**）

| 函数 | 职责 | 复用 |
|------|------|------|
| `xyxy_iou(a, b) -> float` | 两归一化 xyxy 框 IoU | 复用 `rmb_eval_grounding.xyxy_iou` |
| `greedy_match(boxes_a, boxes_b, iou_thr) -> tuple[list, list, list]` | 贪心 IoU 匹配 → `(matched_pairs, unmatched_a, unmatched_b)` | 复用 `rmb_eval_grounding` 匹配逻辑 |
| `classify_sample(person_boxes, yoloe_boxes, iou_thr) -> SampleClass` | 分类（决策表 §5） | 新（核心算法） |
| `to_yolo_label(boxes, img_w, img_h) -> str` | 归一化 xyxy → YOLO `cls cx cy w h` 字符串（clamp 防 w/h 负） | 新 |

**数据模型**：
```
Box = tuple[x1, y1, x2, y2, conf]   # 归一化 xyxy ∈ [0,1]
SampleClass = Literal["drop_empty", "drop_agree", "positive", "negative"]
```

`to_yolo_label` 固定 `cls=0`（person 单类），与 person.pt 训练 `names: {0: person}` 对齐。

## 5. 核心算法——难例分类决策表

每张图先 `greedy_match(person, yoloe, iou_thr)`，再按未配对情况分类。**判据两条**：① YOLOE 有无框（决定正/负/丢）；② 有无未配对框（决定分歧）。

| yoloe 框 | person 框 | 未配对 | 判定 | 输出 |
|----------|-----------|--------|------|------|
| 0 | 0 | — | 空帧 | `drop_empty`（丢弃） |
| 0 | >0 | — | person 全误检 | `negative`（空 .txt） |
| >0 | 0 | — | person 全漏检 | `positive`（YOLOE 全框） |
| >0 | >0 | 双方均 0（全配对） | 完全一致 | `drop_agree`（丢弃） |
| >0 | >0 | 任一方 >0 | 分歧难例 | `positive`（YOLOE 全框） |

**关键约定**：只要 YOLOE 有框，正标注一律用 **YOLOE 全部框**（含与 person 配上的 + 未配上的）。
- 与 person 配上的框：两模型本就一致，用谁的都对；
- 未配上的漏检位置：必须用 YOLOE；
- 统一用 YOLOE 全框最自洽，且保留整图完整标注（训练框更密）。

"YOLOE 全框"而非"仅未配对框"——避免漏掉两模型都检出但位置略有偏差的真人体。

## 6. 错误处理 + 边界

| 情况 | 处理（No Silent Degradation） |
|------|------|
| ffmpeg 未安装 | `mkv_keyframes` 启动即报错退出 |
| 模型文件不存在 | `person_mine` 启动即报错退出 |
| 候选目录无图 / mkv 无 I 帧 | 报错退出（不产空结果） |
| 单张图损坏 | 跳过 + 计入 `report.skipped`，不阻塞整批 |
| 输出目录已存在 | 覆盖（rm 重建），简单可重跑；不做增量（YAGNI） |
| 检测 conf | `--conf` 默认 0.25（person.pt 部署值），两模型共用 |
| GPU OOM | ultralytics stream 模式逐图推理，显存占用恒定，不预期 OOM；若发生则降 `--device cpu` |

## 7. 测试策略

纯函数单测 `tests/det/hardmine_test.py`（Functional Core 在 `src/jxl/det/hardmine.py`），**不依赖模型/GPU**：

- `xyxy_iou`：已知框对验证 IoU 值（含相交/包含/相离）
- `greedy_match`：构造框集，验证 `(matched, unmatched_a, unmatched_b)` 划分
- `classify_sample`：**决策表 5 行全覆盖**，含混合行（`>0,>0,双方>0`→positive）
- `to_yolo_label`：归一化 xyxy → `cls cx cy w h` 字符串正确性（含 clamp 防 x1>x2 导致 w/h 负）

集成验证（手动一次，不进 CI）：小样本 mkv 跑通全链，检查输出 `images/labels/` 结构 + `mining_report.json` 统计。

## 8. 输出结构 + CLI

```
/home/jiang/ws/sgcc/person/dates/2025-07-07/
  images/  *.jpg              # 仅难例（正+负样本）
  labels/  *.txt              # 正样本=YOLO 框行; 负样本=空文件
  mining_report.json          # 统计见下
```

`mining_report.json`：
```json
{
  "total_frames": 0, "skipped": 0,
  "positive": 0, "negative": 0,
  "dropped_empty": 0, "dropped_agree": 0,
  "by_video": {"videoA": {...}}
}
```

```bash
# 1. 抽帧（I-frame）
mkv_keyframes /var/howell/iap/current/ias/sh-sgcc/n001/video /tmp/person_frames

# 2. 双检测 + 比对 + 输出
person_mine /tmp/person_frames /home/jiang/ws/sgcc/person/dates/2025-07-07 \
  --person-model /opt/howell/iap/current/ias/model/person.pt \
  --yoloe-model  /home/jiang/py/jxl/models/yoloe-11l-seg.pt \
  --iou 0.3 --conf 0.25 --device cuda:0
```

## 9. 复用与依赖

**复用**（单一数据源，不重写）：
- `src/jxl/bin/rmb_eval_grounding.py`：`xyxy_iou` + 贪心匹配逻辑（已验证）
- `src/jxl/bin/rmb_yolo_ground.py`：ultralytics `YOLOE.predict(list)` 批量/流式推理模式
- `src/jxl/bin/rmb_ground.py`：归一化坐标处理、 Detection 数据模型思路

**依赖**（均已在本项目 `.venv`）：
- `ultralytics`（YOLOE + YOLO 推理）
- `typer` / `pydantic`（CLI + 数据模型）
- `Pillow`（图片复制；或 `shutil`）
- 系统级：`ffmpeg`（抽帧）

## 10. 风险与待确认

| 项 | 说明 | 缓解 |
|----|------|------|
| 数据源可达性 | `/var/howell/iap/current/ias/sh-sgcc/n001/video` 当前 shell 下未匹配到 mkv（权限/挂载/路径） | 实跑前确认实际录像路径 |
| YOLOE 误差污染 | YOLOE-11l 自身也有漏检/误检，分歧未必全是 person.pt 的错 | 首跑后人工抽检 positive 样本；负样本尤其要抽检（person.pt 误检 vs YOLOE 漏检难分辨） |
| IoU 0.3 过宽 | 阈值太低会把轻微偏差误判为一致（漏抓难例），或反过来 | 看 `mining_report.json` 的 `dropped_agree` 占比，按需调 `--iou` 重跑（比对是纯函数，可单独重跑） |
| 负样本比例 | 监控空帧多，person.pt 在空帧上的误检可能产大量负样本 | report 监控 negative 占比，必要时设上限或后置人工筛 |
