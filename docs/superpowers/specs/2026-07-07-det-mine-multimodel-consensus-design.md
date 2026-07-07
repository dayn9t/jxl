# Det-Mine 多模型共识 + 争议分 cascade 设计（2026-07-07）

> 把 person 难例挖掘从"二元共识（person.pt vs YOLOE 对/错）"升级为"N 模型加权争议分 + cascade 分级处理"，最小化人工。**本期 P1**（打分 + 本地 cascade），P2（在线 grounding）/ P3（人工工具）在 §11 TODO 记录。

## 1. 背景与目标

现有 `person_mine`（commit 545b013→a429fe2）用 person.pt + YOLOE 双检测器二元比对，分歧即难例。三个问题：
- **二元粗糙**：无法区分"3 模型全员分歧"vs"1 模型边缘分歧"
- **单模型污染**：YOLOE 误差直接进标注（上版 spec §10 风险）
- **无优先级**：人工复核无的放矢（5840 样本不知先看哪个）

本设计升级为：
- **N 模型加权争议分**（连续，非二元）：综合多模型一致性 + 各模型可信度权重
- **cascade 分级**：高一致→自动接受；中等→本地共识自动标注；高争议→review 候选集（待 P2/P3）
- **最小化人工**：级联逐级过滤，到人工只剩"模型也拿不准"的少量样本

**目标类别**：当前 person，未来 hand/phone（B+ 类别参数化，零改代码扩展）。

## 2. 已确认决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 范围 | P1 打分+本地 cascade；P2/P3 TODO | P1 独立可用，P2/P3 各自后续 spec |
| 校验器 | YOLOE + Grounding DINO + RF-DETR | 跨架构（CNN / transformer / 开放词汇）错例不重叠 |
| 共识阈值 K | 2（3 校验器 ≥2 认同） | 多数共识，过滤单模型噪声 |
| 评分方式 | **连续争议分**（非二元） | 区分争议程度，支持 cascade + 排序 |
| 模型权重 | RF-DETR:0.4 / GDINO:0.35 / YOLOE:0.25（可配） | 按 COCO AP 量级 |
| 自动标注框来源 | 共识位置 RF-DETR>GDINO>YOLOE 回退 | 最强模型定位最准 |
| 架构 | B+ 类别参数化（`det_mine`，函数式 backend） | 多类零改代码；不上 Protocol（YAGNI） |
| review 阈值 | top 30%（可配 `--review-top`） | 按分数分布自适应 |

## 3. 数据流

```
frames ─► person.pt(target) ──────────────────► target_boxes
       ├─► YOLOE  (set_classes target) ──┐
       ├─► GroundingDINO (text target)  ├─► validators_boxes (3 组)
       └─► RF-DETR (class target)      ──┘
                         │
      hardmine.score_sample(target, validators, weights, iou=0.3, k=2)
                         │
      (dispute_score: float, consensus_boxes: list[Box], breakdown)
                         │
   ┌─────────────────────┼───────────────────────────┐
   │ score==0 全一致       → L0 auto-drop              │
   │ 0<score≤t1 共识清晰    → L1 auto-label(共识框)      │
   │ score>t1 争议大        → L2/L3 review 候选集(排序)  │
   └─────────────────────┴───────────────────────────┘
                         ▼
   <out>/ images/+labels/(L1) + review/(L2/L3 候选) + mining_report.json
```

## 4. 组件

| 文件 | 变化 | 职责 |
|------|------|------|
| `src/jxl/det/hardmine.py` | **新增** `score_sample` + `find_consensus_positions` + `pick_by_priority` | 争议分纯函数（mypy strict + 充分单测）|
| `src/jxl/bin/det_mine.py` | **rename** 自 `person_mine.py` + 加 `detect_gdino`/`detect_rfdetr` + 类别参数化 + cascade 分流 | Imperative Shell，`--validators` 函数表调度 |
| `tests/det/hardmine_test.py` | 扩展 score_sample / find_consensus_positions 测试 | 决策表 + 权重 + 回退 + K 边界 |

## 5. 争议分算法（`hardmine.score_sample`，核心）

模型可信度权重（按 COCO AP 量级，可配 `--validator-weights rfdetr:0.4,gdino:0.35,yoloe:0.25`）：

```
对每图:
  # 误检分: target 框中认同<k 的(person.pt 疑似误报)
  fp = Σ over target框(认同票<k):
         (W_total − W_认同) / W_total      # 强模型都不认 → 越可能真误检 → 分越高

  # 漏检分: 校验器共识位置(≥k 认同)且 target 漏的
  fn = Σ over 共识漏检位置:
         W_认同 / W_total                  # 强模型都认 → 越可能真漏检(高价值难例) → 分越高

  dispute_score = fp + fn                  # 连续 ≥0; 0=全一致; 越大越争议
  consensus_boxes = 共识位置按 RF-DETR>GDINO>YOLOE 回退选框(L1 自动标注用)
```

返回 `(dispute_score, consensus_boxes, breakdown)`，其中 `breakdown` 记录 fp/fn 细节供 report。

**`find_consensus_positions(validators, iou, k)`**：把所有校验器框两两 IoU 匹配，≥k 个重叠的聚成一组（类似反向 NMS——找重叠簇），每组 = 一个共识位置，记录认同的校验器名 + 各自框。

**`pick_by_priority(position, ["rfdetr","gdino","yoloe"])`**：共识位置选标注框——优先 RF-DETR（COCO 60 AP 定位最准），该位置 RF-DETR 无框则回退 GDINO→YOLOE。

## 6. Cascade 分级（P1 实装 L0/L1；L2/L3 输出候选集 + TODO）

| 级 | 条件 | P1 处理 |
|----|------|---------|
| **L0** auto-drop | `dispute_score == 0` | 丢弃（全模型一致，非难例） |
| **L1** auto-label | `0 < score ≤ t1` | 共识框写 YOLO 标注（RF-DETR 优先回退）→ `images/+labels/` |
| **L2** online grounding | `score > t1` | 输出到 `review/` 候选集（排序）— **TODO P2** |
| **L3** human | top-N | 同 `review/` 候选集 — **TODO P3** |

`t1` 默认：按分数排序后 top 30% 进 review（`--review-top 0.3`），或绝对阈值（`--review-threshold <float>`）。P1 **不精确分 L2/L3 边界**——由 P2/P3 自取 `review/` 的 top。

## 7. 输出结构

```
<out>/
  images/+labels/        # L1 自动标注集(YOLO 格式)
  review/
    *.jpg                # L2/L3 候选图(按争议分降序命名)
    manifest.jsonl       # 每行: {image, score, target_boxes, validators:{yoloe,gdino,rfdetr}, breakdown}
  mining_report.json     # {target, L0, L1, review 计数, 分数分布直方图, by_video}
```

## 8. 类别参数化（B+，多类零改代码）

```bash
# 现在: person
det_mine <frames> <out> --target person --target-model /opt/.../person.pt \
    --validators yoloe,gdino,rfdetr --consensus 2 \
    --validator-weights rfdetr:0.4,gdino:0.35,yoloe:0.25 \
    --review-top 0.3 --device cuda:0

# 未来: phone(零改代码)
det_mine <frames> <out> --target phone --target-model ./phone.pt --cls-id 0 \
    --validators yoloe,gdino,rfdetr
```

- `--target` → GDINO prompt + YOLOE `set_classes` + report 类别名
- `--target-model` → 被校验专用权重（默认 person.pt）
- `--cls-id` → `to_yolo_label` 类 id（默认 0）
- `--validators` → 校验器子集（默认 yoloe,gdino,rfdetr）
- `--validator-weights` → 各校验器可信度权重
- `--review-top` / `--review-threshold` → cascade 分流

## 9. 依赖 + 错误处理

- `pyproject.toml` 加 `transformers`（GDINO，`IDEA-Research/grounding-dino-*`）、`rfdetr`（RF-DETR，`pip install rfdetr`）
- GDINO/RF-DETR 权重首次运行下载（HF / Roboflow，需外网——sgcc0 用 proxychains）
- `--validators` 选的校验器对应库未装 → 报错退出（No Silent Degradation）
- `--consensus K` > 校验器数 → 报错（逻辑不可能）
- 权重和不归一 → 警告（不阻断，归一化内部处理）

## 10. 测试

`hardmine_test.py` 扩展（纯函数，不依赖模型）：
- `score_sample`：全一致→0；误检（强模型不认 target 框）fp 高；漏检（强模型共识 target 漏）fn 高；权重归一化正确；回退选框（RF-DETR 缺→GDINO）
- `find_consensus_positions`：3 框重叠聚类；分散不聚；k 阈值截断；跨校验器匹配
- `pick_by_priority`：优先级顺序；缺失回退

`det_mine` backends（imperative，import 冒烟 + 权重下载后小样本集成）：GDINO/RF-DETR 各跑 1 张，验证返回 `{stem: [Box]}` 结构。

## 11. TODO: 后续 phase（防遗忘，各自独立 spec）

### P2 — 在线大模型 grounding（L2 stage）
- **消费**：`<out>/review/manifest.jsonl`（P1 产出的高争议候选集，量已小）
- **做**：调在线 VLM grounding API（豆包 vision / Qwen-VL-Max / GPT-4V）校验每个候选，API 返回 bbox + 类别
- **回填**：结果写回 `review/manifest.jsonl`（加 `online_grounding` 字段），作为第 4 个校验器视角
- **价值**：在线大模型能力强，量小成本可控（review 集已是 top 30% 子集）
- **依赖**：P1 的 review/manifest 格式；VLM API key 配置
- **独立 spec**：P1 验证后 brainstorm

### P3 — 人工标注工具增强（L3 stage）
- **消费**：`<out>/review/manifest.jsonl`（含 P1 多模型框 + P2 在线 grounding，争议最高的 top-N）
- **做**：`jxl_label`（现有人工标注 GUI）增强——同图展示 N 个模型框（颜色区分 target/validators/online）+ 争议分 + 人选/改/确认
- **前置**：解决 `jxl_viewer` 暴露的 opencv GUI 后端问题（`cv2.namedWindow` 未实现，需重装带 GTK 的 opencv 或换 Tk/Qt 后端）
- **价值**：人工只处理"模型也拿不准"的少量样本，且所有模型视角都呈现，判断高效
- **独立 spec**：P2 后 brainstorm

## 12. 关联

- 上版设计：`docs/superpowers/specs/2026-07-07-person-hard-sample-mining-design.md`（二元 consensus，本 spec 升级它）
- 上版实现：`person_mine.py`（commit a429fe2）→ 本 spec rename `det_mine.py` + 升级
- person.pt 训练存档：`docs/2026-06-27-person模型训练部署.md`
- 数据源验证：`/home/jiang/ws/sgcc/person/dates/2025-07-07/`（5840 样本，本 spec 用 det_mine 重跑精炼）
