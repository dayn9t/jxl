# sgcc 多模型重标注 + P2 豆包 grounding 设计（2026-07-08）

> sgcc + video-extract 合并（20124 图），det_mine 多模型共识重标注（L1）+ review 豆包 vision grounding（P2）复检，替旧 YOLOE 单模型标注。提升 sgcc 标注质量（多模型共识 + VLM 裁判 > YOLOE 单模型）。

## 1. 背景与目标

`datasets/sgcc/`（14284 监控）+ `datasets/video-extract/`（5840 视频提取）当前 **YOLOE 单模型标注**（sgcc 原标注 + video-extract person_mine 产出）。单模型标注质量有限（漏检/误检/框偏）。

**目标**：用 det_mine 多模型共识（YOLOE+GDINO+RF-DETR）+ review 豆包 vision grounding 复检，重标注 sgcc，替旧 YOLOE。

## 2. 已确认决策

| 决策 | 选择 |
|------|------|
| 合并 | video-extract 图移 sgcc/（20124），旧 YOLOE 标注 backup（sgcc-yoloe-bak/）|
| L1 标注来源 | det_mine L1 = 校验器（YOLOE+GDINO+RF-DETR）多模型共识（RF-DETR 优先框），person.pt 仅分流不进标注 → 无 person.pt 循环 |
| review 复检 | P2 豆包 vision grounding（doubao-seed-2-0-lite，复用 rmb_ground 范本）|
| scope | P1（L1）+ P2（豆包 grounding）一个 spec |

## 3. 总流程

```
video-extract(5840) ──合并──► sgcc(20124)
                                │
            det_mine 4 模型(person.pt + YOLOE + GDINO + RF-DETR) 跑 sgcc
                                │
                ┌───────────────┴───────────────┐
                │ L1(低争议 ~73%, 多模型共识)     │ → 高质量标注(RF-DETR 优先框)
                │ review(高争议 ~27%, ~5400)    │ → P2 豆包 grounding 复检
                └───────────────┬───────────────┘
                                │
                review → 豆包 vision grounding(P2) → bbox → 合并 L1
                                │
                sgcc/labels = L1 + review-doubao（替旧 YOLOE）
```

## 4. video-extract 合并 sgcc

- video-extract 图（5840）→ `sgcc/images/`（sgcc 扩到 20124）
- 旧 YOLOE 标注 → `sgcc-yoloe-bak/`（images+labels 备份，对比/回退）
- video-extract 集从 datasets/ 移除（合并入 sgcc）
- experiments/ toml 中 `video-extract` 引用移除（已并入 sgcc）

## 5. det_mine L1 重标注（复用现有 det_mine）

- `det_mine sgcc/ <out> --target person --target-model person.pt --validators yoloe,gdino,rfdetr --device cuda:0`
- 产出：L1（~73% 多模型共识，RF-DETR 优先框）+ review（~27% 高争议 ~5400）+ mining_report
- L1 标注替旧 YOLOE（多模型共识 > 单模型）
- review 集（manifest + 图）待 P2

## 6. P2 豆包 grounding review 复检（新）

### detect_doubao backend（复用 rmb_ground 范本）
- 复用 `src/jxl/bin/rmb_ground.py` 的 `load_backend`（豆包配置读取）+ `ground_one`（豆包 grounding：text prompt "person" → bbox）+ `parse_detections`（VLM 输出解析归一化）
- key 从配置文件读（rmb_ground 模式，**绝不硬编码**——doubao key 是 secret，走 llm.json 或环境变量）
- 异步并发（asyncio + semaphore ~6，参考 rmb_ground）

### review 复检流程
- review 集(~5400) → 逐图豆包 grounding（text "person"）→ bbox（归一化）
- **量小（5400 子集）→ 成本可控**（之前评估 VLM 全量 20k 不可行，review 5400 子集可行）
- 豆包 grounding 强 → review 高争议提质量（det_mine 拿不准的，VLM 裁判）
- 结果合并 L1：review 图获得豆包 grounding 标注

### 新 bin: `src/jxl/bin/doubao_relabel.py`
- 读 det_mine review/manifest.jsonl（review 图列表）
- 对每图豆包 grounding → bbox
- 输出 review 图的新 YOLO labels（cls=0）

## 7. 产出与验证

- `sgcc/labels/` = L1 + review-doubao-grounding 合并（替旧 YOLOE）
- `sgcc-yoloe-bak/` 保留旧标注（对比/回退）
- **验证**：PIL 网格抽检（红=新标注框 vs 旧 YOLOE，看是否更准；jxl_viewer opencv GUI 不可用，用 PIL 预览）

## 8. 范围与依赖

- **复用**：det_mine（4 模型 cascade）、rmb_ground（豆包 grounding 范本：load_backend/ground_one/parse_detections）
- **新增**：`doubao_relabel.py` bin（review 豆包 grounding）+ detect_doubao 逻辑（复用 rmb_ground）
- **配置**：豆包 key（llm.json 或环境变量，rmb_ground 模式）
- det-mine 存档 §11 的 P2 独立 spec → 本 spec 实现 P2 简版（sgcc review 复检）

## 9. 关联

- det_mine: `src/jxl/bin/det_mine.py`（多模型 cascade）
- rmb_ground: `src/jxl/bin/rmb_ground.py`（豆包 grounding 范本）
- det-mine 存档: `docs/2026-07-07-det-mine多模型难例挖掘.md`（§11 P2）
- datasets pool: `docs/superpowers/specs/2026-07-08-datasets-pool-design.md`
