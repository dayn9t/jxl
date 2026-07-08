# sgcc 多模型重标注存档（2026-07-08）

> sgcc（含 video-extract 合并 20124 图）从 YOLOE 单模型标注 → det_mine 多模型共识 + 豆包 vision grounding（P2 简版）。**已完成 + 审核纠错**。

## 1. 背景

sgcc（14284 监控）+ video-extract（5840 视频提取）当前 YOLOE 单模型标注，质量待提升。重标注用 det_mine 多模型共识（YOLOE+GDINO+RF-DETR）+ review 豆包 vision grounding 复检。

## 2. 流程（Task 1-5）

1. **合并**：video-extract 图移 sgcc/（14284→20124），旧 YOLOE 标注 backup（sgcc-yoloe-bak/）
2. **det_mine 4 模型跑 sgcc**：L0 14669（73% 全一致）/ L1 3819（19% 多模型共识）/ review 1636（8% 高争议）
3. **doubao_relabel bin**（P2）：review 1636 图豆包 vision grounding → YOLO labels + _errors.jsonl
4. **合并标注**：L0 旧 YOLOE（多模型一致背书）+ L1 多模型共识 + review 豆包 grounding → sgcc 20124 全标
5. **PIL 抽检验证**：新（红 多模型/豆包）vs 旧（绿 YOLOE）对比

## 3. 结果

- **sgcc labels 20124 全标**（替旧 YOLOE）：L0 旧YOLOE多模型背书 14669 + L1 多模型共识 3819 + review 豆包grounding 1636
- doubao_relabel review 1636：**1604 有框 + 32 空(无人) + 0 错误**（豆包 grounding 高质量）
- det_mine sgcc：L0 73% / L1 19% / review 8%（sgcc 是 person.pt 训练数据，模型一致高，分歧少）
- sgcc-yoloe-bak/ 保留旧标注（对比/回退）

## 4. doubao_relabel bin（新）

`src/jxl/bin/doubao_relabel.py`（P2 豆包 grounding review → YOLO labels）：
- 复用 `rmb_ground` 的 `load_backend`/`parse_detections`/`encode_image`（单一数据源）
- key 从 `--cfg` json 读（**绝不硬编码**，secret 安全）
- 异步并发（asyncio + semaphore ~6）
- 错误图落 `_errors.jsonl`（可追溯/重试）

## 5. 审核纠错（code-reviewer agent）

- **B1 类型标注**（Detection/list[Detection]，mypy strict）✓ 修
- **S1 encode_image 去重复**（import from rmb_ground，单一数据源）✓ 修
- **S3 _errors.jsonl**（错误图落盘，可追溯）✓ 修
- **N1-N3**（noqa 清理/manifest 验证/输入校验）✓ 修
- S2 ground_one 参数化（与 rmb_ground 重复，TODO 后续提取共用）

## 6. 关联

- spec/plan: `docs/superpowers/{specs,plans}/2026-07-08-sgcc-relabel-*`
- det_mine（4 模型 cascade）、rmb_ground（豆包 grounding 范本）—— 复用
- det-mine 存档 §11 P2 → 本存档实现 P2 简版（sgcc review 复检）
- datasets pool（`docs/2026-07-08-datasets-pool-design.md`）—— sgcc 是其中一集
