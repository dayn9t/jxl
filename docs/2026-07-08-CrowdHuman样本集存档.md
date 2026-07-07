# CrowdHuman 样本集下载/筛/并入存档（2026-07-08）

> 补 person.pt 非站立/密集缺口。CrowdHuman 19370 图下载 + crowdhuman_to_yolo 转换 + det_mine 4 模型筛难例 + 并入 samples/。**搁置于 2026-07-08，待训练**。

## 1. 背景

person.pt 训练集（sgcc+MOT17/20+COCO）站立/行行为主，**非站立（坐/躺/蹲/弯腰）+ 密集人群缺口**。CrowdHuman（24k 图/470k person，密集人群含 sitting/crouching/bending/遮挡）补此缺口。

## 2. CrowdHuman 下载

- **源**：HF `sshao0516/CrowdHuman` via **hf-mirror**（`curl` resolve URL，绕 SSL；huggingface-cli/hf_hub_download 失败因 env 未生效）
- **标注**：annotation_train.odgt（80MB/15000 图）+ annotation_val.odgt（23MB/4370 图）
- **图 zip**：CrowdHuman_train01/02/03.zip（**分卷，cat 合并**；train01 原下载截断 1.5GB，重下 2.97GB 完整）+ CrowdHuman_val.zip。共 ~13GB
- **耗时**：~1h（hf-mirror ~2-5MB/s 波动）

## 3. crowdhuman_to_yolo 转换工具

`src/jxl/bin/crowdhuman_to_yolo.py`（commit feat(bin): crowdhuman_to_yolo）：
- 读 odgt（每行 JSON: ID + gtboxes[].{tag, fbox, head_attr}）
- 筛 tag=person + ignore=0 + unsure=0，fbox 归一化(/img_w,/img_h) → YOLO cls 0
- 输出 images/(symlink) + labels/*.txt
- 产出：train_yolo 15000 图/346k 框 + val_yolo 4370 图/101k 框 = **19370 图/447k person 框**

## 4. det_mine 4 模型筛 CrowdHuman 难例

det_mine（person.pt + YOLOE + GroundingDINO + RF-DETR，加权争议分 cascade，见 `docs/2026-07-07-det-mine多模型难例挖掘.md`）对 CrowdHuman 跑：

| 集 | L0 drop | L1 自动 | review (27%) | 耗时 |
|----|---------|---------|--------------|------|
| val (4370) | 496 | 2712 | 1162 | 30min |
| train (15000) | 1692 | 9316 | 3992 | 1.7h |
| **合计** | **2188** | **12028** | **5154** | — |

- **L1 自动 12028**：低争议，多模型共识自动标注（RF-DETR 框），高置信
- **review 5154**：高争议难例（person.pt 检不准的密集/非站立/遮挡），争议分 top 173 / 中位 8.4。`review_all/manifest.jsonl`（target_boxes + validators 多模型框，供 P2/P3）

review 27% 远高于监控 dates 的 12%——验证 CrowdHuman 密集+非站立让 person.pt 检不准更多（正是要补的弱点）。

## 5. 并入 samples/

| 来源 | 数量 | 标注质量 |
|------|------|---------|
| 现有（sgcc+MOT+COCO） | 24975 | 人工/COCO（高）|
| CrowdHuman L1 | 12028 | det_mine 多模型共识（高）|
| dates/2025-07-07 | 5840（4831 pos + 1009 neg）| person_mine YOLOE 单模型（中）|
| **samples/ 合计** | **42843** | — |

## 6. 待续（搁置于 2026-07-08）

- **训练**：`jxl_split samples dataset -r 8 1 1` → `yolo_train`（640×640 基线 验证 CrowdHuman 提升非站立/密集；或 960×544 实验 见下）
- **960×540 分辨率实验**（已调研，见对话）：person.pt(640 best) → fine-tune 960×544（batch 24, close_mosaic 10, 100 epoch）。16:9 完美匹配监控 + 小目标 +1-3% mAP + 推理几乎不变慢
- **review 5154**：P2 在线大模型 grounding（豆包/Qwen-VL 校验高争议）或 P3 人工标注工具增强（jxl_label 多模型框展示，需解决 opencv GUI）
- **det_mine 重筛 dates/2025-07-07**：多模型替 person_mine 单模型，提 dates 标注质量（~30min GDINO）

## 7. 文件/路径

- CrowdHuman 数据：`/home/jiang/ws/sgcc/person/CrowdHuman/`（train_yolo/val_yolo/review_all/train_detmine/val_detmine/train_images/val_images + odgt）
- samples：`/home/jiang/ws/sgcc/person/samples/`（42843，images/+labels/）
- 工具：`src/jxl/bin/crowdhuman_to_yolo.py`（odgt→YOLO）+ `src/jxl/bin/det_mine.py`（多模型 cascade）
- det-mine 设计/存档：`docs/superpowers/specs/2026-07-07-det-mine-*.md` + `docs/2026-07-07-det-mine多模型难例挖掘.md`
- person.pt 训练存档：`docs/2026-06-27-person模型训练部署.md`（640×640，mAP50=0.956，待用 42843 重训）

## 8. 网络/下载经验

- CrowdHuman HF：`curl -sL https://hf-mirror.com/datasets/sshao0516/CrowdHuman/resolve/main/<file>` 通
- train01/02/03 是**分卷 zip**（cat 合并；train01 注意截断，验大小 2.97GB）
- huggingface-cli/hf_hub_download 在本机 env 未生效（用系统 python 旧 hub），改 curl hf-mirror
