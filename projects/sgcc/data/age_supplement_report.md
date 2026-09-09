# age 补样执行报告（2026-09-09）

存档 §5 补样方案落地：COCO val2017 图源 + spark VLM 伪标，只补 age 维度。

## 图源与流程

- 图源：COCO val2017 官方（cocodataset.org 直连下载 val2017.zip 815MB + annotations 253MB，
  落 `datasets/coco_val2017/`）。本机既有 `datasets/COCO/`（7,840 张自动标注版）来源不明，未用。
- 候选：instances GT person 框，非 crowd，w,h≥80px → 全量 **3,362**（首批 2,000@≥100px + 补批 1,362）。
- crop：box pad 15% → pad-to-square（原图 context 优先，贴边灰 114），长边封顶 256。
- VLM：spark qwen3.5-35b-a3b-fp8（=192.168.18.182:8000），enable_thinking=False +
  response_format json_object + temperature 0.1 + 并发 32（参数定型）。prompt 仅 age 维度
  （词典 §5 判据）+ 自报 conf。**3,362/3,362 成功（0 解析失败）**。

## 结果

| 档 | 判得 | conf>0.8 | 入库 | 目标 300-500 |
|---|---|---|---|---|
| child | 475 | 448 | **448** | 达标 |
| teen | 228 | 93 | **93** | 缺口 207 |
| adult | 2,378 | — | — | 参照 |
| senior | 160 | — | — | 参照 |
| uncertain | 121 | — | — | — |

- teen 缺口分析：判 teen 的 228 例中仅 41% 过 conf>0.8（child 为 94%）——12-18 边界天然模糊，
  VLM 自报把握低；134 例落在 conf(0.6, 0.8]。val2017 已全量跑尽（≥80px 框无一剩余），属**源上限**。
- 外部视觉抽检（12 例 child/teen 各 6）：11/12 可信；1 例为 GT 框锚定婴儿、pad 后 crop 含成人
  （按框语义仍 child，可接受）。

## 产物

- `age_supplement/{child,teen}/`：psq crop 448 + 93
- `age_supplement/labels_selected.jsonl`：541 行（bank 兼容 schema，provenance=COCO_val2017）
- `age_supplement/labels.jsonl`：全量 3,362 判定（含 adult/senior，审计用）
- `cls_age_psq/{train,val,test}/{child,teen,adult,senior}/`：四档年龄分类数据集
  （属性库 labels_final 非 uncertain 行 + 本补样，stem hash md5%10 → 8:1:1）

| 类 | train | val | test | 合计 |
|---|---|---|---|---|
| child | 362 | 53 | 37 | 452 |
| teen | 90 | 10 | 16 | 116 |
| adult | 10,197 | 1,336 | 1,265 | 12,798 |
| senior | 2,252 | 280 | 273 | 2,805 |

（adult 差 1 行 = labels_final 中 1 条 adult uncertain=True，按词典规则不入训练。）

## teen 补齐选项（待裁决，本轮未执行）

1. train2017 抽批 ~8,000 crop（+~200 teen@0.8，估算 19GB 下载 + 40 分钟）
2. doubao 二审 134 例 conf(0.6,0.8] 边界 teen（改判据为双引擎一致，需用户确认）
3. 接受 93 例（与窗口域 23 例合计 116，class weight 训练）
