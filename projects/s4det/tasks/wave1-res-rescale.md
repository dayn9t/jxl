# 存量样本 960×540 重制执行记录（2026-09-27）

> 任务：存量样本全量重制为检测专用 960×540 低分辨率（用户裁决 2026-09-27，授权「全部立即重制」）。
> 架构语义：原图 1920×1080 永不动（分类器 crop/VLM 细判继续用高分辨率）；重制仅「样本/数据集层」
> images：symlink 原图 → 960×540 实体 jpg（LANCZOS，quality=88，与 T4-b1 已重制版同参）。
> 执行会话：jxl wave1 并行 agent（本文件即执行记录正本——C-execution.md 并发冲突不入）。

## 1. 范围与逐项数量

| 位置（绝对路径） | 帧数 | 处理 | 标签处理 |
|---|---|---|---|
| `/mnt/data/s4det/human_window/window1/images/` | 522 | symlink→960×540 实体 | `labels/*.json` 坐标 ×0.5，imageWidth/Height→960/540（522/522 重写） |
| `/mnt/data/s4det/p3-seedset/batch2/mining/dataset/all/images/` | 1,268 | 同上 | YOLO 归一化零变化（不动） |
| `/mnt/data/s4det/p3-seedset/batch1/mining/dataset/all/images/` | 1,481 | 同上 | 同上 |
| `/mnt/data/s4det/p3-seedset/batch1/mining/dataset/{train,val,test}/images/` | 1,184/148/149 | **零操作**（全量核验 symlink 100% 指向 `all/images/`，自动跟随） | 不动 |
| `/mnt/data/s4det/p3-seedset/negpool/v1/images/` | 250 | 同上 | 空标签不动 |
| 合计重制 | **3,521** | | |

不动项（如实）：smoke-v0、m0-fire、批 1/2 挖掘链中间产物（consensus/review 原图 symlink 树原样）、
seedset pools（negpool 的 symlink 源头 `batch2/seedset/images/` 保持 1920×1080 原图）、
quarantine、weights、eval-gt/t4-b1、eval-gt/t4-b2；window1 的 labels.txt/README.md/manifest.jsonl
（src 指原图 provenance 保留，恰好符合原图可寻址架构）；classes.txt/data.yaml；
batch1 all/dedup_report_labels.txt；train/labels.cache（只缓存标签校验和，不含图像内容，标签未变即有效）。

## 2. 执行前核查

- **window1 已编辑帧 mtime 复核**：522 个 json mtime 全部 = `2026-09-26 11:18`（预标注生成同一时刻，
  零离群）→ 确认用户 0/522 未开始编辑，直接重写无损失，无需 labels-edited-backup/。
- 批 1 切分树指向核验：train 1,184 / val 148 / test 149 全部 symlink → `all/images/`（non-all-targets=0，broken=0）。
- 源尺寸抽查（后升级为全量普查，见 §4）：1920×1080。
- labels md5 重制前存证：batch1 1,481 txt / batch2 1,268 txt（快照 + 全列表 md5 落
  `/tmp/res-rescale-evidence/`，聚合指纹 batch1 `68e44663c84dd080435ffdb4c576f0cf`、
  batch2 `7e81505563a1b5e01c4c5ddaf5553ba4`）。
- window1 labels 重写前全量快照：`/tmp/res-rescale-evidence/window1-labels-orig/`（522 json，用于逐点 0.5 倍对拍）。

## 3. 执行方式

一次性迁移脚本（未入仓，正本路径）：`/tmp/res-rescale-evidence/`
（`rescale_images.py` + `rewrite_window1_labels.py` + `verify.py` + `run-all.log`/`run-rest.log`）。
注意 /tmp 易失——关键参数与证据数字已全部内化到本文件，脚本丢失可按本节复现：

- 图像：`Image.resize(dst, LANCZOS)`，`save(format='JPEG', quality=88)`；临时文件同目录写完
  `os.replace` 原子换名，unlink symlink 在后——单帧原子，中断可幂等续跑（已实体且尺寸合规者 skip）。
- labels：json 解析改字段再落盘（非 sed）；每 shape `points=[[x*0.5,y*0.5]]`，其余键（label/description/
  flags/imagePath）语义原样；`ensure_ascii=False`，单行紧凑（与 X-AnyLabeling 原格式一致）。
- Python：`/home/jiang/cc/py/jxl/.venv/bin/python`（PIL 12.0.0；带宽周日零触网）。

## 4. 矛盾上报与处置：3 帧源图非 1920×1080（决策质疑原则）

**数据与「原图全部 1920×1080」前提矛盾**——全量普查（每 symlink 逐一读 header）：

- 3,521 帧中 **3 帧源为 1920×1104**，同一 howell 源事件（`2026-07-08_6a4de4a8c23555dfba659b89` 与
  `...a9c23555dfba659b8b` 的 r3；前者同时进了批 1 与批 2 数据集）：
  - `batch2/all/images/2026-07-08_6a4de4a8c23555dfba659b89_r3.jpg`
  - `batch2/all/images/2026-07-08_6a4de4a9c23555dfba659b8b_r3.jpg`
  - `batch1/all/images/2026-07-08_6a4de4a8c23555dfba659b89_r3.jpg`
- 其余 3,518 帧全部恰好 1920×1080；window1/negpool 全净。

**处置 = 均匀 ÷2 → 960×552（0.085%）**，理由：唯一不破坏标签正确性的方案（YOLO 归一化坐标尺度不变、
几何完全保持；拉伸 960×540 会作废 y 归一化，裁 24 行会造成几何偏移需标签手术且可能裁掉框）；
训练侧 YOLO letterbox 本就吃非 16:9（这 3 帧此前即以 1104 高在喂训练，语义零变化）。
**留用户改判余地**：若裁决要求严格 960×540，可对 3 帧做「裁 1080 + 标签 y 手术」重做（小手术，随时可补）。

## 5. 验收证据（独立 verify.py 全量跑，VERIFY-ALL-PASS）

| 验收项 | 结果 |
|---|---|
| 四处 images 计数不变 | 522 / 1,268 / 1,481 / 250，全部精确一致 |
| `find -type l` 剩余 symlink | 四处均 = **0**；批 1 切分树按设计保留 symlink → all/（train 1,184/val 148/test 149，抽样解析全部 960×540 合规） |
| 图像尺寸 | **全量**（非抽查）逐文件核：3,518 帧 = 960×540 JPEG，3 异常帧 = 960×552 JPEG（名义清单核对） |
| window1 json | **522/522 全量**对拍重写前快照：逐点恰为旧值 ×0.5（浮点精确相等）、label/description/imagePath 不变、imageWidth=960/imageHeight=540；全域 max point = (960.00, 540.00) 恰达边界不越界 |
| 批 1/2 YOLO labels | md5 重制前后 **byte-identical**（diff 为空；聚合指纹见 §2） |
| 不动项 | labels.txt/README.md/manifest.jsonl/classes.txt/data.yaml/dedup_report mtime 均为迁移前；`.tmp*` 残留 = 0 |
| 原图永不动 | 抽验 consensus/review、/var/howell 源、seedset pool 仍 1920×1080/1104 原样 |
| 磁盘 | 四 images 目录合计约 602MB（92+258+212+40），盘余 1.1T |

## 6. 遗留与未覆盖项

1. **3 帧 960×552 待用户改判**（§4）：默认接受即结案；要求严格 540 则补裁剪+标签手术。
2. `/tmp/res-rescale-evidence/` 易失：本记录已内化全部证据数字；若需长期留存对拍快照可另行归档
   （window1-labels-orig 522 json + labels md5 列表，共约 4MB）。
3. **未覆盖**（按任务书不碰）：smoke-v0、m0-fire、挖掘链中间产物、seedset pools、quarantine、weights、
   eval-gt/t4-b1（另一 agent 在途）、eval-gt/t4-b2（同 agent 领地）——若后续统一改制需另行任务。
4. train/labels.cache 未清未动：只含标签校验和（标签未变，缓存仍有效）；不含图像几何。
5. 下次训练读图即 960×540——若外层脚本有硬编码 1920/1080 假设（如按像素尺寸算 crop）需随用随改，
   本次未发现消费方（数据集仅 YOLO 归一化消费）。

## 7. 时间线

- 2026-09-27 下午：预检（mtime/md5 存证/切分树核验）→ window1 522 转换 → batch2 转至 131 帧 fail-fast
  停（发现 1920×1104 矛盾）→ 全量尺寸普查定位 3 帧 → window1 labels 522 重写恢复一致性 →
  改脚本（均匀 ÷2 尺寸表）续跑 batch2 余 1,137 + batch1 1,481 + negpool 250 → 独立全量验收 PASS
  → 数据根 README 操作日志追加（/mnt/data/s4det/README.md）→ 本记录。
