# M1 v0.1 重训前管线冒烟（wave1-smoke，2026-09-27）

> 性质：干线彩排——用现有数据（batch1 + batch2，不含等人工窗口回收部分）走通
> 「合并→切分→训练→评估」四段，暴露 data.yaml/类别数/训练入口问题，让 M1 真跑零意外。
> 执行环境：本机（RTX 4060 Ti 16G，共卡 jas 业务进程 ~4.4G）。
> Python 一律 `/home/jiang/cc/py/jxl/.venv/bin/python`（带宽周 2026-09-21~27 禁 uv sync/uvx/pip）。
> 记录文件即本文件；C-execution.md 与 projects/s4det/README.md 未动（任务纪律）。

## 结论

**四段全通**：合并 2,679 帧/5,011 框 → 8:1:1 切分（2,143/268/268）→ yolo26n 4 epochs 训练
（loss 单调下降、无 NaN、GPU 4.8-5.1G）→ 独立 val 跑通（含每类明细）。
18 类加载无误（Detect head `[18, 1, True, ...]`，COCO 底座迁移 606/708 项正常）。

暴露问题 5 项（详见「坑清单」）：批间 stem 冲突 70 帧、plastic_loose 全集零框空类、
6 类 val 零实例、`--classes` 传参方式、ultralytics val 落点跟 cwd。

## 数据对账

| 源 | 帧 | 框 | 说明 |
|---|---|---|---|
| batch1 all（/mnt/data/s4det/p3-seedset/batch1/mining/dataset/all） | 1,481 | 1,403（D0 记录） | 2 类（ebike/garbage_bag），706 空标负样本帧 |
| batch2 all（/mnt/data/s4det/p3-seedset/batch2/mining/dataset/all） | 1,268 | 3,683 | 18 类，与批 2 全链记录吻合 |
| 合并后（/mnt/data/s4det/smoke-v0/merged/all） | **2,679** | **5,011** | = batch1 去重叠 1,411 帧/1,328 框 + batch2 全量 |

合并后空标负样本帧 689（batch1 706 空帧中 17 帧在重叠集内被 batch2 版替换）。
切分：train 2,143 / val 268 / test 268（jxl_split 默认 8:1:1）。
val 扫描：268 images, 66 backgrounds, 0 corrupt——空标负样本被正确识别为背景帧。

合并后各类框数（cls id: count）：
`0:1957 1:412 2:447 3:965 4:648 5:77 6:146 7:2 8:41 9:3 10:0 11:211 12:29 13:15 14:3 15:5 16:47 17:3`
—— **cls 10 plastic_loose 全集零框（空类）**；弱类（<10 框）：foam_box 2、leaves_pile 3、
scaffolding 3、vendor_booth 3、shop_display 5。

## 命令序列（M1 直接复用）

```bash
PY=/home/jiang/cc/py/jxl/.venv/bin/python
B1=/mnt/data/s4det/p3-seedset/batch1/mining/dataset/all
B2=/mnt/data/s4det/p3-seedset/batch2/mining/dataset/all
OUT=/mnt/data/s4det/smoke-v0          # M1 换正式目录（见差异清单#0）

# 0) batch1 去重叠层（M1 若批间重叠仍在，同法；symlink 层不动正本）
$PY - <<'EOF'
from pathlib import Path
r1 = Path("/mnt/data/s4det/p3-seedset/batch1/mining/dataset/all")
r2 = Path("/mnt/data/s4det/p3-seedset/batch2/mining/dataset/all")
out = Path("/mnt/data/s4det/smoke-v0/b1-labels-dedup")
out.mkdir(parents=True, exist_ok=True)
b2_stems = {p.stem for p in (r2 / "labels").glob("*.txt")}
kept = skipped = 0
for lbl in sorted((r1 / "labels").glob("*.txt")):
    if lbl.stem in b2_stems:
        skipped += 1
        continue
    (out / lbl.name).symlink_to(lbl.resolve())
    kept += 1
print(f"kept={kept} skipped={skipped}")
EOF

# 1) 合并 → 平铺集 + classes.txt + data.yaml（18 类按 cls id 固定有序传参）
$PY -m vlabel.bin.consensus_dataset $OUT/merged \
  --yolo $OUT/b1-labels-dedup:$B1/images \
  --yolo $B2/labels:$B2/images \
  --classes ebike --classes garbage_bag --classes bicycle --classes car \
  --classes shared_bike --classes tricycle --classes carton --classes foam_box \
  --classes garbage_pile --classes leaves_pile --classes plastic_loose \
  --classes tarp_covered --classes construction_fence --classes piled_goods \
  --classes scaffolding --classes shop_display --classes stall_cart \
  --classes vendor_booth

# 2) 切分 8:1:1（默认 radio 8 1 1；-e .jpg 默认）
$PY -m jxl.bin.jxl_split $OUT/merged/all $OUT/merged

# 3) 冒烟训练（setsid 脱离托管；M1: epochs 100 起步 + 规模适配检查）
mkdir -p /mnt/data/s4det/weights/smoke
cd /home/jiang/cc/py/jxl && setsid nohup $PY -m jxl.bin.yolo_train $OUT/merged/data.yaml \
  --model /home/jiang/cc/py/jxl/yolo26n.pt \
  --epochs 4 --batch 32 --device 0 --workers 4 \
  --project /mnt/data/s4det/weights/smoke --name v0-smoke \
  > $OUT/train.log 2>&1 &

# 4) 独立评估（每类明细；显式传 project 防 cwd 落点漂移）
$PY - <<'EOF'
from ultralytics import YOLO
m = YOLO("/mnt/data/s4det/weights/smoke/v0-smoke/weights/best.pt")
r = m.val(data="/mnt/data/s4det/smoke-v0/merged/data.yaml", split="val",
          device=0, batch=32, workers=4,
          project="/mnt/data/s4det/weights/smoke", name="v0-smoke-val")
print(r.box.mp, r.box.mr, r.box.map50, r.box.map)
EOF
```

产物落点：
- 数据集：/mnt/data/s4det/smoke-v0/merged/（all + train/val/test + classes.txt + data.yaml）
- 训练日志：/mnt/data/s4det/smoke-v0/train.log
- 权重：/mnt/data/s4det/weights/smoke/v0-smoke/weights/{best.pt,last.pt}（5.4MB）+ results.csv

## 坑清单（M1 前须处理或知晓）

1. **批间 stem 冲突 70 帧（M1 前须正式裁决）**：batch1/batch2 `all/images` stem 交集 70。
   证据：70 帧图片文件全部同源（symlink 指向同一原图）；标注仅 1 帧文本全同，其余为
   batch2 版 ⊃ batch1 版（坐标微调 + 新增 18 类框）——batch2 是 18 类全量管线的更新版。
   冒烟处理：建 b1-labels-dedup symlink 层剔除 70 帧，batch2 版优先。M1 若沿用此口径，
   重叠帧的 batch1 版标注即被弃置（信息损失极小，因 batch2 为超集）；正式裁决可复核
   这 70 帧的 batch2 新增框质量。
2. **plastic_loose（cls 10）全集零框**：18 类表里有名无实。训练不报错（YOLO 不要求
   每类有样本），但该类永远 mAP 无意义。M1 前须裁决：window1 回收补量 / 挖矿补量 /
   类表暂留空类。
3. **ultralytics 零实例类的 maps 读数陷阱**：`r.box.maps` 对 val 零实例类（本次 6 类：
   foam_box/leaves_pile/plastic_loose/piled_goods/shop_display/vendor_booth）填充**全局
   map 值**（本次 0.1473），不是该类真实成绩。逐类结论只看有实例类；冒烟 val 每类
   明细见下节。
4. **`--classes` 传参方式**：typer list option **必须逐类重复 `--classes <name>`**，
   空格分隔多值报 `Got unexpected extra arguments`。顺序即 cls id 序，argv 顺序确定
   （assoc 数组迭代序教训在此工具不适用，但仍固定有序列表传参不变）。
5. **ultralytics val 默认 project 跟 cwd 走**：本次独立 val 未传 project，结果意外落
   /home/jiang/py/js4/runs/detect/val-12（heredoc 执行 cwd 所致，非本仓）。M1 评估
   一律显式传 `project=/mnt/data/s4det/weights/...`。
6. 次要：训练日志 tqdm 进度条污染 grep——轮询用 results.csv 或 perl 剥 ANSI
   （`perl -pe 's/\e\[[0-9;]*[A-Za-z\r]//g'`）；jxl_split 在 val/test 分组阶段耗时
   ~20s（大量 symlink，正常）；bandwidth 周内 ultralytics 提示有新版（8.4.163）但
   **不可 pip 升级**，当前 8.4.75 工作正常。

## 冒烟数字（管线冒烟值，非质量结论）

> **WARNING（EVAL-BENCHMARKS 纪律）**：以下全部为管线冒烟值——4 epochs 短训、无人工 GT
> 参照（n001 虚高教训），**不得当质量结论、不得写入任何基准文档**。仅证明数据链路
> 与训练入口工作正常。

loss（results.csv，单调无 NaN）：
- train cls_loss 6.913 → 5.544 → 4.913 → 4.517；val cls_loss 6.691 → 5.240 → 4.568 → 4.257
- mAP50 0.029 → 0.101 → 0.174 → 0.199（每 epoch 上升）

best.pt 独立 val（268 帧/494 实例/66 背景）：P=0.630 R=0.213 mAP50=0.199 mAP50-95=0.147

val 每类（mAP50-95，仅有实例类；数字本身无意义仅证链路）：
ebike 0.232 / car 0.279 / garbage_bag 0.176 / bicycle 0.139 / shared_bike 0.085 /
tricycle 0.002 / carton 0.273 / garbage_pile 0.001 / tarp_covered 0.404 /
construction_fence 0.001 / scaffolding 0.000 / stall_cart 0.176

训练资源实况：GPU_mem 4.79-5.05G（batch 32/640/nano，预算 6G 内）；4 epochs 2.9 分钟；
jas 业务进程全程未受影响；训练前后 dmesg 无 NVIDIA Xid/NVRM 异常。

## M1 真跑前差异清单（冒烟 → M1）

1. **目录换正式位**：冒烟产物在 /mnt/data/s4det/smoke-v0/（一次性）；M1 数据集应落
   /mnt/data/s4det/ 正式 seedset 汇总目录，权重落 /mnt/data/s4det/weights/ 主位。
2. **并入 window1 人工窗口回收产物**（等用户 522 帧编辑完成；T4-b1 轨）。
3. **并入负样本池**（M1 数据面规划中的负样本增强）。
4. **重叠 70 帧口径正式裁决**（坑#1）。
5. **plastic_loose 空类与弱类补量**（坑#2 + 弱类 <10 框：foam_box/leaves_pile/
   scaffolding/vendor_booth/shop_display——8:1:1 后 val 常零实例，mAP 噪声）。
6. **epochs 4 → 100 起步**（D0 先例），配 patience 早停；长跑 OOM 用 `--resume`。
7. **模型档**：yolo26n 起步；每轮评估做规模适配双向检查（skill ⑨ 判据：指标平台期
   无分叉→升档 n→s；train 降 val 冻结→降档），不凭参数量直觉换档。
8. **评估入口统一**：独立 val 显式传 project；逐类表只采信有实例类。
9. **分裂种子**：jxl_split 随机划分未固定种子——M1 若需与冒烟同分布切分，先固定
   随机种子再切（本次未固定，M1 划分会与冒烟不同属正常）。

## 操作归属

本冒烟由 jxl 会话（s4det M1 前置任务）发起；数据正本（batch1/batch2 dataset 树）零改动，
全部产物落 /mnt/data/s4det/smoke-v0/ 与 /mnt/data/s4det/weights/smoke/。
