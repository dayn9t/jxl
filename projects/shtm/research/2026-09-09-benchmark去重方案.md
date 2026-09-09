# SHTM benchmark 去重方案（2026-09-09，只读分析）

> S0 体检后续：全量内容级（md5）去重分析 + 干净重切分方案。数据源 /home/jiang/ws/trash 全程只读；
> 中间产物 /tmp/shtm_dedup/（md5×3、analyze_dedup.py、dup_groups.txt、removed_paths.txt、
> train/val/test_stems.txt、split_stats.txt）。**实际删除/重建等用户批准后执行（§6 脚本草案）。**

## 1. 重复规模（cabin/dates 9,064 张训练图全量 md5）

- **275 个重复组 / 990 张图涉及 / 715 张可删（7.9%）**，去重后唯一帧 8,349（与 S0 估计完全一致）。
- 组结构仅两种，全部为系统性拷贝、无零散意外重复：
  | 批次 | 结构 | 涉及 | 处置 |
  |---|---|---|---|
  | `dates/2023-12-28` | 110 唯一帧 × a~f 六前缀 | 660 | 保留 a_，删 b_~f_ ×550 |
  | `dates/2023-03-13` ≡ `dates/2024-03-13` | 整批 165 张两连拷（stem 都带 2024-03 前缀，日期目录造假） | 330 | 保留 2023-03-13，删 2024-03-13 整批 ×165 |
- 同名 stem 重复 165 个（= 03-13 双拷）；samples/（8,899 扁平池）恰为去 stem 重复后的全集，双拷在拍平时已被覆盖合并——但 **a~f 前缀拷贝因 stem 不同全部进入训练池**，是泄漏的直接来源。
- 新厢房 725 张：内部 0 重复、与 dates 池 0 重叠 → 可干净用作域外 test-domain。

## 2. 泄漏量化（现行 split：train 7,119 / val 890 / test 890）

- **跨 split 泄漏 79 个 hash 组**：25 个同时进 train+val+test、33 个 train+test、21 个 train+val；
  val 侧 54 帧（6.1%）、test 侧 75 帧（8.4%）与对侧内容完全相同。
- split 内冗余文件：train 421 / val 8 / test 17（a~f 拷贝被随机划分到同侧）。
- 79 组全部溯源到 dates 重复组（孤儿 0）→ **仅去重一项即可根除跨 split 精确重复泄漏**。
- 虚高幅度估计（2025-12-14 终评 all mAP50=0.974 / mAP50-95=0.916）：
  精确重复帧≈必然命中（per-frame mAP≈1.0），按 6.1% 占比 × 与真实泛化的差（~0.85-0.91）
  估 **mAP50-95 直接虚高 ~0.6-0.9 pt**；更大的未量化项是固定机位近邻帧
  （238/248 个 val 摄像头组同现 train，机制相同、强度稍弱），合计预计干净 benchmark 上
  **mAP50-95 下移 1~2 pt（→0.90 附近）**。量级远小于 n001 毒框事故（1,508 框 vs 此处 val 侧 54 帧），但同型。

## 3. 去重规则（275/275 组确定可判）

1. 保留**标注框数更多**者（对应 labels txt 非空行数）——实测 275 组全部平局（标签随图一起拷贝，框数相同）；
2. 平局取**更早日期路径**（字典序）：a_ 前缀 < b_~f_；`dates/2023-03-13/` < `dates/2024-03-13/`。
- 删除清单 `/tmp/shtm_dedup/removed_paths.txt`：715 jpg + 715 txt = 1,430 行绝对路径（成对，可直接 xargs rm）。
- 重复组全貌 `dup_groups.txt`（hash + 各路径）。

## 4. 干净重切分（8:1:1，分层键 = 摄像头 + 日期）

- **同 (cam,date) stratum 整组进同一 split**（grouped，1,744 个 stratum；cam 判别复用 S0 cluster_scan
  的 keys_of 正则族）——同摄像头同日帧不再跨 split，比按比例分层更彻底地切断近邻帧泄漏。
- 分配器：stratum 按规模降序贪心，代价 = 帧数 8:1:1 偏差 + 五类框数 8:1:1 偏差 + stratum 覆盖
  （权重 0.3），确定性输出（无随机种子，可复现）。
- 结果（stems 见 /tmp/shtm_dedup/{train,val,test}_stems.txt）：
  | split | 帧数 | opening | lid | can | dump | person |
  |---|---|---|---|---|---|---|
  | train | 6,730 (80.6%) | 11,339 | 2,673 | 536 | 481 | 257 |
  | val | 808 (9.7%) | 1,390 | 334 | 66 | 60 | 32 |
  | test | 811 (9.7%) | 1,440 | 335 | 65 | 60 | 32 |
- 五类均衡（旧随机切分 can 曾在 test 仅 8 框的翻车已避免）；val 覆盖 245 / test 176 个摄像头组。
- **残余泄漏声明**：481 个 cam 中 335 个跨多日、256 个仍出现在 ≥2 split（同 cam 不同日）。
  若要归零需 cam 级整组切分或时间 holdout——覆盖率代价大，建议作为后续独立决策，本期不做。

## 5. 产物清单（/tmp/shtm_dedup/）

| 文件 | 内容 |
|---|---|
| md5_dates.txt / md5_dataset.txt / md5_newroom.txt | xargs 并行 md5（9,064 / 8,899 / 725） |
| dup_groups.txt | 275 组：hash + 全部路径 |
| removed_paths.txt | 删除清单 1,430 行（jpg+txt 绝对路径） |
| train/val/test_stems.txt | 新切分 6,730 / 808 / 811 |
| split_stats.txt | 上表统计原文 |
| analyze_dedup.py | 全流程分析脚本（只读，可复跑） |

## 6. 执行脚本草案（待批准后跑，预计 <10 分钟）

```bash
# 0) 备份（硬链快照，几乎不占空间）
cp -al /home/jiang/ws/trash/cabin/dates  /home/jiang/ws/trash/cabin/dates.dedup-bak
cp -al /home/jiang/ws/trash/cabin/samples /home/jiang/ws/trash/cabin/samples.dedup-bak
# 1) 按清单删除（715 对）
xargs -d '\n' rm -v < /tmp/shtm_dedup/removed_paths.txt
# 2) 重建扁平池与软链 split（伪码：清空 samples/、dataset/ 后按 stem 清单从 dates/ 重建软链，
#    结构与现行一致：samples/images|labels/<stem>，dataset/<split>/images/<stem> -> samples）
python3 rebuild_split.py /tmp/shtm_dedup
# 3) 校验：重算三 split md5，断言跨 split hash=0、split 内重复=0；类分布 diff 对比 §4 表
# 4) 基线：cabin.pt 新 val 评测一次，记录"去重后真实基线"（预计 mAP50-95 ~0.90）
```

## 7. 开放项（需用户裁决）

1. samples/ + dataset/ 全量重建是否可接受（现行 dataset.yaml 路径不变，仅内容换血，旧 split 不可再复现）。
2. 残余同 cam 跨日泄漏（256/481）是否升级为 cam 级分组切分 / 时间 holdout。
3. `dates/2024-03-13` 删 165 张后目录留空壳还是连目录删除。
4. person 弱类（val 32 实例）补样另开任务，不阻塞本次去重。

## 关联

- S0 体检：`2026-09-09-S0体检报告.md`；盘点：`2026-09-08-trash数据盘点.md`；方案：`2026-09-09-检测器改进流程方案.md`
