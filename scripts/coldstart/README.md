# attr-wave2 冷启动数据与训练（审计链）

属性类事件第二波（jail `docs/superpowers/specs/2026-09-19-attr-wave2-design.md`）Task 5 的
数据来源、下载命令与转换记录。**数据集本体不入任何仓**（本目录 `.gitignore` 已排除），
入库产物只有 jail 仓 `rs/s3-detention/models/yolo26n-cls-*.onnx`。

类名契约（单一数据源 = plan Task 5 Interfaces；jail 侧按名找 positive 下标）：

| 事件 | class_names（序 = ImageFolder 目录名排序） | positive |
|---|---|---|
| head-cover | `["normal", "covered"]` | covered（idx 1） |
| drowse | `["awake", "drowsy"]` | drowsy（idx 1） |
| phone-in-hand | `["other", "phone_in_hand"]` | phone_in_hand（idx 1） |

工具（本目录）：

- `golden_check.py` — .pt predict vs 契约化 ONNX 前向概率对拍（导出数值保真，<1e-3；
  双侧同一 ultralytics 式预处理，不覆盖 Rust/usls 在线路径）
- `eval_classifier.py` — 契约化模型 × ImageFolder val/test → 混淆矩阵 + top-1（EVAL 基准数字产器）

## head-cover（SLP）——❌ 受阻，模型未入库

- 目标来源：SLP（Simultaneously-collected multimodal Lying Pose，Northeastern ACLab，
  109 受试者 RGB/LWIR/depth/PM，uncover/thin/thick 覆盖条件）。
- 官方下载：`https://binary.coe.neu.edu/Research/AClab/SLP/SLP2022.zip`（17.5 GB，直链可下）。
- **受阻点**：zip 内数据文件加密（实测中央目录 15,489/15,904 entry 置 encryption bit）；
  密码须在 ACLab 页面填 request form 并用机构邮箱申请——本任务无可用机构邮箱通道，
  且规则禁止注册/表单类获取。HF 无镜像（`SLP` 关键词仅文本数据集；in-bed/blanket 关键词空）。
- 结论：head-cover 冷启动 blocked，`yolo26n-cls-head-cover.onnx` 不入库（No Silent
  Degradation——占位模型禁止）。jail 侧该事件走 fake-classifier 测试 + example 注释态。
- 代码仓（仅文档参考，未用其数据）：`slp/code/` = github.com/ostadabbas/SLP-Dataset-and-Code 浅克隆。

## drowse——akahana/Driver-Drowsiness-Dataset（HF 公开镜像）

- 来源：`https://huggingface.co/datasets/akahana/Driver-Drowsiness-Dataset`
  （匿名直链，无需注册；paper *Detection and Prediction of Driver Drowsiness ...*，
  WITS 2020，DOI 10.1007/978-981-33-6893-4_6）。
- 形态：HF parquet（image + class_label 0=Drowsy / 1=Non Drowsy；train 33,434 / test 8,359）。
- 域差声明：驾驶员座舱域，非监控室值班民警域——仅预训练/链路验证，指标待域内微调。
- 下载：

  ```bash
  cd drowse/data_raw
  for f in train-0000{0..4}-of-00005 test-0000{0,1}-of-00002; do
    curl -LO "https://huggingface.co/datasets/akahana/Driver-Drowsiness-Dataset/resolve/main/data/$f.parquet"
  done
  ```

- 转换：`uv run python scripts/coldstart/drowse/parquet_to_imagefolder.py`（polars 读
  parquet——venv 既有依赖；HF 0=Drowsy→`drowsy/`、1=Non Drowsy→`awake/`；train 内
  每分片每类前 5% 作 val；test 原样）。
- **标签方向验证**（转换前核）：HF imagefolder 的 label 序 = 类目录名排序
  （"Drowsy"(0) < "Non Drowsy"(1)，与 dataset card 一致）；蒙太奇抽检 label-0 多数
  闭眼/垂头、label-1 全部清醒。标签为 NTHU-DDD clip 级——含少量过渡帧噪声（个别
  高置信 drowsy 预测帧目视清醒），README 记录在案，域内微调时复核。
- 训练：`uv run yolo classify train model=yolo26n-cls.pt data=<abs>/drowse/dataset imgsz=224 epochs=30 device=0 project=<abs>/drowse/runs name=train`
  （本机 RTX 4060 Ti；产物 `drowse/runs/train/weights/best.pt`）。
- 导出（契约化）：

  ```bash
  uv run python -m jxl.bin.export_yolo_with_contract \
    scripts/coldstart/drowse/runs/train/weights/best.pt \
    ~/cc/next/jail/rs/s3-detention/models/yolo26n-cls-drowse.onnx \
    --task classify --imgsz 224 --crop head:0.35
  ```

- 结果：见下「训练结果」表。

## phone-in-hand——State Farm 正例 + COCO 负例（均公开直链）

- 正例来源：`https://huggingface.co/datasets/gymprathap/Driver-Distracted-Dataset`
  （匿名直链 4.3 GB zip；State Farm Distracted Driver Detection 竞赛集公开镜像）。
  10 类司机行为 c0-c9（c1-c4 = 左右手打字/打电话，即手持手机行为）。
- 负例来源：COCO 2017 `cell phone(77)` 标注框——多为桌面/收纳中的手机（「有手机
  但不在手里」= 复核层负分布；少量手持帧为可接受标注噪声）。
  - val2017 全量 262 实例/214 图（`http://images.cocodataset.org/zips/val2017.zip`）；
  - train2017 定向单图 4,803 张/6,434 实例。⚠️ 单图直连
    `images.cocodataset.org` 在本机 DNS 挂起——改走 S3 端点
    `https://s3.amazonaws.com/images.cocodataset.org/train2017/<file>`（URL 清单
    由脚本外一次性生成：`train2017_phone_urls_s3.txt`，xargs -P 8 curl）。
- 裁切配方（与在线 PhoneVerifier 同源于契约 `phone_context:0.5`，复用
  `jxl.bin.build_attr_crops.phone_context_crop`）：
  - 正例 `phone_in_hand`：c1-c4 → yolo26n 检 cell phone(67) conf≥0.25
    （实测产出率 ~15%/图）→ bbox 外扩 0.5 crop，全量 9,256 源图 → ~1.5K crop；
  - 负例 `other`：COCO 标注框 → 同策略 crop，取前 2,000（`MAX_OTHER_CROP`，
    与正例量级平衡）。
  - （弃用配方：c0/c5-c9 低阈值假阳性作负例——conf 0.10 下产出为 0，不可行。）
- 下载：

  ```bash
  cd phone_in_hand/data_raw
  curl -LO "https://huggingface.co/datasets/gymprathap/Driver-Distracted-Dataset/resolve/main/Distracted-Driver-Detection-Dataset.zip"
  unzip -q Distracted-Driver-Detection-Dataset.zip   # 解出 imgs/train/c0..c9
  cd ../coco
  curl -LO http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip
  curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip
  xargs -P 8 -n 1 curl -s --retry 3 -O --output-dir train2017_dl < train2017_phone_urls_s3.txt
  ```

- 构建：`uv run python scripts/coldstart/phone_in_hand/build_phone_dataset.py`
  （每类每 20 张取 1 作 val；`phone_manifest.json` 留痕）。
- 训练/导出：同 drowse，`--crop phone_context:0.5`，产物
  `rs/s3-detention/models/yolo26n-cls-phone-in-hand.onnx`。

## 训练结果（golden 对拍 + EVAL 基准数字）

每模型：golden 对拍（.pt predict vs 契约化 ONNX 前向，导出保真 <1e-3，不覆盖 Rust/usls 在线路径）+
`eval_classifier.py` val 混淆矩阵（⚠️ 代理级——域外分布，待域内微调）。

| 模型 | 训练 val top-1 | golden worst | EVAL val | 备注 |
|---|---|---|---|---|
| `yolo26n-cls-drowse.onnx` | 1.000（30ep） | 2.61e-15（3 crop） | top1=1.000（10,026 张：awake 4,655 / drowsy 5,371 全对） | 域=驾驶员人脸；标签 clip 级含过渡帧噪声 |
| `yolo26n-cls-phone-in-hand.onnx` | 0.994（30ep） | 5.49e-16（4 crop） | top1=1.000（177 张：other 100 / phone_in_hand 77 全对） | 正=司机手持；负=COCO 桌面手机（域差最大，仅预训练） |
| `yolo26n-cls-head-cover.onnx` | —（受阻） | — | — | SLP 密码表单不可得，未训练未入库 |

训练环境：本机 RTX 4060 Ti（drowse 30ep ≈ 22min；phone 30ep ≈ 2min），
`device=0`；golden_check 固定 `device=cpu`（pt/onnx 同设备确定性对拍）。
