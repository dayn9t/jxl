# 任务书 D0：v0-mini 训练（P4 干线验证 + 专项权重，2026-09-25）

> 性质：小任务——用批 1 种子集打通训练干线 + 产出 s4det 专项替身权重（C 遗留#4 的复壮前置）。
> 前置：批 1 dataset 已验收（1,481 帧/1,403 框，近重复清洗后落盘；s4 会话 2026-09-25）。执行：jxl 会话。

## 内容

1. **训练**：`yolo_train /mnt/data/s4det/p3-seedset/batch1/mining/dataset/data.yaml --model yolo26n.pt`（官方 COCO 底座，skill ⑨ 单场景特化档；epochs 100 起步，规模适配检查按 skill ⑨ 双向判据记录）。
2. **产物**：best.pt → `/mnt/data/s4det/weights/s4det-v0mini.pt`；profiles（targets/*.toml）weights 字段换装此权重（garbage_bag 复壮重挖在批 2 一并进行，勿单独立即重挖）。
3. **指标落档**：train/val mAP 记入 `docs/EVAL-BENCHMARKS.md` §2 表——**标注「内部预演指标，WARNING-1 未解除前不得当质量结论」**（无人工 GT，n001 虚高教训）。
4. **纪律**：训练前 `nvidia-smi` 查共卡余量（nano batch 32 需 ~6G，绝不杀业务进程）；长跑用 setsid 脱离托管；OOM 用 --resume 续。

## 验收

训练 exit 0 + val 指标产出 + 指标落档 + weights 换装完成，回写 D0-execution 备注（可并入 C-execution.md 尾节）。
