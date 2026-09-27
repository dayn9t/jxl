# 批 1 garbage_bag 全量复审（wave1，2026-09-27）

- 方法：复用批 2 gb_audit 管道（run_audit_b1.py 适配）——机器双审（最新加固 prompt sha256_12=77506deff114 + downscale=2，qwen38-local→qwen-flash 二审）+ 逐框目视终裁 + 对照组。
- 结果（301 框）：**确认污染 19（6.3%）** / clean 274 + 双 VLM 漏检目视平反 8；污染主面=篷布/防水布/苫布/泡沫箱（同批 2），来源 review 12 + L1 7。
- 清洗执行（主会话）：19 框全删（19/19 数值匹配命中），**b1 1,401→1,382**；备份 labels-backup-20260927.tar；产物在 /mnt/data/s4det/p3-seedset/batch1/mining/gb_audit/。
- **v0mini 重训裁决（依据量化）**：garbage_bag 类内污染 21/303≈6.9% 曾入 v0mini 训练集；v0mini 的下游消费（批 2 复壮链 target 票）产物已清洗，且 M1 v0.1 将以干净数据 + yolo26 官方底座重训替代——**不单独重训 v0mini**，污染链就此截断。
- 教训（承批 2）：校验器支持闸门（批 1 有）确实显著压污染（6.3% vs 批 2 无闸门 20.3%）——复壮链 target 票质量是新污染入口，M1 后换新权重时同步换。
