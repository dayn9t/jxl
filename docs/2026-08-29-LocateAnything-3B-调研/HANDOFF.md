# HANDOFF：LocateAnything-3B 目标定位后端接入 jxl

> 新会话开工文件。读完后先进 plan mode 出设计，确认后再写代码。

## 任务

仿 `src/jxl/det/yolo/d2d_yoloe.py` 的本地 detector 模式，新增 LocateAnything detector 类，并可注册为 `det_mine` 校验器候选。

## 先读

1. `docs/2026-08-29-LocateAnything-3B-调研/03-部署与微调.md`（环境/推理 API/微调）
2. `src/jxl/bin/rmb_ground.py`（云端 VLM 后端参照）、`src/jxl/bin/det_mine.py`（校验器接口）
3. `~/.claude/kb/30-areas/vlm-vision-grounding/20260829-locateanything-3b-model-card.md`（速查）

## 硬约束

| 项 | 值 |
|---|---|
| 环境 | `transformers==4.57.1` + `trust_remote_code=True`，独立 venv（与 jxl 主 uv 环境隔离） |
| 权重 | HF `nvidia/LocateAnything-3B` + MoonViT 单独拉 `moonshotai/MoonViT-SO-400M`；国内走 HF_ENDPOINT 镜像或 ModelScope `nv-community/LocateAnything-3B` |
| 推理 | 用官方 `LocateAnythingWorker`（不能裸调 `model.generate()`）；本机 RTX 4060 Ti 16GB，`--attn la_flash`，大图限 `--max-size` |
| 协议 | 坐标 [0,1000] 千分制整数，`<box><x1><y1><x2><y2></box>`；官方 `parse_boxes()` 解析 |
| 坑 | 多类 query 有 label corruption（官方承认）→ **单类逐条 query**；无 confidence 输出；结果有重复框需去重；`<box>none</box>` = 无目标 |
| 许可 | NVIDIA License **非商用**，只进研究/评估链路，不进商用标注管线 |
