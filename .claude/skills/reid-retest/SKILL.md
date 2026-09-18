---
name: reid-retest
description: OSNet/reid 新版本的 §7.4 两段门复测流程（temp 通道）。当交付了新 reid 权重需复测（管道门/定标门）、replay 参数扫描、cos_threshold 定标、benchmark 六合并组或 split_points 对账时使用。含 v0.9→v0.10 视频源迁移、pkill 自噬、uv 孤儿进程等踩坑清单。
---

# OSNet reid 复测流程（§7.4 两段门，temp 通道）

> 前身：v2 复测 + v2.1 预复测（2026-09-17，`/mnt/data/jiang/ws/iapx/n001-ft21-retest/`）。
> 该目录本身是可复制模板（scripts/ 三件套 + 产物结构）。

## 1. temp 通道搭建（复制 ft21 模式）

```
scripts/<tag>.toml         # baseline 复制改 reid=<新 tag>
scripts/reembed_<tag>.py   # 改 CFG/TEMP 指向 + VIDEO_ROOT_V10 patch + assert cfg.reid
scripts/replay_eval.py     # run_mod.VIDEO_ROOT monkeypatch（_scan_mkvs 读模块全局）
```

- **VIDEO_ROOT 必 patch 到 v0.10**：`/var/howell/iap/v0.10/ias/sh-sgcc/n001/video`
  （v0.9 已整体迁移下线，不 patch 则 reembed 空转/replay "no caches"）
- 生产 iapx 通道**零写入**——全部在 temp 目录

## 2. 跑批

1. reembed（849 session 量级 ~65min，4060Ti；有 resume）
2. replay_eval 参数扫描：thr × guard 网格，注意 thr 定标带会随版本**移位**
   （v2→v2.1 从 ~0.62 移到 0.50-0.60，v2 旧运行值不可沿用）

## 3. 两段门判读

| 门 | 判据 | 基线参照 |
|---|---|---|
| 管道门 | GT 15 窗 F1≥0.9677 + 过切 ≤2 + 换人 0（±10s 对齐） | handcrafted=0.9677 |
| 定标门 | v2c ASYM same_p5 > diff_p95 | v2 +0.001 / v2.1 +0.051 |

- 改善归因必须落到**具体窗口**（如 07-01 13-50 窗）——全量 PASS 但靶点未修要如实报告
  （v2.1 案例：10-41 段层不对症）

## 4. benchmark 对账

- 六合并组（`benchmark/v3/expected-visits.json` merge_groups，sessions 落段匹配；
  对照 v2/v3 status 的 fixed 翻转——**持平也要看构成**）
- **split_points 交叉**：候选对/训练对跨 split_points（换人点）= 毒标注警报
  （2026-09-17 勘误：铁证对 3/5 跨换人错标——origin/时间戳可核）

## 5. 落盘

eval-report.md（含口径说明）+ 定标报告 + 通知单**同文件追加**复测结果段 +
memory 更新。发现靶点未修 → vN+1 方向建议（附归因）。

## 踩坑清单（每次必读）

- `pkill -f <pattern>` 在 ssh 命令串里自噬会话 → 用字符类 `pkill -f "[p]attern"`
- kill uv wrapper 不杀 python 子进程 → 杀后 `pgrep -af` 查孤儿
- iapx `consts.VIDEO_ROOT` 常量若仍未修，一切复测先确认视频源真实存在
  （`find ... -name "*.mkv" | wc -l`）
