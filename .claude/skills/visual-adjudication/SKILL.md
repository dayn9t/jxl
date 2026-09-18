---
name: visual-adjudication
description: 看图裁决流程——从 mkv+检测缓存裁代表帧、多链路 VLM 鉴定（spark 首选/4.5v 备用/人工兜底）、三选一结论记录。当需要判定检测框内容物（静物 vs 真人）、跨帧身份比对（同人或换人）、悬案裁决（benchmark 争议段）、或对负样本池抽检复核时使用。
---

# 看图裁决（裁帧 → VLM 鉴定 → 结论记录）

> 前身案例：10-41 铁证对裁决（2026-09-17，发现 3/5 跨换人错标）、neg0906
> 全检（66 张零剔除）、09-06 三时段静物判定。工具：`gencheck/vlm_gate.py`。

## 1. 裁代表帧（mkv keyframe ↔ 缓存 json 对齐）

```python
# 缓存 frame.time "2026-09-06/09-40-03.000"；mkv 名 09-40-02.000 = 起始秒
offset = secs(frame_time) - secs(mkv_stem)          # 相对秒
# ffmpeg 前置 seek（keyframe ~2s 对齐，误差无碍静物；动态场景取两侧帧）
ffmpeg -ss {max(0, offset-0.5):.1f} -i <mkv> -frames:v 1 \
       -vf crop={w}:{h}:{x}:{y} -q:v 2 out.jpg      # 以 bbox 为中心 ±pad(60)，640 窗口按需
```

选帧策略：跨时段（早/峰/晚）各取代表；同簇去重后再裁；怀疑「内容物变化」时
每簇多帧。

## 2. 鉴定链路（按序降级）

| 序 | 链路 | 用法 | 坑 |
|---|---|---|---|
| 1 | **spark**（首选） | `vlm_gate.py --dir <crops> --mode neg-person / object-id` | 并发 ≤3 硬红线；enable_thinking:false（否则 content=None） |
| 2 | 4.5v analyze_image | Read 图 → 拿 CDN URL → MCP 调用 | **429 限速常见**（三张连发即触发）；URL 一次性——过期/400 就重新 Read 换新 URL；签名复制易错一字符 |
| 3 | 人工 | 每簇首张 + 全部「有人」判定复核 | verifier 会错，SAM 3 EV 也只近人 |

**CDN 内容寻址彩蛋**：Read 返回的 URL 若与此前某张完全相同（含签名），两文件
内容相同——可免费发现「同图重复」（10-41 案例中借此发现 p2_a≡p1_a）。

## 3. 结论记录（三选一，禁含糊）

- **全对** / **全错** / **混合**（逐项列表：哪些对哪些错、判据=衣着/发型/体型特征组）
- 身份比对必须做 **det_index 交叉核对表**（帧时间戳 × det 序号 → 特征 → 裁决）；
  同帧多人共存（如同帧 det0/det2）是错配的常见物证
- 裁决结论直接进 research 文档（悬案）/ vlm_gate_result.json（入池门）

## 4. 红线

判定「真人」的标准从严（任何姿态/位置/倒影都算）——负样本门宁误杀勿放过
（毒标注代价 > 少几张素材）；「无人」结论全过才入池，exit code 即门槛。
