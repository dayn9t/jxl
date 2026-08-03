---
title: vdt 演示 CLI 与完整可视化设计 spec
project: py/jxl
date: 2026-08-02
status: design-approved, ready-for-impl
scope: 合并演示能力到 vdt run（--config 可选 + 完整可视化 + 选项开关）
audience: 实现 vdt 演示 CLI 的工程师 / Claude
depends_on: docs/superpowers/specs/2026-08-01-video-detect-track-design.md（vdt 已实现）
---

# vdt 演示 CLI 与完整可视化 — 设计 spec

> 给读者：本文基于**已实现**的 vdt 模块（见 `2026-08-01-video-detect-track-design.md`，P1/P2/P3 已落地：detect→双模 track→条件性 pose 管线 + `Tracks` 数据结构 + `vdt run` CLI），在其上增加「完整跟踪演示可视化」能力，并把演示与产物统一合并到单一 `vdt run` 命令。

## 0. 一句话目标

让用户**一条命令**对视频做跟踪演示，输出把所有跟踪信息绘制上去的视频：

```
vdt run video.mkv --out-video demo.mp4
```

零配置（智能默认 = iou 跟踪 + yolo26n 检测 + pose 骨架全开），输出视频含**完整可视化**：检测框 + 按 track_id 稳定配色 + COCO 17 点 pose 骨架 + 运动轨迹尾迹 + 顶部 HUD。必要时用选项 / `--config` 精细控制。

## 1. 范围与不在范围

**在范围**
- 新建 `src/jxl/vdt/draw.py` 可视化库（纯绘制函数 Functional Core）
- 合并到 `vdt run`：`--config` 改可选（无则内置默认配置工厂）、`--out-video` 走完整可视化、加可视化开关选项
- pose 骨架 / 轨迹尾迹 / HUD / ID 稳定配色四类绘制

**不在范围**
- ❌ 新增独立 `vdt demo` 命令（已否决：合并到 `run`，YAGNI；一个命令 + 选项覆盖全场景）
- ❌ 实时流 / 交互式调参
- ❌ 可视化美学的精细调优（字号 / 配色具体数值用合理默认，不调参）

## 2. 现状与复用

- **现有 `vdt run video --config X --out-video out.mp4` 已能输出标注视频**，但 `cli.annotate_video` 仅调 `draw_d2d_objects` 画**检测框 + `id(置信度)` 文字**、颜色按**类别**（`COLORS7[cls]`）——**不画 pose 关键点**（即使 Tracks 里有 kpts）、无轨迹尾迹 / 时间戳 / HUD，且 `--config` 当前必需。
- `jxl.det.d2d.draw_d2d_objects` / `jxl.io.draw.draw_boxf`（框 + label）；`jvi.drawing.color`（COLORS7 / random_color）。
- **`Tracks` 数据齐备**：`Track.frames[i]` 含 `objects`（带 `id`、归一化 `rect`）+ 对齐的 `kpts`（归一化 `Point`，COCO 17 点）。渲染所需信息全在。
- COCO 17 点骨架拓扑：标准（mmpose RTMPose `coco` skeleton 为权威源）。

## 3. 架构（方案 A）

新建 `vdt/draw.py` 可视化库（Functional Core，纯函数 + 可独立单测），`cli.py` 升级为薄消费者复用它。

```
src/jxl/vdt/
├── draw.py     # 新建：DrawOpts + 绘制函数 + TrailBuffer + render_demo_frame
└── cli.py      # 改：_default_config 工厂 + --config 可选 + render_video（替换 annotate_video）+ 选项开关
```

**为何独立 `draw.py`**：绘制是 vdt 演示专属逻辑、是纯函数（输入 canvas + 数据 → 就地画）→ 独立模块 + 充分单测（设计原则 6 Functional Core）；`cli.py` 维持薄壳不膨胀（现 438 行，绘制塞入会超 800、SRP 乱）。

## 4. 组件 — `draw.py`

### `DrawOpts`（frozen dataclass，绘制开关）

```python
@dataclass(frozen=True, slots=True)
class DrawOpts:
    box: bool = True
    skeleton: bool = True
    trail: bool = True
    hud: bool = True
    trail_len: int = 30
```

### ID 稳定配色

```python
def color_for_id(track_id: int) -> tuple[int, int, int]:
    """HSV 色相 = (track_id * 0.6180339) % 1.0（黄金角分布，相邻 id 色相差大），
    S=0.85、V=0.9 → BGR uint8 三元组。同 id 跨帧稳定。"""
```

> 替换现有按类别的 `COLORS7[cls]`——演示场景多人同类别（cls=0 person）需按 id 区分。

### 绘制函数（均就地画到 `canvas: np.ndarray`，输入坐标为归一化 → 内部转像素）

- `draw_track_box(canvas, obj: D2dObject, color: tuple[int,int,int]) -> None`
  - `obj.rect`（归一化）→ 像素框 `cv2.rectangle`；左上角 ID 标签 `cv2.putText(f"#{obj.id}")`（带背景）。
- `COCO_SKELETON_EDGES: tuple[tuple[int, int], ...]`
  - COCO 17 点骨架边集，以 **mmpose RTMPose `coco` skeleton** 为权威源（实现时 port）。每条边端点 ∈ `[0, 17)`、无重复边。
- `draw_pose_skeleton(canvas, kpts: Keypoints | None, color, kconf: float = 0.35) -> None`
  - `kpts is None` → 不画。否则对每个关键点：`conf < kconf` 视为不可见，**不画该点圆点、且其参与的边不画**；可见点画 `cv2.circle`，两端皆可见的边画 `cv2.line`。
- `TrailBuffer`（命令式，但 `draw` 是纯绘制）：
  ```python
  class TrailBuffer:
      def __init__(self, max_len: int) -> None: ...      # per-id deque[Point]（归一化中心）， maxlen=max_len
      def push(self, track_id: int, center: Point) -> None: ...
      def draw(self, canvas: np.ndarray) -> None: ...    # 用 color_for_id 配色；逐段衰减
  ```
  - `draw` 衰减：每条 trail 逐相邻段，按年龄（age=0 新 → max_len 旧）alpha 线性 `1.0→0.2`、线宽 `2→1`；分段 `cv2.line`（因 `cv2.polylines` 单色单宽，无法逐段衰减）。
- `draw_hud(canvas, frame_idx: int, ts_ms: int, n_objects: int, tracker_mode: str) -> None`
  - 左上角半透明背景条 + `cv2.putText`：`f"f#{frame_idx}  {ts_ms/1000:.1f}s  |  {n_objects} objs  |  {tracker_mode}"`。
- `render_demo_frame(canvas, objects, kpts, trails, frame_idx, ts_ms, tracker_mode, opts: DrawOpts) -> None`
  - 按 `opts` 总装，绘制顺序 `box → skeleton → trail → hud`（后绘制者在顶层；HUD 最后画以居顶不被框/骨架/尾迹遮）。

### 坐标转换

归一化坐标 → 像素：`x_px = x_norm * img_w`，`y_px = y_norm * img_h`（`canvas.shape[:2]` 得 `img_h, img_w`）。复用 `jvi` `Rect.absolutize(Size)`（`draw_track_box`）/ 直接乘（点）。

## 5. `cli.py` 改造

### `_default_config() -> VdtConfig`（默认配置工厂）

```python
_REPO_ROOT = Path(__file__).resolve().parents[3]

def _default_config() -> VdtConfig:
    return VdtConfig(
        tracker="iou",
        decode=DecodeCfg(fps=25.0),                                   # 多数视频 stride≈1（近全采）
        det=DetCfg(model=str(_REPO_ROOT / "yolo26n.pt"), conf=0.3, classes=[0]),
        tracker_cfg=IouCfg(iou_thr=0.3, max_age=30, min_hits=2),
        pose=PoseCfg(model=str(_REPO_ROOT / "rtmpose-17-m.onnx"), keyframe_every=5, min_hits=2),
    )
```

模型 gitignored；新 clone 缺失 → `run` 阶段 `build_detector` / `build_pose` 抛 `ModelLoadError`（fail-fast，提示路径，spec §9）。`fps=25`：对 25/30fps 视频 stride≈1 近全采；精确控制（含低帧率 reid）用 `--config`。

### `run_cmd` 选项（合并后的命令面）

```
vdt run <video> [--config <toml>] [--tracker iou|reid] [--no-pose]
               [--out-tracks <json>] [--out-video <mp4>]
               [--no-box] [--no-skeleton] [--no-trails] [--no-hud] [--trail-len <N>]
```

- `--config`：**可选**。`None` → `_default_config()`；非 None → `load_config`（可被 `--tracker` / `--no-pose` 覆盖）。
- `--tracker`：覆盖配置中的 tracker（须与 config 的 `tracker_cfg` 子表匹配，否则 `BadParameter`，沿用现有 `load_config` 逻辑）。
- `--no-pose`：剥离 `pose`（`config.pose=None`）。
- 可视化开关（`--no-box/--no-skeleton/--no-trails/--no-hud`）+ `--trail-len` → 构造 `DrawOpts`；**仅 `--out-video` 时生效**（无视频输出则不画）。
- **至少一个 `--out-*`**：否则无产出 → `typer.BadParameter`。

### `render_video(video_path, tracks, out_path, opts)` — 替换 `annotate_video`

```python
def render_video(video_path, tracks: Tracks, out_path: Path, opts: DrawOpts) -> None:
    # 1. 从 Tracks（按 id 聚合）拆回逐帧：{frame_idx: (objects, kpts)}
    frame_map: dict[int, tuple[list[D2dObject], list[Keypoints | None]]] = {}
    for tr in tracks.tracks:
        for fr in tr.frames:
            objs, kpts = frame_map.setdefault(fr.frame_idx, ([], []))
            objs.extend(fr.objects); kpts.extend(fr.kpts)
    # 2. 同 fps 重解码（frame_idx 与 Tracks 对齐）
    decoder = OcvDecoder(video_path, tracks.config.decode)
    trails = TrailBuffer(opts.trail_len)
    writer = cv2.VideoWriter(...)
    for frame_idx, ts_ms, frame in decoder:
        objs, kpts = frame_map.get(frame_idx, ([], []))
        canvas = frame.copy()
        for ob in objs:                              # 尾迹按时间累积
            if ob.id != 0:
                trails.push(ob.id, _center(ob.rect))
        render_demo_frame(canvas, objs, kpts, trails, frame_idx, ts_ms,
                          tracks.config.tracker, opts)
        writer.write(canvas)
```

> **关键**：`Tracks` 是按 id 聚合的时间线，不直接给「某时刻的尾迹」；尾迹必须在渲染循环按 `frame_idx` 顺序累积。`_center(rect) = rect.center()`（归一化 Point）。

## 6. 错误处理（沿用 spec §9 fail-fast）

| 场景 | 处理 |
|---|---|
| 无 `--out-*` | `BadParameter`（无产出） |
| `--trail-len <= 0` | `BadParameter` |
| `--config` 不存在 / 非法 `--tracker` | `BadParameter`（沿用 `load_config`） |
| 模型缺失（默认配置或 config 指向不存在权重） | `ModelLoadError`（`run` 阶段） |
| `VideoWriter` 打不开 / 重解码 0 帧 | `DecodeError`（均已实现） |

`DrawOpts`：`frozen` dataclass（不可变）；`trail_len>0` 在 cli 层构造前校验。

## 7. 测试

**`draw.py` 纯函数单测（零模型）**
- `color_for_id`：稳定（同 id 多次调用同色）+ 区分（前 20 个不同 id 无碰撞）。
- `COCO_SKELETON_EDGES`：每边端点 ∈ `[0,17)`、无重复边、非空。
- `draw_pose_skeleton`：合成 `Keypoints`（部分点 `conf<0.35`）→ 断言低置信点不画、其邻接边不画；`kpts=None` 不画。
- `TrailBuffer`：环形（push 超过 `max_len` 后丢最旧）+ `draw` 不崩（合成 canvas）。
- `render_demo_frame`：各 `DrawOpts` 开关组合（全开 / 全关 / 仅 hud 等）不崩（合成 canvas + objects + kpts）。

**`cli.py` 单测**
- `_default_config`：字段正确（tracker=iou、model 路径、pose 非 None）。
- `--config` 可选：无 → 默认配置；有 → `load_config`。
- `--trail-len 0` / 无 `--out-*` → `BadParameter`。
- 可视化开关 → 正确 `DrawOpts`。

**集成**
- 合成短视频（`decoder._make_synthetic_video`）+ 合成 `Tracks`（含 id + kpts）→ `run --out-video` → 输出 mp4 可读（`cv2.VideoCapture` 帧数对齐）。

## 8. 已定决策日志

- **命令形态** = 合并到 `vdt run`（否决独立 `vdt demo`：YAGNI；`--config` 可选 + 选项驱动覆盖全场景，认知负担小）。
- **可视化深度** = 完整（框 + ID 稳定配色 + pose 骨架 + 轨迹尾迹 + HUD）。
- **零配置默认** = iou + `yolo26n.pt` + pose（`rtmpose-17-m.onnx`）全开。
- **代码组织** = 新建 `draw.py`（Functional Core，纯函数可单测；`cli` 薄壳）。
- **fps 默认** = 25（近全采；精确用 config）。
- **ID 配色** = HSV 黄金角（替换按类别 `COLORS7`）。
- **绘制顺序** = box → skeleton → trail → hud（后画者在顶层；HUD 居顶不被遮）。

## 9. 实现分期（建议）

- **D1** — `draw.py` 可视化库 + 纯函数单测（独立、零模型、先落地）。
- **D2** — `cli.py` 改造：`_default_config` + `--config` 可选 + `render_video`（替换 `annotate_video`）+ 选项开关 + cli 单测 + 集成测试。

每阶段独立可跑、可测、可 commit。
