# Video Event Overlay（vtag）设计

> 日期: 2026-08-03 · 状态: 设计稿

## 1. 背景与目标

`vdt run` 产出检测/跟踪/pose 演示视频(左上角绿色 HUD + 检测框 + `#id` 标签 + 骨架 + 尾迹)。
本工具是其后处理环节:**在既有视频上,按用户指定的时间段,于画面右下角叠加醒目的红色中文事件标签**(如「打架」「跌倒」),输出新视频。

`vdt run` → `vtag` 构成管线;两者业务无交集,仅在视频 IO(解码/编码)上共通,提取共享层。

**目标一句话**:给定视频 + 一组事件(起止秒 + 中文名),在每个事件时段的画面右下角叠加大号红字,写出 `_tagged.mp4`。

## 2. 非目标（YAGNI）

- 不做事件检测/识别(事件由用户显式指定,不从 pose/检测自动推断)。
- 不暴露位置/颜色 CLI(右下角 + 红色硬默认;`--font-size`/`--font` 可调)。
- 不做批量(单视频;批处理靠 shell 循环)。
- 不做动画/淡入淡出(静态叠字)。

## 3. 架构（三层）

```
┌─────────────────────────────────────────────┐
│ 共享视频 IO   jxl/io/video.py（新建）         │
│   VideoReader / VideoWriter（零业务依赖）     │
└───────────────┬─────────────────┬───────────┘
                │                 │
        ┌───────▼───────┐  ┌──────▼──────┐
        │ vdt（迁移）    │  │ vtag（新建） │
        │ OcvDecoder→    │  │ jxl/vtag/   │
        │  VideoReader   │  │  cli.py     │
        │ render→        │  │  overlay.py │
        │  VideoWriter   │  │             │
        └────────────────┘  └─────────────┘
```

- **共享层** `jxl/io/video.py`:通用视频读/写,不依赖任何业务子包。`vdt` 与 `vtag` 共用,消除解码/编码逻辑重复(单一数据源)。
- **vdt(最小迁移)**:`OcvDecoder` 改为基于 `VideoReader` 的薄适配(保持 `DecodeCfg` 接口不变,行为不变);`render_video` 内联的 `VideoWriter` 逻辑改用共享 `VideoWriter`。
- **vtag(新建 `jxl/vtag/`)**:`cli.py`(Typer 参数 + IO 编排,imperative shell)+ `overlay.py`(PIL 中文叠加,Functional Core)。

## 4. 共享层 `jxl/io/video.py`

```python
class VideoReader:
    """逐帧解码器。sample_fps=None 读全部帧(原 fps)；否则按 fps 等间隔采样。"""
    def __init__(self, path: str, sample_fps: float | None = None) -> None: ...
    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]:  # (frame_idx, ts_ms, frame_bgr)
        # 沿用 OcvDecoder 的 grab()/条件 retrieve() 优化(避免 cap.set 重解码)
    @property
    def fps(self) -> float: ...
    @property
    def size(self) -> tuple[int, int]:  # (w, h)

class VideoWriter:
    """H.264/mp4v 编码器,context manager 自动 release。"""
    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None: ...
    def __enter__(self) -> "VideoWriter": ...
    def __exit__(self, *exc) -> None: ...  # release
    def write(self, frame: np.ndarray) -> None: ...
```

迁移来源:`OcvDecoder`(jxl/vdt/decoder.py)的 grab/retrieve/采样逻辑提炼为 `VideoReader`;`render_video`(jxl/vdt/cli.py)的 fourcc/isOpened/write/release 提炼为 `VideoWriter`。

## 5. vdt 迁移影响面

- `jxl/vdt/decoder.py`:`OcvDecoder` 改为持有 `VideoReader`,转发 `__iter__`/`fps`/`size`;`DecodeCfg(fps)` → `sample_fps`。`_make_synthetic_video`(测试 helper)保留原处。
- `jxl/vdt/cli.py`:`render_video` 用 `with VideoWriter(...) as w: w.write(...)` 替换内联 `cv2.VideoWriter`。
- 测试:vdt 原有 decoder/render 测试保持绿(行为不变);`OcvDecoder` 公共接口不变,调用方(pipeline/cli)无需改。

## 6. vtag 模块

### 6.1 `jxl/vtag/overlay.py`（Functional Core,纯函数,零视频依赖）

```python
@dataclass(frozen=True, slots=True)
class EventSpec:
    name: str
    start: float   # 秒
    end: float     # 秒,start < end

@dataclass(frozen=True, slots=True)
class TagOpts:
    font_path: Path
    font_size: int = 48
    color_rgb: tuple[int, int, int] = (255, 0, 0)      # 红
    stroke_rgb: tuple[int, int, int] = (255, 255, 255) # 白描边
    stroke_width: int = 2
    margin: int = 20

def parse_event(s: str) -> EventSpec:
    """'打架,2.0-5.0' → EventSpec('打架', 2.0, 5.0)。
    非法(无逗号 / 非数 / start>=end / 负值)→ ValueError。"""

def draw_tags(
    canvas: np.ndarray, names: list[str], font: ImageFont.FreeTypeFont, opts: TagOpts
) -> np.ndarray:
    """右下角竖排红字白描边。names 为空 → 原样返回(零开销,不转 PIL)。
    否则 cv2(BGR)→PIL(RGB)→ImageDraw.text(stroke)→转回 BGR。"""
```

字体加载在 cli 渲染循环外做一次(`ImageFont.truetype(opts.font_path, opts.font_size)`),传入 `draw_tags`,不每帧 reload。

### 6.2 `jxl/vtag/cli.py`（imperative shell）

单命令工具(无子命令,唯一功能即叠 tag):

```python
app = typer.Typer(help="视频事件标签叠加（右下角中文红字）")

@app.command()
def main(
    video: Annotated[Path, typer.Argument(help="输入视频")],
    event: Annotated[list[str], typer.Option("--event", help="'名称,起秒-止秒'，可重复")],
    out: Annotated[Path | None, typer.Option("--out", help="输出（默认 <input>_tagged.<ext>）")] = None,
    font_size: Annotated[int, typer.Option("--font-size", help="字号（默认 48）")] = 48,
    font: Annotated[Path | None, typer.Option("--font", help="字体文件（默认 Noto CJK）")] = None,
) -> None: ...
```

console script:`vtag = "jxl.vtag.cli:app"`,调用形态 `vtag <video> --event ...`。

## 7. CLI 接口

```
vtag <video> --event '打架,2.0-5.0' --event '跌倒,10.0-12.0' [--out OUT] [--font-size 48] [--font PATH]
```

| 参数 | 必需 | 说明 |
|------|------|------|
| `video` | 是 | 输入视频(任意 mp4) |
| `--event` | 是(≥1) | `名称,起秒-止秒`,可重复 |
| `--out` | 否 | 默认 `<input>_tagged.<ext>` |
| `--font-size` | 否 | 默认 48 |
| `--font` | 否 | 默认系统 Noto Sans CJK |

事件格式:`<名称>,<起>-<止>`,名称不含逗号,时间为浮点秒,`起 < 止`。

## 8. 数据流

```
events = [parse_event(e) for e in --event]          # 校验
with VideoReader(video) as r, VideoWriter(out, r.fps, r.size) as w:
    font = ImageFont.truetype(opts.font_path, opts.font_size)
    for frame_idx, ts_ms, frame in r:
        t = ts_ms / 1000
        active = [e.name for e in events if e.start <= t < e.end]   # [start,end)
        frame = draw_tags(frame, active, font, opts)
        w.write(frame)
```

- 区间左闭右开 `[start, end)`。
- 多事件同时段:都叠,右下角竖排(首个事件最靠下)。

## 9. 视觉规格

- 位置:右下角,距右/下边 `margin`(默认 20px)。
- 文字:红色填充 `(255,0,0)` + 白色描边 `(255,255,255)`,`stroke_width=2`(任意背景可读)。
- 字号:默认 48(`--font-size`)。
- 字体:默认 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`(index 0;「打架/跌倒」等常用字无 region 字形差异);`--font` 可覆盖。
- 多事件:竖排,行间距随字号。

## 10. 错误处理（No Silent Degradation,严格不回退）

| 场景 | 行为 |
|------|------|
| 视频/字体文件不存在 | `typer.BadParameter` |
| `--event` 格式错 / `起>=止` / 负值 | `typer.BadParameter` |
| 无 `--event` | `typer.BadParameter`(至少一个) |
| 事件时段超出视频时长 | `typer.BadParameter`(不静默裁剪) |
| 视频无帧(0 帧) | 报错退出 |
| `VideoWriter` 打开失败 | 报错退出 |
| 字体加载失败(`truetype` 抛错) | 报错退出(不回退 cv2.putText) |

事件时段超出校验:读完首帧/元数据后用 `VideoReader` 的总时长(或首末 ts)判断;若实现复杂,至少校验 `end > 0` 且对明显越界报错。

## 11. 测试策略

**`jxl/io/video.py`**(共享层):
- `VideoReader` 全帧模式(sample_fps=None)帧数 == cv2 帧数;采样模式(fps=原fps)不丢帧。
- `VideoWriter` context 退出后写出可读 mp4(`cv2.VideoCapture` 可打开,帧数一致)。

**`jxl/vtag/overlay.py`**(纯函数,零视频依赖):
- `parse_event` 正常解析 + 各类非法(无逗号/非数/起>=止/负)→ `ValueError`。
- `draw_tags` 空列表 → 返回原帧(`np.array_equal`)。
- `draw_tags` 有事件 → 右下角 ROI 出现红色像素(`G<50 & R>200 & B<50`),左上角 ROI 不变。
- `draw_tags` 多事件 → 竖排(两个标签 y 不重叠)。

**`jxl/vtag/cli.py`**:
- `parse_event` 集成;非法 `--event` → `BadParameter`;无 `--event` → `BadParameter`;视频/字体不存在 → `BadParameter`。
- 端到端:合成视频 + 2 事件 → 输出可读 mp4;叠加帧(事件时段内)与非叠加帧(时段外)像素有差。

**vdt 回归**:vdt 原 decoder/render 测试迁移后全绿。

## 12. 库选型

- Pillow(已是 jxl 依赖,`pillow>=10.0`)— 中文 `ImageDraw.text`。
- cv2 — 视频 IO + BGR↔RGB 转换。
- typer/orjson/loguru/pydantic — 沿用 jxl 栈。
- 字体:Noto Sans CJK(系统自带,不引入新文件)。

无新增第三方依赖。

## 13. 参考

- vdt demo 设计:`2026-08-02-vdt-demo-design.md`
- vdt 检测跟踪设计:`2026-08-01-video-detect-track-design.md`
