# Video Event Overlay（vtag）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `vtag` 工具——在既有视频上按事件时段于右下角叠加红色中文事件标签,输出新视频。

**Architecture:** 三层——共享 `jxl/io/video.py`(VideoReader/VideoWriter)、vdt 迁移、vtag 新建(`cli.py` + `overlay.py`)。详见 spec `docs/superpowers/specs/2026-08-03-video-event-overlay-design.md`(权威,本 plan 引用其段落)。

**Tech Stack:** Python 3.12 / cv2 / Pillow(typer/orjson/loguru/pydantic 沿用 jxl 栈)。

## Global Constraints

- spec:`docs/superpowers/specs/2026-08-03-video-event-overlay-design.md` 权威,接口签名见 spec §4/§6。
- 编码规范:j-python-strict——mypy strict、ruff、完整类型注解、`@dataclass(frozen=True, slots=True)`、No Silent Degradation(严格不回退)、文件 <800 行/函数 <50 行。
- TDD:先测试(RED)→ 实现(GREEN)→ 提交;每个 Task 独立提交。
- 无新增第三方依赖(Pillow `>=10.0` 已是依赖)。
- 事件区间 `[start, end)` 秒;多事件竖排;右下角红字 `(255,0,0)` + 白描边 `(255,255,255)`。
- 测试惯例:参考 `src/jxl/vdt/draw.py` / `cli.py` 的 **inline test**(module-level `test_*` 函数,pytest 自动发现)。

---

### Task 1: 共享视频 IO 层 `jxl/io/video.py`

**Files:**
- Create: `src/jxl/io/video.py`
- Test: inline(`src/jxl/io/video.py` 末尾 `test_*` 函数)

**Consumes:** cv2(VideoCapture/VideoWriter)、numpy、jxl.vdt.decoder._make_synthetic_video(合成视频测试 helper,已存在)。
**Produces:** `VideoReader`、`VideoWriter`(spec §4 接口)。

**Steps:**
- [ ] 写 `VideoReader` 测试:用 `_make_synthetic_video` 生成合成视频;断言全帧模式(`sample_fps=None`)帧数 == 合成帧数、`fps`/`size` 属性正确。
- [ ] 写 `VideoWriter` 测试:写 N 帧 → `cv2.VideoCapture` 读回,断言 `isOpened` + 帧数一致;`with` 退出后文件可读。
- [ ] 实现 `VideoReader`:从 `jxl/vdt/decoder.py` 的 `OcvDecoder` 提炼 grab/retrieve 优化(避免 `cap.set` 重解码);`sample_fps=None` 读全帧,否则按 fps 采样;`__iter__` yield `(frame_idx, ts_ms, frame_bgr)`;`fps`/`size` property。
- [ ] 实现 `VideoWriter`:`fourcc=mp4v`、`isOpened` 检查(失败抛错)、`write(frame)`、context manager(`__enter__`/`__exit__` release)。
- [ ] `uv run pytest src/jxl/io/video.py -q` + `uv run mypy src/jxl/io/video.py` + `uv run ruff check src/jxl/io/video.py` 全过。
- [ ] Commit: `feat(io): add shared VideoReader/VideoWriter`。

### Task 2: vdt 迁移到共享层

**Files:**
- Modify: `src/jxl/vdt/decoder.py`(`OcvDecoder` → 持有 `VideoReader` 薄适配,公共接口不变)
- Modify: `src/jxl/vdt/cli.py`(`render_video` 用 `VideoWriter` context)

**Consumes:** Task 1 的 `VideoReader`/`VideoWriter`。
**Produces:** `OcvDecoder` 接口不变(行为不变);`render_video` 用共享 `VideoWriter`。

**Steps:**
- [ ] 基线:`uv run pytest src/jxl/vdt/ -q` 确认全绿(迁移前)。
- [ ] `OcvDecoder` 重构:内部委托 `VideoReader`(构造时 `DecodeCfg.fps` → `VideoReader(sample_fps=...)`);`__iter__`/`fps`/`size` 转发。保持 `_make_synthetic_video`、`DecodeError` 原处。公共签名(构造参数、iter yield)不变 → pipeline/cli 调用方无需改。
- [ ] `render_video`:用 `with VideoWriter(out, tracks.fps, (width, height)) as w:` + `w.write(canvas)` 替换内联 `cv2.VideoWriter`(含 isOpened 检查、finally release)。
- [ ] `uv run pytest src/jxl/vdt/ -q` → **必须仍全绿**(行为不变是硬约束)。
- [ ] mypy + ruff(vdt 两个文件)。
- [ ] Commit: `refactor(vdt): migrate OcvDecoder/render_video to shared video IO`。

### Task 3: vtag overlay.py（Functional Core）

**Files:**
- Create: `src/jxl/vtag/__init__.py`
- Create: `src/jxl/vtag/overlay.py`(inline 测试)

**Consumes:** numpy、PIL(`ImageFont`/`ImageDraw`/`Image`)、cv2(`cvtColor`)。
**Produces:** `EventSpec`、`TagOpts`、`parse_event`、`draw_tags`(spec §6.1)。

**Steps:**
- [ ] `parse_event` 测试:`'打架,2.0-5.0'` → `EventSpec('打架',2.0,5.0)`;非法(无逗号/非数/`起>=止`/负值)→ `ValueError`。
- [ ] `draw_tags` 测试:空 `names` → `np.array_equal(返回, 原帧)`;有事件 → 右下角 ROI 出红(`R>200 & G<50 & B<50`)且左上角 ROI 不变;多事件 → 两标签 y 区间不重叠(竖排)。
- [ ] 实现 `EventSpec`/`TagOpts`(`frozen=True, slots=True`)、`parse_event`、`draw_tags`(cv2 BGR→PIL RGB→`ImageDraw.text(stroke_width, stroke_fill)`→转回;空列表直接返回原帧)。
- [ ] `uv run pytest src/jxl/vtag/overlay.py -q` + mypy + ruff。
- [ ] Commit: `feat(vtag): add overlay functional core (EventSpec/draw_tags)`。

### Task 4: vtag cli.py + console script

**Files:**
- Create: `src/jxl/vtag/cli.py`(inline 测试)
- Modify: `pyproject.toml`(`[project.scripts]` 加 `vtag = "jxl.vtag.cli:app"`)

**Consumes:** Task 1(`VideoReader`/`VideoWriter`)、Task 3(`overlay`)。
**Produces:** `vtag` 命令(spec §6.2/§7/§8)。

**Steps:**
- [ ] CLI 测试:`parse_event` 集成;非法/缺失 `--event` → `BadParameter`;视频/字体不存在 → `BadParameter`;端到端(合成视频 + 2 事件 → 可读 mp4;事件时段内帧 vs 时段外帧像素有差)。
- [ ] 实现 `main`:参数(spec §7)、`events=[parse_event(e) for e in event]`、默认 `out=<input>_tagged.<ext>`、默认字体路径 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`(`--font` 覆盖)、`ImageFont.truetype` 循环外加载一次、`VideoReader`/`VideoWriter` 循环 + `draw_tags`。
- [ ] No Silent Degradation:字体 `truetype` 失败 / `VideoWriter` 打不开 / 视频无帧 → 报错退出,不回退。
- [ ] 注册 console script;`uv run vtag --help` 正常;`uv run vtag <合成视频> --event '测试,0.5-1.5' --out /tmp/x.mp4` 端到端通。
- [ ] mypy + ruff。
- [ ] Commit: `feat(vtag): add CLI + console script`。

---

## 依赖与执行顺序

- Task 1 → Task 2(2 依赖共享层);Task 1 → Task 4;Task 3 → Task 4;Task 3 独立于 1/2。
- 可并行序:1 → (2 ‖ 3) → 4。
- 硬约束:Task 2 后 vdt 全测试必须仍绿(行为不变)。
