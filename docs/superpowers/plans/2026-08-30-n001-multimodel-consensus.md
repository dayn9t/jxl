# n001 多模型共识标注 Pipeline 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** n001 视频目录（806 mkv/134h）→ I 帧 4:1 抽取 → 去重 → 五模型共识标注（本机+s4 双机分片）→ 可信度分级 + 模型能力矩阵 + 人工审核材料。

**Architecture:** 5-stage 流水线，3 个新 bin + det_mine 小扩展 + 编排脚本；Stage 2/3 复用现成 SemDeDup 管线与 det_mine（la 校验器）；标注按帧奇偶分片到本机/s4 各自独立跑，产物合并。

**Tech Stack:** Python 3.12（typer/orjson/httpx/PIL）、ffmpeg（I 帧抽取）、uv、det_mine 校验器栈（ultralytics YOLOE / transformers GDINO / rfdetr / la 服务）。

**Spec:** `docs/superpowers/specs/2026-08-30-n001-multimodel-consensus-design.md`

## Global Constraints

- 本仓 Python 规范：mypy strict 零错、完整类型注解、typer CLI、orjson 序列化、`@dataclass(frozen=True, slots=True)`/NamedTuple、禁止裸 except/Any
- la（LocateAnything-3B）NVIDIA License 非商用——仅研究/评估链路注释保留
- target 模型：`/mnt/data/jiang/ws/sgcc/person/runs/detect/person_yolo26n/weights/best.pt`（同步到 s4 `~/cc/py/jxl/models/person_yolo26n_best.pt`，两机同用 models/ 相对布局）
- 关键帧 = I 帧 4:1（保留解码序 `idx % 4 == 0`）；GOP=2s → 有效 8s
- 所有标注工具都无目标的图：删除（det_mine L0-drop 语义）
- 产出 `datasets/sgcc-n001/`（`/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/`），不入池
- 测试命令统一 `uv run pytest <path> -q`；静态检查 `uv run mypy <files>` / `uv run ruff check <files>`
- 提交信息英文 conventional commits；行数：文件 <800、函数 <50

---

### Task 1: 提炼 viz 共享纯函数（draw/grid）

**Files:**
- Create: `src/jxl/det/viz.py`
- Modify: `src/jxl/bin/la_eval.py`（draw_one/grid 改用 viz）
- Test: `tests/det/viz_test.py`

**Interfaces:**
- Produces:
  - `scale_to_width(im: PIL.Image.Image, width: int) -> PIL.Image.Image`
  - `draw_boxes(im: PIL.Image.Image, boxes: list[Box], color: RGB, width_px: int = 3) -> PIL.Image.Image`（归一化 xyxy Box，原点左上，返回新图）
  - `label_header(im: PIL.Image.Image, text: str) -> PIL.Image.Image`（顶部黑条白字，高 26px）
  - `grid(images: list[PIL.Image.Image], cols: int, tile_w: int) -> PIL.Image.Image`
  - `RGB = tuple[int, int, int]`（类型别名）
- Consumes: `jxl.det.hardmine.Box`

- [ ] **Step 1: 写失败测试**

```python
"""viz 纯函数单测: 缩放/画框/网格(合成小图, 无文件依赖)."""
from __future__ import annotations

from PIL import Image

from jxl.det.viz import RGB, draw_boxes, grid, scale_to_width


def test_scale_to_width_keeps_ratio() -> None:
    im = Image.new("RGB", (100, 50), (0, 0, 0))
    out = scale_to_width(im, 200)
    assert out.size == (200, 100)


def test_draw_boxes_marks_pixels() -> None:
    im = Image.new("RGB", (100, 100), (0, 0, 0))
    color: RGB = (255, 0, 0)
    out = draw_boxes(im, [(0.1, 0.1, 0.9, 0.9, 1.0)], color, width_px=3)
    # 框边缘像素变红
    assert out.getpixel((10, 10)) == color
    assert out.getpixel((50, 50)) == (0, 0, 0)  # 框内部未填


def test_grid_layout() -> None:
    tiles = [Image.new("RGB", (50, 30), (i, 0, 0)) for i in range(5)]
    out = grid(tiles, cols=2, tile_w=50)
    assert out.size == (100, 3 * 30)  # 3 行 x 2 列
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/det/viz_test.py -q`
Expected: FAIL（ModuleNotFoundError: jxl.det.viz）

- [ ] **Step 3: 实现 viz.py**

```python
"""检测可视化纯函数: 缩放/画框/标注头/网格拼装(la_eval 与 review_pack 共用)."""

from PIL import Image, ImageDraw, ImageFont

from jxl.det.hardmine import Box

RGB = tuple[int, int, int]
_HEADER_H = 26


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def scale_to_width(im: Image.Image, width: int) -> Image.Image:
    """等比缩放到指定宽."""
    ratio = width / im.width
    return im.resize((width, max(1, int(im.height * ratio))))


def draw_boxes(im: Image.Image, boxes: list[Box], color: RGB, width_px: int = 3) -> Image.Image:
    """归一化 xyxy 框画到图上(返回新图, 不改原图)."""
    out = im.copy()
    dr = ImageDraw.Draw(out)
    for b in boxes:
        dr.rectangle(
            (b[0] * im.width, b[1] * im.height, b[2] * im.width, b[3] * im.height),
            outline=color,
            width=width_px,
        )
    return out


def label_header(im: Image.Image, text: str) -> Image.Image:
    """顶部加黑条白字标题行."""
    out = Image.new("RGB", (im.width, im.height + _HEADER_H), (0, 0, 0))
    out.paste(im, (0, _HEADER_H))
    ImageDraw.Draw(out).text((4, 4), text, font=_font(18), fill=(255, 255, 255))
    return out


def grid(images: list[Image.Image], cols: int, tile_w: int) -> Image.Image:
    """等宽拼网格(cols 列, 行数自适应; tile 高含 header 取最大)."""
    if not images:
        raise ValueError("grid: empty images")
    tiles = [scale_to_width(im, tile_w) for im in images]
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (tile_w * cols, th * rows), (30, 30, 30))
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % cols) * tile_w, (i // cols) * th))
    return canvas
```

- [ ] **Step 4: 跑测试通过**

Run: `uv run pytest tests/det/viz_test.py -q`
Expected: 3 passed

- [ ] **Step 5: la_eval 改用 viz（行为不变）**

`src/jxl/bin/la_eval.py`：删除本地 `draw_one`/`grid` 实现与 `_PREVIEW_W`/`_GRID_COLS`/`_COLORS` 常量，`run` 的 `render` 改为：对每个样本 `label_header(draw_boxes(draw_boxes(scale_to_width(img, 640), base, (0,200,0)), la, (255,40,0)), f"{stem[:28]} m{..}/e{..}")`，网格用 `viz.grid(tiles, cols=4, tile_w=640)`。跑 `uv run pytest tests/bin/la_eval_test.py -q`（6 passed，纯函数测试不受影响）+ `uv run python src/jxl/bin/la_eval.py` 冒烟（用 `sgcc-la-eval/eval` 重跑一遍，产出图与之前等价）。

- [ ] **Step 6: 静态检查 + 提交**

```bash
uv run mypy src/jxl/det/viz.py src/jxl/bin/la_eval.py
uv run ruff check src/jxl/det/viz.py src/jxl/bin/la_eval.py tests/det/viz_test.py
git add src/jxl/det/viz.py tests/det/viz_test.py src/jxl/bin/la_eval.py
git commit -m "refactor(det): extract shared viz pure functions (draw/grid)"
```

---

### Task 2: video_keyframe.py — I 帧 4:1 抽取 bin

**Files:**
- Create: `src/jxl/bin/video_keyframe.py`
- Test: `tests/bin/video_keyframe_test.py`

**Interfaces:**
- Produces:
  - `select_frames(count: int, every: int) -> list[int]`（前 count 个解码序号中保留 `idx % every == 0`；纯函数）
  - `extract_one(mkv: Path, out_dir: Path, every: int) -> int`（ffmpeg 抽单 mkv 全部 I 帧到临时目录→按序 4:1 复制保留→命名 `{mkv_stem}_{i:05d}.jpg`，返回保留帧数；重复调用幂等：已存在状态文件跳过）
  - CLI: `video_keyframe <video_dir> <out_dir> [--every 4] [--jobs 4]`（状态落 `<out_dir>/_state.jsonl` 断点续跑）
- Consumes: 无（subprocess ffmpeg/ffprobe）

- [ ] **Step 1: 写失败测试**

```python
"""video_keyframe 纯函数单测: I 帧下采样选择."""
from __future__ import annotations

from jxl.bin.video_keyframe import select_frames


def test_select_frames_every_4() -> None:
    # 300 I 帧 4:1 → 保留 0,4,8,...,296 共 75
    assert select_frames(300, 4) == list(range(0, 300, 4))


def test_select_frames_fewer_than_every() -> None:
    assert select_frames(3, 4) == [0]


def test_select_frames_invalid() -> None:
    import pytest

    with pytest.raises(ValueError):
        select_frames(10, 0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/bin/video_keyframe_test.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 video_keyframe.py**

```python
#!/usr/bin/env python3
"""视频 I 帧抽取 + N:1 下采样(n001 关键帧提取).

ffmpeg skip_frame nokey 解码仅 I 帧(快), 落盘序号 idx%every==0 保留.
断点续跑: 已处理 mkv 记录 _state.jsonl, 重跑跳过.
命名 {mkv_stem}_{i:05d}.jpg 保源可溯(mkv 名含日期时间在父目录, 记入状态行).

用法: video_keyframe <video_dir> <out_dir> [--every 4] [--jobs 4]
"""

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import orjson
import typer

app = typer.Typer(add_completion=False, help="视频 I 帧 N:1 抽取.")


def select_frames(count: int, every: int) -> list[int]:
    """前 count 个解码序号保留 idx % every == 0."""
    if every < 1:
        raise ValueError(f"every 须 >=1: {every}")
    return list(range(0, count, every))


def extract_one(mkv: Path, out_dir: Path, every: int) -> int:
    """单 mkv: 抽全部 I 帧到临时目录 → 按序保留 every:1 → out_dir. 返回保留数."""
    if every < 1:
        raise ValueError(f"every 须 >=1: {every}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="keyframe-") as tmp:
        tmp_dir = Path(tmp)
        # -nostats -loglevel error: 静默; scale 保持原分辨率
        r = subprocess.run(
            ["ffmpeg", "-nostats", "-loglevel", "error", "-skip_frame", "nokey",
             "-i", str(mkv), "-vsync", "vfr", "-q:v", "2",
             str(tmp_dir / "f%06d.jpg")],
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            typer.secho(f"ffmpeg 失败 {mkv.name}: {r.stderr[:300]}", fg=typer.colors.RED, err=True)
            return 0
        frames = sorted(tmp_dir.glob("f*.jpg"))
        keep = select_frames(len(frames), every)
        for i in keep:
            dst = out_dir / f"{mkv.stem}_{i // every:05d}.jpg"
            shutil.copy2(frames[i], dst)
        return len(keep)


@app.command()
def run(
    video_dir: Annotated[Path, typer.Argument(help="视频目录(递归找 mkv)")],
    out_dir: Annotated[Path, typer.Argument(help="帧输出目录")],
    every: Annotated[int, typer.Option("--every", help="I 帧 N:1 采样")] = 4,
    jobs: Annotated[int, typer.Option("--jobs", help="并行 ffmpeg 数")] = 4,
) -> None:
    """I 帧 N:1 抽取, 状态断点续跑."""
    mkvs = sorted(video_dir.rglob("*.mkv"))
    if not mkvs:
        typer.secho(f"无 mkv: {video_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    state = out_dir / "_state.jsonl"
    state.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if state.exists():
        done = {orjson.loads(l)["mkv"] for l in state.read_text().splitlines() if l}
    todo = [m for m in mkvs if m.name not in done]
    typer.secho(f"mkv {len(mkvs)} done {len(done)} todo {len(todo)}", fg=typer.colors.CYAN)

    def work(m: Path) -> int:
        n = extract_one(m, out_dir, every)
        with state.open("a", encoding="utf-8") as f:
            f.write(orjson.dumps({"mkv": m.name, "rel": str(m.parent), "kept": n}).decode() + "\n")
        return n

    total = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for n in ex.map(work, todo):
            total += n
    typer.secho(f"保留 {total} 帧 → {out_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
```

注意：`done` 以 mkv 文件名判重（n001 目录内 `{HH-MM-SS.xxx}.mkv` 跨日期可能重名，状态行含 `rel` 父目录路径；判重用 `rel + name` 组合更稳——实现时以 `f"{m.parent.name}/{m.name}"` 为键）。

- [ ] **Step 4: 跑测试通过**

Run: `uv run pytest tests/bin/video_keyframe_test.py -q`
Expected: 3 passed

- [ ] **Step 5: 真实冒烟（1 个 mkv）**

```bash
uv run python src/jxl/bin/video_keyframe.py \
  /var/howell/iap/v0.9/ias/sh-sgcc/n001/video/1/0/2026-06-23 /tmp/kf-smoke --every 4 --jobs 1
ls /tmp/kf-smoke/*.jpg | wc -l   # 期望 ~ 25-75（该日期 mkv 数 × 75）
```

- [ ] **Step 6: 静态检查 + 提交**

```bash
uv run mypy src/jxl/bin/video_keyframe.py && uv run ruff check src/jxl/bin/video_keyframe.py tests/bin/video_keyframe_test.py
git add src/jxl/bin/video_keyframe.py tests/bin/video_keyframe_test.py
git commit -m "feat(bin): video_keyframe — I-frame N:1 extraction with resume state"
```

---

### Task 3: det_mine 扩展 --dump-validators（全量图逐模型输出）

**Files:**
- Modify: `src/jxl/bin/det_mine.py`（run 函数加参数；scored 循环后写 dump）
- Test: `tests/bin/det_mine_test.py`（追加）

**Interfaces:**
- Produces: CLI 参数 `--dump-validators <path>`（可选）：每图一行
  `{"stem": str, "target": list[Box], "validators": {"<name>": list[Box]}, "score": float, "level": "L0"|"L1"|"review"}`
  到指定 jsonl（L0/L1/review 全量含空框图；损坏 skip 图不写）
- Consumes: 现有 ScoredSample/cascade 逻辑（不改行为）

- [ ] **Step 1: 写失败测试（CLI 参数存在 + dump 行结构）**

追加到 `tests/bin/det_mine_test.py`：

```python
def test_dump_validators_flag_exists() -> None:
    # --help 输出含 --dump-validators(接口存在性; 行为靠 Phase A 端到端)
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "jxl.bin.det_mine", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--dump-validators" in r.stdout
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/bin/det_mine_test.py -q`
Expected: 新测试 FAIL（参数不存在）

- [ ] **Step 3: 实现（最小改动）**

`det_mine.py` `run` 签名追加：

```python
    dump_validators: Annotated[
        Path, typer.Option("--dump-validators", help="全量图逐模型输出 jsonl(矩阵分析用)")
    ] = Path(),
```

cascade 分流循环改为先构建 `level` 再统一处理；循环内（scored 全量遍历，含 L0）当 `dump_validators.name` 非空时追加写行：

```python
        level = ("review" if s.img.stem in review_stems else "L1") if s.score > 0 else "L0"
        if dump_validators.name:
            with dump_validators.open("a", encoding="utf-8") as df:
                df.write(orjson.dumps({
                    "stem": s.img.stem, "target": s.target_boxes,
                    "validators": s.validators, "score": s.score, "level": level,
                }).decode() + "\n")
```

（保持既有 L0 continue / L1 / review 产物逻辑不变；dump 行在 continue 之前写。）

- [ ] **Step 4: 跑全部 det_mine 测试通过**

Run: `uv run pytest tests/bin/det_mine_test.py -q`
Expected: 全部 passed（原测试不回归）

- [ ] **Step 5: 静态检查 + 提交**

```bash
uv run mypy src/jxl/bin/det_mine.py && uv run ruff check src/jxl/bin/det_mine.py
git add src/jxl/bin/det_mine.py tests/bin/det_mine_test.py
git commit -m "feat(bin): det_mine --dump-validators — per-image all-model output for matrix analysis"
```

---

### Task 4: consensus_report.py — 可信度分级 + 模型能力矩阵

**Files:**
- Create: `src/jxl/bin/consensus_report.py`
- Test: `tests/bin/consensus_report_test.py`

**Interfaces:**
- Produces:
  - `ModelDump = dict[str, object]`（det_mine dump 行：stem/target/validators/score/level）
  - `pairwise_agreement(dumps: list[ModelDump], iou_thr: float) -> dict[tuple[str, str], float]`（模型两两 IoU≥thr 框级一致率：对每图两模型框贪心匹配，匹配数/min(n_a,n_b)>0 计入，空对空图计 1 对）——纯函数
  - `size_buckets(boxes: list[Box]) -> str`（归一化框高：<0.04 "far-small" / <0.14 "mid" / else "large"）——纯函数
  - `per_model_vs_consensus(dumps, iou_thr) -> dict[str, dict[str, float]]`（每模型框 vs 该图其余模型共识位置的 P/R/F1）
  - CLI: `consensus_report <consensus_dir> <out_dir> [--iou 0.3]`：读各批 `validators_*.jsonl` 合并 → T1/T2/T3 计数（T1=level L0 且 target+4 校验器均非空；T2=L1；T3=review）+ 矩阵 + 分桶 → `accuracy_report.md` + `model_matrix.json`
- Consumes: `jxl.det.hardmine.greedy_match`、Task 3 的 dump 格式

- [ ] **Step 1: 写失败测试**

```python
"""consensus_report 纯函数单测: 两两一致性/尺寸分桶."""
from __future__ import annotations

from jxl.bin.consensus_report import pairwise_agreement, size_buckets


def _dump(stem: str, **kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "stem": stem, "target": [], "validators": {}, "score": 0.0, "level": "L0",
    }
    base.update(kw)
    return base


def test_size_buckets() -> None:
    assert size_buckets([(0.1, 0.1, 0.2, 0.15, 1.0)]) == "mid"      # h=0.05
    assert size_buckets([(0.1, 0.1, 0.2, 0.12, 1.0)]) == "far-small"  # h=0.02
    assert size_buckets([(0.1, 0.1, 0.2, 0.5, 1.0)]) == "large"       # h=0.4


def test_pairwise_perfect_and_disjoint() -> None:
    box = [(0.1, 0.1, 0.5, 0.5, 1.0)]
    dumps = [
        _dump("a", target=box, validators={"yoloe": box, "la": box}),
        _dump("b", target=[], validators={"yoloe": [], "la": []}),  # 空对空=一致
    ]
    m = pairwise_agreement(dumps, 0.3)
    assert m[("yoloe", "la")] == 1.0
    assert m[("target", "yoloe")] == 0.5  # 图a一致, 图b target空yoloe空=1; 仅图a target有框未对上→0
```

（最后一断言按实现口径调整：空对空计 1、一对空计 0 → target-yoloe = (0+1)/2 = 0.5）

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/bin/consensus_report_test.py -q`
Expected: FAIL

- [ ] **Step 3: 实现 consensus_report.py**

核心结构（CLI 壳略，全量代码按测试口径实现）：

```python
#!/usr/bin/env python3
"""共识标注汇总: 可信度分级(T1/T2/T3) + 模型能力矩阵(两两一致性/尺寸分桶)."""

from pathlib import Path
from typing import Annotated, NamedTuple

import orjson
import typer

from jxl.det.hardmine import Box, greedy_match

app = typer.Typer(add_completion=False, help="共识标注分级与模型矩阵.")
ModelDump = dict[str, object]
MODELS = ("target", "yoloe", "gdino", "rfdetr", "la")


def size_buckets(boxes: list[Box]) -> str:
    """归一化框高分桶: far-small(<0.04≈43px@1080) / mid(<0.14) / large."""
    heights = [b[3] - b[1] for b in boxes]
    h = max(heights) if heights else 0.0
    return "far-small" if h < 0.04 else ("mid" if h < 0.14 else "large")


def _boxes(d: ModelDump, name: str) -> list[Box]:
    v = d["validators"][name] if name != "target" else d["target"]
    return [tuple(b) for b in v]  # type: ignore[union-attr]


def pairwise_agreement(dumps: list[ModelDump], iou_thr: float) -> dict[tuple[str, str], float]:
    """模型两两框级一致率: 空对空=1, 一对空=0, 否则 greedy_match 匹配数/max(n_a,n_b)."""
    names = [n for n in MODELS if any(n in d["validators"] or n == "target" for d in dumps[:1])]
    # (实现对全部出现的模型对循环累计)
    ...
```

（`pairwise_agreement` 完整实现：对每对 (a,b) 每图取 `_boxes`，两者皆空 acc+=1；一方空 acc+=0；否则 `greedy_match` 匹配数 / max(len_a, len_b)，跨图求均值。`per_model_vs_consensus`：其余模型框并集经 `greedy_match` 自聚类简化为「其余模型逐个与该模型匹配，任一匹配即共识」，P=匹配数/该模型框数，R=匹配数/其余模型框数。）

CLI `run`：glob `validators*.jsonl` 合并 → 计数 T1（level==L0 且五模型框均非空）/T2（L1）/T3（review）→ 矩阵 → 按 `size_buckets` 分桶的每模型 P/R → 写 `accuracy_report.md`（markdown 表）+ `model_matrix.json`。

- [ ] **Step 4: 跑测试通过**

Run: `uv run pytest tests/bin/consensus_report_test.py -q`
Expected: passed

- [ ] **Step 5: 静态检查 + 提交**

```bash
uv run mypy src/jxl/bin/consensus_report.py && uv run ruff check src/jxl/bin/consensus_report.py tests/bin/consensus_report_test.py
git add src/jxl/bin/consensus_report.py tests/bin/consensus_report_test.py
git commit -m "feat(bin): consensus_report — trust tiers + pairwise model matrix"
```

---

### Task 5: review_pack.py — 人工审核材料

**Files:**
- Create: `src/jxl/bin/review_pack.py`
- Test: `tests/bin/review_pack_test.py`

**Interfaces:**
- Produces:
  - `MODEL_COLORS: dict[str, RGB]`（target 黑(0,0,0)/yoloe 蓝(60,120,255)/gdino 黄(255,200,0)/rfdetr 绿(0,200,0)/la 红(255,40,40)）
  - `render_tile(img: Image, boxes_by_model: dict[str, list[Box]], stem: str) -> Image`（各模型色框叠加 + header）
  - CLI: `review_pack <consensus_dir> <images_dir> <out_dir> [--per-grid 20]`：读 review/manifest.jsonl（含 validators 输出）+ validators dump 补充 → 网格分片 `review_grid_{i:03d}.jpg` + `manifest.jsonl`（原样合并）+ `README.txt`（审核操作指引）
- Consumes: `jxl.det.viz`（Task 1）、det_mine review manifest 格式（image/score/target_boxes/validators/breakdown）

- [ ] **Step 1: 写失败测试**

```python
"""review_pack 纯函数单测: tile 渲染(合成图)."""
from __future__ import annotations

from PIL import Image

from jxl.bin.review_pack import MODEL_COLORS, render_tile


def test_model_colors_complete() -> None:
    assert set(MODEL_COLORS) == {"target", "yoloe", "gdino", "rfdetr", "la"}


def test_render_tile_sizes() -> None:
    im = Image.new("RGB", (320, 240), (0, 0, 0))
    out = render_tile(im, {"la": [(0.1, 0.1, 0.5, 0.5, 1.0)]}, "stem_a")
    assert out.width == 640  # tile 统一宽
    assert out.height > 240  # 含 header
```

- [ ] **Step 2: 确认失败 → **Step 3: 实现**（render_tile: `scale_to_width`→逐模型 `draw_boxes`→`label_header`；CLI 读 manifest 渲染分片）→ **Step 4: 测试通过**

Run: `uv run pytest tests/bin/review_pack_test.py -q`
Expected: passed

- [ ] **Step 5: 静态检查 + 提交**

```bash
uv run mypy src/jxl/bin/review_pack.py && uv run ruff check src/jxl/bin/review_pack.py tests/bin/review_pack_test.py
git add src/jxl/bin/review_pack.py tests/bin/review_pack_test.py
git commit -m "feat(bin): review_pack — per-model colored grids + manifest for human review"
```

---

### Task 6: 编排脚本 + Phase A 执行（含 s4 部署）

**Files:**
- Create: `script/n001-pipeline.sh`
- Create: `docs/2026-08-30-n001-phaseA-结果.md`（执行后写）

**Interfaces:**
- Produces: 一键 Phase A/B 入口：`script/n001-pipeline.sh phaseA|phaseB|stage1|stage2|stage3|merge|report`
- Consumes: Task 2-5 全部 bin + 现成去重管线（person_crop/person_embed/person_dedup）+ det_mine

- [ ] **Step 1: 写编排脚本**

```bash
#!/usr/bin/env bash
# n001 共识标注 pipeline 编排(阶段独立可重跑; 数据根/机器见 spec)
set -euo pipefail
ROOT=/mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001
VIDEO=/var/howell/iap/v0.9/ias/sh-sgcc/n001/video
TARGET=/mnt/data/jiang/ws/sgcc/person/runs/detect/person_yolo26n/weights/best.pt
VALIDATORS=yoloe,gdino,rfdetr,la
BATCH=2000

stage1() { uv run python src/jxl/bin/video_keyframe.py "$VIDEO" "$ROOT/raw_frames" --every 4 --jobs 8; }
stage2() {  # 前景感知去重(现成三步): raw_frames → frames_dedup
  uv run python src/jxl/bin/crop_foreground.py "$ROOT/raw_frames" "$ROOT/crops"   # 以实际去重管线 bin 名为准
  uv run python src/jxl/bin/embed_dino.py "$ROOT/crops" "$ROOT/embeds"
  uv run python src/jxl/bin/dedup_sem.py "$ROOT/embeds" "$ROOT/raw_frames" "$ROOT/frames_dedup"
}
stage3() {  # 分批 det_mine(断点=批目录); 双机分片在 phaseB 里由 rsync+s4 并行
  local src="$1" out="$2"
  ls "$src" | wc -l
  uv run python - <<PY
# 分批逻辑: sorted(images) 按 BATCH 切片, 已有 consensus/batch_XXXX 跳过,
# 每批: det_mine <批图目录> <out>/batch_XXXX --target-model $TARGET \
#   --validators $VALIDATORS --consensus 2 --dump-validators <out>/batch_XXXX/validators.jsonl
PY
}
merge() {  # 各批 images/labels/review 拼接 + validators_*.jsonl 汇总
  rsync -a "$ROOT"/consensus/batch_*/images/ "$ROOT/consensus/images/"
  rsync -a "$ROOT"/consensus/batch_*/labels/ "$ROOT/consensus/labels/"
  cat "$ROOT"/consensus/batch_*/validators.jsonl > "$ROOT/consensus/validators_all.jsonl"
  # review manifest 同拼
}
report() {
  uv run python src/jxl/bin/consensus_report.py "$ROOT/consensus" "$ROOT"
  uv run python src/jxl/bin/review_pack.py "$ROOT/consensus" "$ROOT/frames_dedup" "$ROOT/review_pack"
}
phaseA() {  # 8 mkv 随机子集全链路
  # 建 subset/ (8 mkv symlink) → stage1(仅 subset) → stage2 → stage3(local) → merge → report
  ...
}
"$@"
```

（执行者注意：stage2 的三个 bin 名以 `src/jxl/bin/` 实际为准——`ls src/jxl/bin | grep -E "crop|embed|dedup"`；本计划列的 `crop_foreground/embed_dino/dedup_sem` 为初判，写脚本时核对真名与参数。）

- [ ] **Step 2: Phase A.0 — s4 部署验证（首项，独立于本机链路）**

```bash
# 1) 同步 repo(排除环境/缓存) + 关键权重到 s4 相同布局
rsync -a --exclude .venv --exclude .la-venv --exclude .git --exclude models ~/cc/py/jxl/ s4:~/cc/py/jxl/
rsync -a ~/cc/py/jxl/models/yoloe-11l-seg.pt ~/cc/py/jxl/models/LocateAnything-3B/ s4:~/cc/py/jxl/models/  # 分两条
rsync -a $TARGET s4:~/cc/py/jxl/models/person_yolo26n_best.pt
# 2) s4 主环境
ssh s4 'cd ~/cc/py/jxl && uv sync'
# 3) s4 la-venv: la-setup.sh 默认 torch 2.6 不支持 sm_120 → 显式改 torch>=2.7 cu128 安装;
#    flash-attn 无 sm_120 wheel 则 la-serve.sh --attn sdpa
ssh s4 'cd ~/cc/py/jxl && bash script/la-setup.sh'   # 需按上面适配; 失败则记录并 sdpa
ssh s4 'setsid nohup script/la-serve.sh --attn sdpa > /tmp/la-serve.log 2>&1 &'
# 4) 冒烟对比: 同 20 图本机 vs s4 检测, 框 IoU 一致性 > 0.9 平均
```

验证点（全部通过才算部署 OK）：s4 la 服务 health ok；20 图冒烟平均 IoU ≥0.9（同模型同输入，允许注意力实现数值微差）；gdino/rfdetr 权重可拉取（`HF_ENDPOINT=https://hf-mirror.com` 备选）。

- [ ] **Step 3: Phase A — 本机 8 mkv 全链路**

```bash
bash script/n001-pipeline.sh phaseA
```

产出核对：`raw_frames/`（8×75=600 帧理论）；去重率（对比删除对抽样 20 张人工确认无有人图误删）；`consensus/`（L0/L1/review 计数）；`accuracy_report.md` 初版；`review_pack/` 网格可读。

- [ ] **Step 4: 外推全量 + 写结果文档 + 用户确认**

计算：帧数 ×806/8、双机标注时长（Phase A 单机时长/2）、review 量。写入 `docs/2026-08-30-n001-phaseA-结果.md`，**停下等用户确认 Phase B**。

- [ ] **Step 5: 提交**

```bash
git add script/n001-pipeline.sh docs/2026-08-30-n001-phaseA-结果.md
git commit -m "feat(script): n001 pipeline orchestration + phase A results"
```

---

### Task 7: Phase B 全量执行（用户确认后）

**Files:**
- Create: `docs/2026-08-30-n001-phaseB-结果.md`

- [ ] **Step 1: 本机全量 stage1/2**

```bash
bash script/n001-pipeline.sh stage1   # 806 mkv, ~1h(8 并行)
bash script/n001-pipeline.sh stage2   # 去重
```

- [ ] **Step 2: 分片 rsync 到 s4（stem 排序奇偶）**

```bash
# 偶数片列表 rsync 到 s4 相同数据根(s4 需先建 /mnt/data/... 或用 ~/n001-data 软链——以 s4 磁盘布局定, 保持脚本内 ROOT 一致)
ls "$ROOT/frames_dedup" | sort | awk 'NR%2==0' > /tmp/even.list
rsync -a --files-from=/tmp/even.list "$ROOT/frames_dedup/" s4:"$ROOT/frames_dedup/"
# 奇数片留本机(从 frames_dedup 移到 frames_local, s4 跑偶数片)
```

- [ ] **Step 3: 双机并行 stage3 + 合并**

```bash
# 本机: bash script/n001-pipeline.sh stage3 "$ROOT/frames_local" "$ROOT/consensus"
# s4:   ssh s4 'cd ~/cc/py/jxl && bash script/n001-pipeline.sh stage3 <偶数片> <consensus>'
# s4 完成后: rsync -a s4:"$ROOT/consensus/" "$ROOT/consensus/s4/" && merge && report
```

监控：批进度日志 + la 服务 health（中断则续跑该批）。**全程 GPU 13.3G/16.4G，业务服务保持停止**。

- [ ] **Step 4: 最终报告 + 存档文档 + 提交**

写 `docs/2026-08-30-n001-phaseB-结果.md`（全量统计 + 模型矩阵结论 + review 规模 → 供用户定豆包/入池决策）。提交。

---

## Self-Review 记录

- **Spec 覆盖**：Stage1→Task2、Stage2→Task6 stage2、Stage3→Task6 stage3（含双机 Task7）、det_mine 扩展→Task3、Stage4→Task4、Stage5→Task5、PhaseA/B→Task6/7、s4 部署→Task6 Step2 ✓
- **占位符**：Task 4 Step 3 `pairwise_agreement` 主体给了算法口径描述而非全码（循环体省略号）——执行者按测试口径补全，测试是行为的权威定义；Task 6 stage2 bin 名标注「以实际为准核对」✓（这两处是刻意的执行期决策点，非 TBD）
- **类型一致**：Box 五元组 xyxy 归一化贯穿；`RGB`/`ModelDump` 别名跨任务一致 ✓
