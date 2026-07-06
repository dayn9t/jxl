# Person 难例挖掘 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从监控 mkv 抽关键帧 → person.pt + YOLOE 双检测交叉比对 → 分歧样本自动标注（YOLO 格式）回灌训练集。

**Architecture:** Functional Core（`src/jxl/det/hardmine.py` 纯函数：框级 IoU 匹配 + 难例分类 + YOLO 标注生成，mypy strict + 充分单测）+ Imperative Shell（`src/jxl/bin/person_mine.py` 双检测 CLI + `src/jxl/bin/mkv_keyframes.py` I-frame 抽帧 CLI）。

**Tech Stack:** Python 3.12 / ultralytics（YOLO+YOLOE 推理）/ typer（CLI）/ orjson（report）/ ffmpeg（抽帧，系统级）。

## Global Constraints

- Python `~=3.12.0`；lib 代码 mypy strict（`warn_return_any`/`warn_unused_ignores`/`no_implicit_optional`），`src/jxl/bin/` 被 mypy `exclude` 不强制。
- ruff `select=ALL` + 全局 ignore 清单（见 `pyproject.toml`）；`src/jxl/bin/*.py` 有 per-file-ignores（ANN/T201/PLR0913 等）。
- 测试跑法：`uv run pytest`（j-python 规则：禁止 subprocess 调 pytest）。
- 库锁定（pyproject 已声明）：typer / pydantic / orjson / ultralytics>=8.4.65 / pillow>=10.0；系统级 ffmpeg。
- No Silent Degradation：模型文件/ffmpeg/空目录缺失即报错退出，不静默回退。
- 提交信息英文；attribution 全局禁用（不加 Co-Authored-By）。
- YOLO 标注单类 `cls=0`（person），与 `person.pt` 训练 `names:{0:person}` 对齐。

---

## File Structure

| 文件 | 职责 | mypy |
|------|------|------|
| Create `src/jxl/det/hardmine.py` | Functional Core：`Box`/`SampleClass`/`xyxy_iou`/`greedy_match`/`to_yolo_label`/`classify_sample` | strict |
| Create `tests/det/hardmine_test.py` | 纯函数单测（决策表全覆盖） | strict |
| Create `src/jxl/bin/mkv_keyframes.py` | I-frame 抽帧 CLI（ffmpeg） | 排除 |
| Create `src/jxl/bin/person_mine.py` | 双检测 + 比对 + 输出 CLI（调用 hardmine） | 排除 |

---

### Task 1: hardmine.py — 数据模型 + xyxy_iou + greedy_match

**Files:**
- Create: `src/jxl/det/hardmine.py`
- Test: `tests/det/hardmine_test.py`

**Interfaces:**
- Produces: `Box`（`tuple[float,float,float,float,float]` 归一化 xyxy+conf）、`xyxy_iou(a, b) -> float`、`greedy_match(boxes_a, boxes_b, iou_thr) -> tuple[matched, unmatched_a, unmatched_b]`

- [ ] **Step 1: 写失败测试（数据模型 + xyxy_iou + greedy_match）**

Create `tests/det/hardmine_test.py`:
```python
"""hardmine 纯函数单测: 框级 IoU 匹配 + 难例分类 + YOLO 标注生成。"""
from __future__ import annotations

from jxl.det.hardmine import (
    greedy_match,
    xyxy_iou,
)

IOU_THR = 0.3


def test_xyxy_iou_identical() -> None:
    assert xyxy_iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)) == 1.0


def test_xyxy_iou_disjoint() -> None:
    assert xyxy_iou((0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)) == 0.0


def test_xyxy_iou_partial() -> None:
    # 交 0.4*0.4=0.16 / 并 (0.25+0.25-0.16)=0.34
    iou = xyxy_iou((0.0, 0.0, 0.5, 0.5), (0.1, 0.1, 0.6, 0.6))
    assert abs(iou - 0.16 / 0.34) < 1e-9


def test_xyxy_iou_contained() -> None:
    # 小框完全在大框内: IoU = 小框面积 / 大框面积 = 0.04 / 1.0
    iou = xyxy_iou((0.0, 0.0, 1.0, 1.0), (0.4, 0.4, 0.6, 0.6))
    assert abs(iou - 0.04) < 1e-9


def test_greedy_match_all_matched() -> None:
    a = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.6, 0.6, 0.9, 0.9, 0.8)]
    b = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.6, 0.6, 0.9, 0.9, 0.8)]
    matched, ua, ub = greedy_match(a, b, IOU_THR)
    assert len(matched) == 2
    assert ua == [] and ub == []


def test_greedy_match_unmatched_both() -> None:
    a = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.8, 0.8, 0.95, 0.95, 0.8)]
    b = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.8, 0.1, 0.95, 0.3, 0.7)]
    matched, ua, ub = greedy_match(a, b, IOU_THR)
    assert len(matched) == 1
    assert ua == [1] and ub == [1]


def test_greedy_match_below_threshold() -> None:
    a = [(0.0, 0.0, 0.1, 0.1, 0.9)]
    b = [(0.9, 0.9, 1.0, 1.0, 0.9)]
    matched, ua, ub = greedy_match(a, b, IOU_THR)
    assert matched == [] and ua == [0] and ub == [0]


def test_greedy_match_empty_inputs() -> None:
    matched, ua, ub = greedy_match([], [], IOU_THR)
    assert matched == [] and ua == [] and ub == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/det/hardmine_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jxl.det.hardmine'`

- [ ] **Step 3: 实现 Box + xyxy_iou + greedy_match**

Create `src/jxl/det/hardmine.py`:
```python
"""Cross-detector 难例挖掘核心算法（Functional Core）。

纯函数: 双检测器框级比对 + 难例分类 + YOLO 标注生成。
供 bin/person_mine.py（Imperative Shell）调用。无 IO/模型依赖，充分单测。
"""
from __future__ import annotations

from enum import StrEnum

# 归一化 xyxy + 置信度: (x1, y1, x2, y2, conf), 坐标 ∈ [0,1]
Box = tuple[float, float, float, float, float]


def xyxy_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """两归一化 xyxy 框 (x1,y1,x2,y2) 的 IoU。无交集/零面积返回 0。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(
    boxes_a: list[Box],
    boxes_b: list[Box],
    iou_thr: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """贪心 IoU 匹配（按 IoU 降序配对，IoU<iou_thr 不配）。

    Returns:
        (matched[(idx_a, idx_b, iou)], unmatched_a[idx], unmatched_b[idx])
    """
    pairs = sorted(
        (
            (xyxy_iou(a[:4], b[:4]), ia, ib)
            for ia, a in enumerate(boxes_a)
            for ib, b in enumerate(boxes_b)
        ),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iov, ia, ib in pairs:
        if iov < iou_thr:
            break
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        matched.append((ia, ib, iov))
    unmatched_a = [i for i in range(len(boxes_a)) if i not in used_a]
    unmatched_b = [i for i in range(len(boxes_b)) if i not in used_b]
    return matched, unmatched_a, unmatched_b
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/det/hardmine_test.py -v`
Expected: PASS（8 tests）

- [ ] **Step 5: mypy + ruff（lib 要过 strict）**

Run: `uv run mypy src/jxl/det/hardmine.py`
Expected: no issues

Run: `uv run ruff check src/jxl/det/hardmine.py tests/det/hardmine_test.py`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add src/jxl/det/hardmine.py tests/det/hardmine_test.py
git commit -m "feat(det): add hardmine core (xyxy_iou, greedy_match) + tests"
```

---

### Task 2: hardmine.py — SampleClass + to_yolo_label + classify_sample

**Files:**
- Modify: `src/jxl/det/hardmine.py`（追加）
- Test: `tests/det/hardmine_test.py`（追加）

**Interfaces:**
- Consumes: `Box`, `greedy_match`（Task 1）
- Produces: `SampleClass`（StrEnum: DROP_EMPTY/DROP_AGREE/POSITIVE/NEGATIVE）、`to_yolo_label(boxes, cls_id=0) -> str`、`classify_sample(person_boxes, yoloe_boxes, iou_thr) -> SampleClass`

- [ ] **Step 1: 追加失败测试**

Append to `tests/det/hardmine_test.py`（在现有 import 块加入新符号）:
```python
from jxl.det.hardmine import (
    SampleClass,
    classify_sample,
    to_yolo_label,
)
```
Append test functions:
```python
def test_to_yolo_label_basic() -> None:
    boxes = [(0.1, 0.2, 0.3, 0.4, 0.95)]
    assert to_yolo_label(boxes, cls_id=0) == "0 0.200000 0.300000 0.200000 0.200000"


def test_to_yolo_label_swapped_coords_clamped() -> None:
    # x1>x2 / y1>y2 规整（防 w/h 负）
    boxes = [(0.3, 0.4, 0.1, 0.2, 0.5)]
    assert to_yolo_label(boxes, cls_id=0) == "0 0.200000 0.300000 0.200000 0.200000"


def test_to_yolo_label_empty() -> None:
    assert to_yolo_label([], cls_id=0) == ""


def test_to_yolo_label_multi_box() -> None:
    boxes = [(0.0, 0.0, 0.5, 1.0, 0.9), (0.5, 0.5, 1.0, 1.0, 0.8)]
    out = to_yolo_label(boxes, cls_id=0)
    assert out == "0 0.250000 0.500000 0.500000 1.000000\n0 0.750000 0.750000 0.500000 0.500000"


def test_classify_drop_empty() -> None:
    assert classify_sample([], [], IOU_THR) == SampleClass.DROP_EMPTY


def test_classify_negative_person_false_positive() -> None:
    # YOLOE 无框、person 有框 → 误检负样本
    assert classify_sample([(0.1, 0.1, 0.5, 0.5, 0.9)], [], IOU_THR) == SampleClass.NEGATIVE


def test_classify_positive_person_missed() -> None:
    # YOLOE 有框、person 无框 → 漏检正样本
    assert classify_sample([], [(0.1, 0.1, 0.5, 0.5, 0.9)], IOU_THR) == SampleClass.POSITIVE


def test_classify_drop_agree() -> None:
    box = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    assert classify_sample(box, box, IOU_THR) == SampleClass.DROP_AGREE


def test_classify_positive_extra_yoloe() -> None:
    # YOLOE 多出框（漏检位置）→ 分歧正样本
    person = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    yoloe = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.7, 0.7, 0.9, 0.9, 0.8)]
    assert classify_sample(person, yoloe, IOU_THR) == SampleClass.POSITIVE


def test_classify_positive_extra_person() -> None:
    # person 多出框、但 YOLOE 仍有人 → 正样本（YOLOE 框）
    yoloe = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    person = [(0.1, 0.1, 0.5, 0.5, 0.9), (0.7, 0.7, 0.9, 0.9, 0.8)]
    assert classify_sample(person, yoloe, IOU_THR) == SampleClass.POSITIVE
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/det/hardmine_test.py -v`
Expected: FAIL — `ImportError: cannot import name 'SampleClass'`

- [ ] **Step 3: 实现 SampleClass + to_yolo_label + classify_sample**

Append to `src/jxl/det/hardmine.py`（在 `Box` 定义之后）:
```python
class SampleClass(StrEnum):
    """双检测器比对后的样本分类。"""

    DROP_EMPTY = "drop_empty"  # 两检测器均无框（空帧）→ 丢弃
    DROP_AGREE = "drop_agree"  # 两检测器框完全配对（一致）→ 丢弃
    POSITIVE = "positive"      # YOLOE 有框且与 person.pt 分歧 → 正样本（YOLOE 框）
    NEGATIVE = "negative"      # YOLOE 无框、person.pt 有框（误检）→ 负样本（空 txt）
```
Append after `greedy_match`:
```python
def to_yolo_label(boxes: list[Box], cls_id: int = 0) -> str:
    """归一化 xyxy Box 列表 → YOLO 标注行（cls cx cy w h），每行一框。

    Box 坐标已归一化（caller 从 ultralytics boxes.xyxyn 取），无需图像尺寸。
    """
    lines: list[str] = []
    for x1, y1, x2, y2, _conf in boxes:
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def classify_sample(
    person_boxes: list[Box],
    yoloe_boxes: list[Box],
    iou_thr: float,
) -> SampleClass:
    """双检测器框级比对 → 难例分类（决策见设计文档 §5）。

    判据: ① YOLOE 有无框（决定正/负/丢）; ② 有无未配对框（决定分歧）。
    只要 YOLOE 有框，正标注一律用 YOLOE 框（更可信）。
    """
    if not yoloe_boxes and not person_boxes:
        return SampleClass.DROP_EMPTY
    if not yoloe_boxes:
        return SampleClass.NEGATIVE
    if not person_boxes:
        return SampleClass.POSITIVE
    _matched, unmatched_p, unmatched_y = greedy_match(person_boxes, yoloe_boxes, iou_thr)
    if not unmatched_p and not unmatched_y:
        return SampleClass.DROP_AGREE
    return SampleClass.POSITIVE
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/det/hardmine_test.py -v`
Expected: PASS（all）

- [ ] **Step 5: mypy + ruff**

Run: `uv run mypy src/jxl/det/hardmine.py && uv run ruff check src/jxl/det/hardmine.py tests/det/hardmine_test.py`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add src/jxl/det/hardmine.py tests/det/hardmine_test.py
git commit -m "feat(det): add hardmine SampleClass, to_yolo_label, classify_sample + tests"
```

---

### Task 3: bin/mkv_keyframes.py — mkv I-frame 抽帧 CLI

**Files:**
- Create: `src/jxl/bin/mkv_keyframes.py`

**Interfaces:**
- Produces: typer `app`，CLI `mkv_keyframes <src_dir> <dst_dir>`；函数 `extract_keyframes(src_dir, dst_dir) -> list[Path]`
- 无单测（Imperative Shell，依赖 ffmpeg/视频文件）；靠手动 + ruff 验证

- [ ] **Step 1: 实现 mkv_keyframes.py**

Create `src/jxl/bin/mkv_keyframes.py`:
```python
#!/home/jiang/py/jxl/.venv/bin/python
"""从 mkv 视频提取编码关键帧（I-frame）→ 图片目录。

ffmpeg 无损提取所有 I 帧（select=eq(pict_type,I)），全量不抽样。
输出扁平 jpg，命名 {video_stem}_{frame_idx:06d}.jpg。

用法:
    mkv_keyframes <src_dir> <dst_dir>
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, help="mkv → 编码关键帧(I-frame)提取。")

_MKV_EXT = ".mkv"


def extract_keyframes(src_dir: Path, dst_dir: Path) -> list[Path]:
    """递归找 mkv → ffmpeg 提取 I 帧 → 扁平 jpg。返回处理的 mkv 列表。"""
    if not shutil.which("ffmpeg"):
        typer.secho("未找到 ffmpeg，请先安装。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    mkvs = sorted(src_dir.rglob(f"*{_MKV_EXT}"))
    if not mkvs:
        typer.secho(f"未找到 mkv: {src_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for mkv in mkvs:
        out_pattern = str(dst_dir / f"{mkv.stem}_%06d.jpg")
        cmd = [
            "ffmpeg", "-i", str(mkv),
            "-vf", r"select=eq(pict_type\,I)",
            "-vsync", "vfr",
            "-q:v", "2",
            out_pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if result.returncode != 0:
            typer.secho(
                f"ffmpeg 失败 {mkv.name}: {result.stderr[-300:]}",
                fg=typer.colors.YELLOW, err=True,
            )
    return mkvs


@app.command()
def main(
    src_dir: Annotated[Path, typer.Argument(help="mkv 源目录（递归）")],
    dst_dir: Annotated[Path, typer.Argument(help="输出图片目录")],
) -> None:
    """递归抽取所有 mkv 的编码关键帧到扁平 jpg 目录。"""
    mkvs = extract_keyframes(src_dir, dst_dir)
    typer.secho(f"处理 {len(mkvs)} 个 mkv → {dst_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: ruff 检查**

Run: `uv run ruff check src/jxl/bin/mkv_keyframes.py`
Expected: no issues（S603 等 bin per-file-ignores 已豁免；显式 `# noqa: S603` 仅为自文档）

- [ ] **Step 3: 手动验证 ffmpeg 调用语法（不依赖真实 mkv）**

Run（确认 filter 解析无误，用 ffmpeg 自带 testsrc 生成 1s mkv 再抽帧）:
```bash
uv run python -c "
import subprocess, tempfile, os
from pathlib import Path
d = Path(tempfile.mkdtemp())
mkv = d/'t.mkv'
subprocess.run(['ffmpeg','-f','lavfi','-i','testsrc=duration=1:size=320x240:rate=10','-g','5',str(mkv)],check=True,capture_output=True)
from jxl.bin.mkv_keyframes import extract_keyframes
out = d/'frames'
extract_keyframes(mkv.parent, out)
print('产出帧数:', len(list(out.glob('*.jpg'))))
"
```
Expected: `产出帧数: >=1`（ffmpeg 无报错，I-frame 提取链路通）

- [ ] **Step 4: Commit**

```bash
git add src/jxl/bin/mkv_keyframes.py
git commit -m "feat(bin): add mkv_keyframes (I-frame extraction via ffmpeg)"
```

---

### Task 4: bin/person_mine.py — 双检测 + 比对 + 输出 CLI

**Files:**
- Create: `src/jxl/bin/person_mine.py`

**Interfaces:**
- Consumes: `Box`, `SampleClass`, `classify_sample`, `to_yolo_label`（Task 1/2）
- Produces: typer `app`，CLI `person_mine <frames_dir> <out_dir> --person-model ... --yoloe-model ... --iou --conf --device`
- 无单测（依赖 GPU/模型）；靠 ruff + 手动集成验证

- [ ] **Step 1: 实现 person_mine.py**

Create `src/jxl/bin/person_mine.py`:
```python
#!/home/jiang/py/jxl/.venv/bin/python
"""Person 难例挖掘: person.pt + YOLOE 双检测 → 框级比对 → 难例 YOLO 集。

从候选帧目录跑两个检测器（部署 person.pt + 通用 YOLOE-11l），对每图框级 IoU
比对，分歧样本保留（正样本用 YOLOE 框，误检为空 txt 负样本），一致/空帧丢弃。
输出 YOLO images/+labels/ + mining_report.json。

用法:
    person_mine <frames_dir> <out_dir> \
        --person-model /opt/howell/iap/current/ias/model/person.pt \
        --yoloe-model  /home/jiang/py/jxl/models/yoloe-11l-seg.pt \
        --iou 0.3 --conf 0.25 --device cuda:0
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Annotated

import orjson
import typer
from ultralytics import YOLO, YOLOE

from jxl.det.hardmine import (
    Box,
    SampleClass,
    classify_sample,
    to_yolo_label,
)

app = typer.Typer(add_completion=False, help="Person 难例挖掘: 双检测器比对 → 难例 YOLO 集。")

PERSON_CLS = 0  # COCO person 类 id（yoloe-11l-seg.pt 默认 COCO 80 类）
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def gather_images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def _detect(  # noqa: ANN202
    model: YOLO | YOLOE,
    paths: list[Path],
    conf: float,
    iou: float,
    device: str,
    classes: list[int] | None = None,
) -> dict[str, list[Box]]:
    """通用 ultralytics 检测 → {stem: [Box]}。Box 坐标取 boxes.xyxyn（归一化）。"""
    kwargs: dict[str, object] = {"conf": conf, "iou": iou, "verbose": False}
    if classes is not None:
        kwargs["classes"] = classes
    if device:
        kwargs["device"] = device
    out: dict[str, list[Box]] = {}
    predict_args: dict[str, object] = {"stream": True}
    predict_args.update(kwargs)
    for path, res in zip(
        paths, model.predict([str(p) for p in paths], **predict_args), strict=False
    ):  # noqa: PGH003
        boxes: list[Box] = []
        if res.boxes is not None and len(res.boxes):
            xy = res.boxes.xyxyn
            cf = res.boxes.conf
            for i in range(len(xy)):
                b = xy[i].tolist()
                boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(cf[i])))
        out[path.stem] = boxes
    return out


def write_yolo_sample(out_dir: Path, img_path: Path, boxes: list[Box] | None) -> None:
    """复制图 + 写 txt: boxes=None→空文件(负样本); list→YOLO 框行(正样本)。"""
    dst_img = out_dir / "images" / img_path.name
    dst_lbl = out_dir / "labels" / (img_path.stem + ".txt")
    shutil.copy2(img_path, dst_img)
    content = "" if boxes is None else to_yolo_label(boxes, cls_id=0)
    dst_lbl.write_text(content, encoding="utf-8")


@app.command()
def run(  # noqa: PLR0913
    frames_dir: Annotated[Path, typer.Argument(help="候选帧目录（递归）")],
    out_dir: Annotated[Path, typer.Argument(help="输出 YOLO 集目录")],
    person_model: Annotated[Path, typer.Option("--person-model", help="person.pt 路径")] = Path(
        "/opt/howell/iap/current/ias/model/person.pt"
    ),
    yoloe_model: Annotated[Path, typer.Option("--yoloe-model", help="YOLOE 权重路径")] = Path(
        "/home/jiang/py/jxl/models/yoloe-11l-seg.pt"
    ),
    iou: Annotated[float, typer.Option("--iou", help="框级匹配 IoU 阈值（放宽）")] = 0.3,
    conf: Annotated[float, typer.Option("--conf", help="检测置信度阈值（两模型共用）")] = 0.25,
    device: Annotated[str, typer.Option("--device", help="cuda:0 / cpu，空=自动")] = "",
) -> None:
    """双检测 person.pt + YOLOE → 框级比对 → 难例 YOLO 集 + report。"""
    if not person_model.is_file():
        typer.secho(f"person 模型不存在: {person_model}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not yoloe_model.is_file():
        typer.secho(f"YOLOE 模型不存在: {yoloe_model}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    imgs = gather_images(frames_dir)
    if not imgs:
        typer.secho(f"候选目录无图: {frames_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)

    typer.secho(
        f"双检测 {len(imgs)} 张 @ person={person_model.name} yoloe={yoloe_model.name}",
        fg=typer.colors.CYAN,
    )
    person_map = _detect(YOLO(str(person_model)), imgs, conf, iou, device)
    yoloe_map = _detect(YOLOE(str(yoloe_model)), imgs, conf, iou, device, classes=[PERSON_CLS])

    counts: Counter[str] = Counter()
    by_video: dict[str, Counter[str]] = {}
    for img in imgs:
        stem = img.stem
        pb = person_map.get(stem, [])
        yb = yoloe_map.get(stem, [])
        cls = classify_sample(pb, yb, iou)
        counts[cls.value] += 1
        video = stem.rsplit("_", 1)[0] if "_" in stem else stem
        by_video.setdefault(video, Counter())[cls.value] += 1
        if cls is SampleClass.POSITIVE:
            write_yolo_sample(out_dir, img, yb)
        elif cls is SampleClass.NEGATIVE:
            write_yolo_sample(out_dir, img, None)

    report = {
        "total_frames": len(imgs),
        "positive": counts.get("positive", 0),
        "negative": counts.get("negative", 0),
        "dropped_empty": counts.get("drop_empty", 0),
        "dropped_agree": counts.get("drop_agree", 0),
        "by_video": {k: dict(v) for k, v in by_video.items()},
    }
    (out_dir / "mining_report.json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    typer.secho(
        f"正样本 {report['positive']} | 负样本 {report['negative']} | "
        f"丢弃(空/一致) {report['dropped_empty']}/{report['dropped_agree']} → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: ruff 检查**

Run: `uv run ruff check src/jxl/bin/person_mine.py`
Expected: no issues（ANN/PLR0913/PGH003 等 bin per-file-ignores 已豁免；显式 noqa 为自文档）

- [ ] **Step 3: 导入冒烟测试（不加载模型，仅验证 import + typer app 可解析）**

Run: `uv run python -c "from jxl.bin.person_mine import app, gather_images, write_yolo_sample; print('import ok')"`
Expected: `import ok`（无 ImportError）

- [ ] **Step 4: 手动集成验证（小样本，依赖 GPU + 两个模型）**

Run（造一张测试图，跑通双检测 + 比对 + 输出）:
```bash
uv run python -c "
import tempfile
from pathlib import Path
from jvi.image.image_nda import ImageNda
from jvi.geo.size2d import SIZE_HD
d = Path(tempfile.mkdtemp())/'frames'
d.mkdir()
ImageNda(SIZE_HD).save(str(d/'v1_000001.jpg'))  # 空白 HD 图
out = Path(tempfile.mkdtemp())/'out'
import subprocess, sys
r = subprocess.run([sys.executable,'-m','jxl.bin.person_mine',str(d),str(out),'--device','cuda:0'],capture_output=True,text=True)
print(r.stdout, r.stderr[-500:])
print('report:', (out/'mining_report.json').read_text() if (out/'mining_report.json').exists() else 'N/A')
"
```
Expected: report 显示该空白图为 `dropped_empty` 或 `negative`（两模型在空白图上都无框 → drop_empty）；无 traceback。

- [ ] **Step 5: Commit**

```bash
git add src/jxl/bin/person_mine.py
git commit -m "feat(bin): add person_mine (dual-detector hard sample mining)"
```

---

### Task 5: 全量收尾（mypy + ruff + pytest + 集成）

- [ ] **Step 1: 全量 pytest**

Run: `uv run pytest tests/det/hardmine_test.py -v`
Expected: PASS（all hardmine tests）

- [ ] **Step 2: lib mypy strict**

Run: `uv run mypy src/jxl/det/hardmine.py`
Expected: no issues

- [ ] **Step 3: 全量 ruff（新文件）**

Run: `uv run ruff check src/jxl/det/hardmine.py src/jxl/bin/mkv_keyframes.py src/jxl/bin/person_mine.py tests/det/hardmine_test.py`
Expected: no issues

- [ ] **Step 4: 端到端冒烟（抽帧 + 挖掘，真实 mkv 路径待运维确认）**

若 `/var/howell/iap/current/ias/sh-sgcc/n001/video` 可达且有 mkv:
```bash
uv run python -m jxl.bin.mkv_keyframes /var/howell/iap/current/ias/sh-sgcc/n001/video /tmp/person_frames
uv run python -m jxl.bin.person_mine /tmp/person_frames /home/jiang/ws/sgcc/person/dates/2025-07-07 --device cuda:0
```
Expected: `dates/2025-07-07/` 下 `images/`+`labels/`+`mining_report.json` 产出；report 各类计数合理。
若数据源不可达：跳过本步（已在 spec §10 列为风险），记录待运维。

- [ ] **Step 5: 最终 commit（如有残留改动）**

```bash
git status  # 确认干净
```

---

## Self-Review 记录

- **Spec coverage**: spec §4 组件（壳/核）→ Task 1/2（核 lib）+ Task 4（壳 bin）；§5 决策表 → Task 2 `classify_sample` 测试全覆盖；§6 错误处理 → Task 4 `run`（模型/空目录报错）+ Task 3（ffmpeg 报错）；§7 测试 → Task 1/2；§8 输出+CLI → Task 4。✓
- **结构偏离 spec 说明**: 纯函数从 `person_mine.py`(bin) 提到 `src/jxl/det/hardmine.py`(lib)，以获 mypy strict 保障（bin 被 mypy exclude）。spec 壳/核分离意图不变。已在 File Structure 注明。
- **Type consistency**: `Box`=5-tuple 贯穿；`SampleClass` 枚举值（drop_empty/drop_agree/positive/negative）在 `classify_sample` 返回、`run` 的 Counter key、report 字段一致。✓
- **No placeholders**: 所有步骤含完整代码/命令/期望。✓
