# Det-Mine 多模型共识 + 争议分 cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 person 难例挖掘从二元共识升级为 N 模型加权争议分 + cascade 分级（person.pt + YOLOE + GDINO + RF-DETR）。

**Architecture:** Functional Core（`hardmine.py` 加 `find_consensus_positions`/`pick_by_priority`/`score_sample` 纯函数，mypy strict + 充分单测）+ Imperative Shell（`person_mine.py` rename `det_mine.py`，加 GDINO/RF-DETR backend + 类别参数化 + cascade 分流）。

**Tech Stack:** Python 3.12 / ultralytics（YOLO/YOLOE）/ transformers（GroundingDINO）/ rfdetr（RF-DETR）/ typer / orjson。

## Global Constraints

- Python `~=3.12.0`；lib mypy strict；`src/jxl/bin/` 被 mypy exclude。
- ruff `select=ALL` + 全局 ignore；`src/jxl/bin/*.py` per-file-ignores（ANN/T201/PLR0913/PGH003/UP042 等）。
- 测试 `uv run pytest`（禁 subprocess 调 pytest）。
- 新依赖：`transformers`（GDINO `IDEA-Research/grounding-dino-*`）、`rfdetr`（`pip install rfdetr`）。
- No Silent Degradation：validator 库缺失 / `--consensus`>校验器数 / 模型文件缺失 → 报错退出。
- commit 英文，无 attribution。
- 默认 K=2，权重 `rfdetr:0.4,gdino:0.35,yoloe:0.25`，review top 30%。

---

## File Structure

| 文件 | 变化 | 职责 |
|------|------|------|
| `src/jxl/det/hardmine.py` | Modify（追加） | `find_consensus_positions` / `pick_by_priority` / `ScoreResult` / `score_sample` |
| `tests/det/hardmine_test.py` | Modify（追加） | 上述纯函数单测 |
| `pyproject.toml` | Modify | 加 `transformers`、`rfdetr` |
| `src/jxl/bin/det_mine.py` | Create（rename from `person_mine.py`） | GDINO/RF-DETR backend + 类别参数化 + cascade |

---

### Task 1: hardmine — find_consensus_positions

**Files:**
- Modify: `src/jxl/det/hardmine.py`（追加）
- Test: `tests/det/hardmine_test.py`（追加）

**Interfaces:**
- Consumes: `Box`, `xyxy_iou`（已有）
- Produces: `find_consensus_positions(validators: dict[str, list[Box]], iou_thr: float, k: int) -> list[tuple[Box, dict[str, Box]]]`

- [ ] **Step 1: 写失败测试**

Append to `tests/det/hardmine_test.py`（在 import 块加 `find_consensus_positions`）:
```python
from jxl.det.hardmine import find_consensus_positions
```
Append tests:
```python
def test_find_consensus_positions_all_agree() -> None:
    validators = {
        "yoloe": [(0.1, 0.1, 0.5, 0.5, 0.9)],
        "gdino": [(0.1, 0.1, 0.5, 0.5, 0.95)],
        "rfdetr": [(0.12, 0.12, 0.52, 0.52, 0.99)],
    }
    positions = find_consensus_positions(validators, 0.3, 2)
    assert len(positions) == 1
    assert set(positions[0][1].keys()) == {"yoloe", "gdino", "rfdetr"}


def test_find_consensus_positions_split() -> None:
    validators = {
        "yoloe": [(0.1, 0.1, 0.3, 0.3, 0.9)],
        "gdino": [(0.1, 0.1, 0.3, 0.3, 0.95)],
        "rfdetr": [(0.7, 0.7, 0.9, 0.9, 0.99)],
    }
    positions = find_consensus_positions(validators, 0.3, 2)
    assert len(positions) == 1
    assert set(positions[0][1].keys()) == {"yoloe", "gdino"}


def test_find_consensus_positions_below_k() -> None:
    validators = {
        "yoloe": [(0.1, 0.1, 0.2, 0.2, 0.9)],
        "gdino": [(0.4, 0.4, 0.5, 0.5, 0.95)],
        "rfdetr": [(0.7, 0.7, 0.8, 0.8, 0.99)],
    }
    assert find_consensus_positions(validators, 0.3, 2) == []


def test_find_consensus_positions_empty() -> None:
    assert find_consensus_positions({"yoloe": [], "gdino": []}, 0.3, 2) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/det/hardmine_test.py -k find_consensus -v`
Expected: FAIL — `ImportError: cannot import name 'find_consensus_positions'`

- [ ] **Step 3: 实现**

Append to `src/jxl/det/hardmine.py`（在 `greedy_match` 之后）:
```python
def find_consensus_positions(
    validators: dict[str, list[Box]],
    iou_thr: float,
    k: int,
) -> list[tuple[Box, dict[str, Box]]]:
    """跨校验器贪心聚类: 找 ≥k 个校验器 IoU 重叠的共识位置。

    按 conf 降序以每个框为种子，向其他校验器找 IoU 最高的未消费框配对；
    支持校验器数 ≥k 则记为一个共识位置。
    Returns: [(representative_box, {validator_name: box}), ...]
    """
    candidates = sorted(
        ((name, idx, box) for name, boxes in validators.items() for idx, box in enumerate(boxes)),
        key=lambda x: x[2][4],
        reverse=True,
    )
    consumed: dict[str, set[int]] = {name: set() for name in validators}
    positions: list[tuple[Box, dict[str, Box]]] = []
    for name_s, idx_s, box_s in candidates:
        if idx_s in consumed[name_s]:
            continue
        supporters: dict[str, Box] = {name_s: box_s}
        consumed[name_s].add(idx_s)
        for name_o, boxes_o in validators.items():
            if name_o == name_s:
                continue
            best_idx: int | None = None
            best_iou = iou_thr
            for idx_o, box_o in enumerate(boxes_o):
                if idx_o in consumed[name_o]:
                    continue
                iov = xyxy_iou(box_s[:4], box_o[:4])
                if iov >= best_iou:
                    best_iou = iov
                    best_idx = idx_o
            if best_idx is not None:
                supporters[name_o] = boxes_o[best_idx]
                consumed[name_o].add(best_idx)
        if len(supporters) >= k:
            positions.append((box_s, supporters))
    return positions
```

- [ ] **Step 4: 跑测试确认通过 + mypy + ruff**

Run: `uv run pytest tests/det/hardmine_test.py -k find_consensus -v && uv run mypy src/jxl/det/hardmine.py && uv run ruff check src/jxl/det/hardmine.py tests/det/hardmine_test.py`
Expected: 4 passed / no issues / All checks passed

- [ ] **Step 5: Commit**

```bash
git add src/jxl/det/hardmine.py tests/det/hardmine_test.py
git commit -m "feat(det): add find_consensus_positions + tests"
```

---

### Task 2: hardmine — pick_by_priority + ScoreResult

**Files:** Modify `src/jxl/det/hardmine.py` / `tests/det/hardmine_test.py`

**Interfaces:**
- Produces: `pick_by_priority(supporters, priority) -> Box | None`；`ScoreResult(NamedTuple)`

- [ ] **Step 1: 写失败测试**

Append import `pick_by_priority`；append:
```python
def test_pick_by_priority_first() -> None:
    s = {"yoloe": (0.1, 0.1, 0.5, 0.5, 0.9), "rfdetr": (0.12, 0.12, 0.52, 0.52, 0.99)}
    assert pick_by_priority(s, ["rfdetr", "gdino", "yoloe"]) == s["rfdetr"]


def test_pick_by_priority_fallback() -> None:
    s = {"yoloe": (0.1, 0.1, 0.5, 0.5, 0.9)}
    assert pick_by_priority(s, ["rfdetr", "gdino", "yoloe"]) == s["yoloe"]


def test_pick_by_priority_none() -> None:
    assert pick_by_priority({}, ["rfdetr"]) is None
```

- [ ] **Step 2: 跑确认失败**

Run: `uv run pytest tests/det/hardmine_test.py -k pick_by_priority -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 实现**

在 `src/jxl/det/hardmine.py` 顶部 import 加 `NamedTuple`：
```python
from typing import NamedTuple
```
在 `find_consensus_positions` 之后追加：
```python
def pick_by_priority(supporters: dict[str, Box], priority: list[str]) -> Box | None:
    """按优先级返回第一个存在的校验器框（标注框回退选框用）。"""
    for name in priority:
        if name in supporters:
            return supporters[name]
    return None


class ScoreResult(NamedTuple):
    """score_sample 的返回: 争议分 + 共识标注框 + 分项计数。"""

    score: float
    boxes: list[Box]
    fp_count: int
    fn_count: int
```

- [ ] **Step 4: 跑通过 + mypy + ruff**

Run: `uv run pytest tests/det/hardmine_test.py -k pick_by_priority -v && uv run mypy src/jxl/det/hardmine.py && uv run ruff check src/jxl/det/hardmine.py`
Expected: 3 passed / no issues / All checks passed

- [ ] **Step 5: Commit**

```bash
git add src/jxl/det/hardmine.py tests/det/hardmine_test.py
git commit -m "feat(det): add pick_by_priority + ScoreResult"
```

---

### Task 3: hardmine — score_sample（核心）

**Files:** Modify `src/jxl/det/hardmine.py` / `tests/det/hardmine_test.py`

**Interfaces:**
- Consumes: `find_consensus_positions`, `pick_by_priority`, `ScoreResult`, `xyxy_iou`
- Produces: `score_sample(target_boxes, validators, weights, iou_thr, k, priority=None) -> ScoreResult`

- [ ] **Step 1: 写失败测试**

Append import `score_sample`；append（常量）:
```python
WEIGHTS = {"yoloe": 0.25, "gdino": 0.35, "rfdetr": 0.4}
PRIORITY = ["rfdetr", "gdino", "yoloe"]
```
Append tests:
```python
def test_score_sample_full_agreement() -> None:
    box = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    validators = {"yoloe": box, "gdino": box, "rfdetr": box}
    r = score_sample(box, validators, WEIGHTS, 0.3, 2, PRIORITY)
    assert r.score == 0.0
    assert r.fp_count == 0
    assert r.fn_count == 0
    assert len(r.boxes) == 1


def test_score_sample_target_missed() -> None:
    validators = {
        "yoloe": [(0.1, 0.1, 0.5, 0.5, 0.9)],
        "gdino": [(0.1, 0.1, 0.5, 0.5, 0.95)],
        "rfdetr": [(0.12, 0.12, 0.52, 0.52, 0.99)],
    }
    r = score_sample([], validators, WEIGHTS, 0.3, 2, PRIORITY)
    assert r.fn_count == 1
    assert abs(r.score - 1.0) < 1e-9  # 全员认同, W_agree/W_total=1.0
    assert r.boxes[0] == validators["rfdetr"][0]  # RF-DETR 优先


def test_score_sample_target_false_positive() -> None:
    target = [(0.1, 0.1, 0.5, 0.5, 0.9)]
    validators = {"yoloe": [], "gdino": [], "rfdetr": []}
    r = score_sample(target, validators, WEIGHTS, 0.3, 2, PRIORITY)
    assert r.fp_count == 1
    assert abs(r.score - 1.0) < 1e-9  # 无人认同, (W_total-0)/W_total=1.0
    assert r.boxes == []


def test_score_sample_partial_miss_one_validator() -> None:
    # target 漏, 但仅 yoloe+gdino 认同(2/3) → fn = (0.25+0.35)/1.0 = 0.6
    validators = {
        "yoloe": [(0.1, 0.1, 0.5, 0.5, 0.9)],
        "gdino": [(0.1, 0.1, 0.5, 0.5, 0.95)],
        "rfdetr": [],
    }
    r = score_sample([], validators, WEIGHTS, 0.3, 2, PRIORITY)
    assert r.fn_count == 1
    assert abs(r.score - 0.6) < 1e-9
    assert r.boxes[0] == validators["gdino"][0]  # RF-DETR 缺→GDINO 回退
```

- [ ] **Step 2: 跑确认失败**

Run: `uv run pytest tests/det/hardmine_test.py -k score_sample -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 实现**

Append to `src/jxl/det/hardmine.py`（在 `ScoreResult` 之后）:
```python
def score_sample(
    target_boxes: list[Box],
    validators: dict[str, list[Box]],
    weights: dict[str, float],
    iou_thr: float,
    k: int,
    priority: list[str] | None = None,
) -> ScoreResult:
    """N 模型加权争议分 + 共识标注框。

    fp = Σ target框(认同票<k): (W_total − W_认同)/W_total  (疑似 target 误检)
    fn = Σ 共识漏检位置: W_认同/W_total                   (target 漏, 校验器共识)
    score = fp + fn; 0=全一致; 越大越争议。
    boxes = 共识位置按 priority 回退选框(供 L1 自动标注)。
    """
    if priority is None:
        priority = sorted(weights.keys(), key=lambda n: weights[n], reverse=True)
    w_total = sum(weights.values()) or 1.0

    fp = 0.0
    fp_count = 0
    for tbox in target_boxes:
        w_agree = 0.0
        votes = 0
        for vname, vboxes in validators.items():
            if any(xyxy_iou(tbox[:4], vb[:4]) >= iou_thr for vb in vboxes):
                votes += 1
                w_agree += weights.get(vname, 0.0)
        if votes < k:
            fp += (w_total - w_agree) / w_total
            fp_count += 1

    fn = 0.0
    fn_count = 0
    boxes: list[Box] = []
    for _pos_box, supporters in find_consensus_positions(validators, iou_thr, k):
        covered = any(xyxy_iou(_pos_box[:4], tb[:4]) >= iou_thr for tb in target_boxes)
        if not covered:
            w_agree = sum(weights.get(n, 0.0) for n in supporters)
            fn += w_agree / w_total
            fn_count += 1
        picked = pick_by_priority(supporters, priority)
        if picked is not None:
            boxes.append(picked)

    return ScoreResult(score=fp + fn, boxes=boxes, fp_count=fp_count, fn_count=fn_count)
```

- [ ] **Step 4: 跑通过 + mypy + ruff**

Run: `uv run pytest tests/det/hardmine_test.py -v && uv run mypy src/jxl/det/hardmine.py && uv run ruff check src/jxl/det/hardmine.py tests/det/hardmine_test.py`
Expected: all passed / no issues / All checks passed

- [ ] **Step 5: Commit**

```bash
git add src/jxl/det/hardmine.py tests/det/hardmine_test.py
git commit -m "feat(det): add score_sample (weighted dispute score) + tests"
```

---

### Task 4: pyproject — 加 transformers + rfdetr 依赖

**Files:** Modify `pyproject.toml`

- [ ] **Step 1: 加依赖**

在 `pyproject.toml` 的 `[project] dependencies` 末尾加（rmb 合成栈注释后）:
```toml
    "transformers",    # Grounding DINO 开放词汇检测器 (det_mine validator)
    "rfdetr",          # RF-DETR COCO SOTA 检测器 (det_mine validator)
```

- [ ] **Step 2: sync + 验证导入**

Run: `uv sync 2>&1 | tail -5 && uv run python -c "import transformers; from rfdetr import RFDETRBase; print('transformers', transformers.__version__)" 2>&1 | tail -3`
Expected: uv sync 成功（可能下载重依赖，需外网）；import 成功打印版本。
**注**：若 uv sync 因外网失败，本机用 `uv pip install transformers rfdetr`；sgcc0 用 proxychains（见 [[sgcc0-proxychains]]）。

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add transformers (GroundingDINO) + rfdetr"
```

---

### Task 5: det_mine — rename + detect_gdino backend

**Files:**
- Rename: `src/jxl/bin/person_mine.py` → `src/jxl/bin/det_mine.py`
- Modify: `src/jxl/bin/det_mine.py`

**注**：Grounding DINO 的 transformers API（`post_process_grounded_object_detection`）签名随版本变。下列为参考实现，**实现时按本机 transformers 版本实测调整**（Step 3 验证）。

- [ ] **Step 1: rename**

```bash
git mv src/jxl/bin/person_mine.py src/jxl/bin/det_mine.py
```

- [ ] **Step 2: 加 detect_gdino**

在 `src/jxl/bin/det_mine.py` 加 import（顶部）:
```python
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image
import torch
```
在 `detect_yoloe` 之后加:
```python
def detect_gdino(
    paths: list[Path],
    model_name: str,
    text: str,
    device: str,
) -> dict[str, list[Box]]:
    """Grounding DINO 开放词汇检测: text prompt (如 'person') → {stem: [Box]}。

    API 以 transformers 版本为准 (post_process_grounded_object_detection)。
    Box 坐标归一化到 [0,1]。
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = GroundingDinoForObjectDetection.from_pretrained(model_name)
    dev = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    text_prompt = text + " ."  # GDINO 约定: 末尾点终止
    out: dict[str, list[Box]] = {}
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25
        )[0]
        w, h = image.size
        boxes: list[Box] = []
        for box, score in zip(results["boxes"], results["scores"], strict=False):
            x1, y1, x2, y2 = box.tolist()
            boxes.append((x1 / w, y1 / h, x2 / w, y2 / h, float(score)))
        out[path.stem] = boxes
    return out
```

- [ ] **Step 3: ruff + import 冒烟 + 小样本验证（API 实测）**

Run: `uv run ruff check src/jxl/bin/det_mine.py && uv run python -c "from jxl.bin.det_mine import detect_gdino; print('import ok')"`
Expected: All checks passed / import ok

Run（API 实测，下载权重后跑 1 张）:
```bash
uv run python -c "
from pathlib import Path
from jxl.bin.det_mine import detect_gdino
import tempfile
from PIL import Image
import numpy as np
d = Path(tempfile.mkdtemp()); img = d/'t.jpg'
Image.fromarray(np.zeros((480,640,3),dtype=np.uint8)).save(img)
r = detect_gdino([img], 'IDEA-Research/grounding-dino-tiny', 'person', '')
print('gdino ok, stems:', list(r.keys()))
"
```
Expected: 权重下载后打印 `gdino ok, stems: ['t']`（空图无框但 stem 在）。若 API 报错，按 transformers 版本调整 `post_process_grounded_object_detection` 调用签名。

- [ ] **Step 4: Commit**

```bash
git add src/jxl/bin/det_mine.py
git commit -m "feat(bin): det_mine rename + detect_gdino (Grounding DINO) backend"
```

---

### Task 6: det_mine — detect_rfdetr backend

**Files:** Modify `src/jxl/bin/det_mine.py`

**注**：rfdetr 的 `predict` 返回 supervision `Detections` 对象。参考实现，实现时实测。

- [ ] **Step 1: 加 detect_rfdetr**

在 `src/jxl/bin/det_mine.py` 加 import:
```python
import cv2
```
在 `detect_gdino` 之后加:
```python
def detect_rfdetr(
    paths: list[Path],
    model,  # rfdetr RFDETRBase/Small/Large 实例 (caller 构造，避免循环重建)
    class_id: int,
    conf: float,
) -> dict[str, list[Box]]:
    """RF-DETR COCO 检测: 筛 class_id (COCO person=0) → {stem: [Box]}。

    model 由 caller 构造 (RFDETRBase/Large)，本函数仅 predict。
    API 以 rfdetr 版本为准 (model.predict → supervision Detections)。
    """
    out: dict[str, list[Box]] = {}
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        detections = model.predict(frame, threshold=conf)
        h, w = frame.shape[:2]
        boxes: list[Box] = []
        xyxy = detections.xyxy
        cls = detections.class_id
        cf = detections.confidence
        for i in range(len(xyxy)):
            if int(cls[i]) != class_id:
                continue
            x1, y1, x2, y2 = xyxy[i].tolist()
            boxes.append((x1 / w, y1 / h, x2 / w, y2 / h, float(cf[i])))
        out[path.stem] = boxes
    return out
```

- [ ] **Step 2: ruff + import + 小样本验证**

Run: `uv run ruff check src/jxl/bin/det_mine.py && uv run python -c "from jxl.bin.det_mine import detect_rfdetr; print('import ok')"`
Expected: All checks passed / import ok

Run（实测）:
```bash
uv run python -c "
from pathlib import Path
from rfdetr import RFDETRBase
from jxl.bin.det_mine import detect_rfdetr
import tempfile
from PIL import Image
import numpy as np
d = Path(tempfile.mkdtemp()); img = d/'t.jpg'
Image.fromarray(np.zeros((480,640,3),dtype=np.uint8)).save(img)
m = RFDETRBase()
r = detect_rfdetr([img], m, class_id=0, conf=0.5)
print('rfdetr ok, stems:', list(r.keys()))
"
```
Expected: 权重下载后 `rfdetr ok`。若 `model.predict` 签名不同（如 `detections.xyxy` 字段名），按 rfdetr 版本调整。

- [ ] **Step 3: Commit**

```bash
git add src/jxl/bin/det_mine.py
git commit -m "feat(bin): det_mine detect_rfdetr (RF-DETR) backend"
```

---

### Task 7: det_mine — 类别参数化 + cascade 分流 + run

**Files:** Modify `src/jxl/bin/det_mine.py`

**Interfaces:**
- Consumes: `score_sample`, `to_yolo_label`, `SampleClass`/`Box`（hardmine），`detect_*` backends
- Produces: 重写 `run()`（CLI），cascade 分流到 `images/+labels/`（L1）与 `review/`（L2/L3）

- [ ] **Step 1: 改 import + run（cascade 版）**

在 `src/jxl/bin/det_mine.py` 的 hardmine import 加 `score_sample`：
```python
from jxl.det.hardmine import (
    Box,
    score_sample,
    to_yolo_label,
)
```
重写 `run` 函数（替换原 `run`）:
```python
VALIDATOR_BACKENDS = {"yoloe", "gdino", "rfdetr"}


def _parse_weights(s: str) -> dict[str, float]:
    """'rfdetr:0.4,gdino:0.35' → {'rfdetr':0.4, ...}"""
    out: dict[str, float] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        name, _, val = kv.partition(":")
        out[name.strip()] = float(val)
    return out


@app.command()
def run(  # noqa: PLR0913
    frames_dir: Annotated[Path, typer.Argument(help="候选帧目录（递归）")],
    out_dir: Annotated[Path, typer.Argument(help="输出目录")],
    target: Annotated[str, typer.Option("--target", help="目标类(决定 prompt/set_classes/cls_id)")] = "person",
    target_model: Annotated[Path, typer.Option("--target-model", help="被校验专用 YOLO 权重")] = Path(
        "/opt/howell/iap/current/ias/model/person.pt"
    ),
    cls_id: Annotated[int, typer.Option("--cls-id", help="YOLO 标注类 id")] = 0,
    validators: Annotated[str, typer.Option("--validators", help="校验器组合(逗号分隔)")] = "yoloe,gdino,rfdetr",
    weights: Annotated[str, typer.Option("--validator-weights", help="权重 name:w,...")] = "rfdetr:0.4,gdino:0.35,yoloe:0.25",
    gdino_model: Annotated[str, typer.Option("--gdino-model", help="Grounding DINO HF 模型名")] = "IDEA-Research/grounding-dino-tiny",
    rfdetr_variant: Annotated[str, typer.Option("--rfdetr-variant", help="RF-DETR 变体 base/small/large")] = "base",
    iou: Annotated[float, typer.Option("--iou", help="IoU 匹配阈值")] = 0.3,
    consensus: Annotated[int, typer.Option("--consensus", help="共识校验器数 K")] = 2,
    review_top: Annotated[float, typer.Option("--review-top", help="高争议进 review 的比例")] = 0.3,
    conf: Annotated[float, typer.Option("--conf", help="检测置信度")] = 0.25,
    device: Annotated[str, typer.Option("--device", help="cuda:0/cpu")] = "",
) -> None:
    """N 模型加权争议分 + cascade: L0 丢弃 / L1 自动标注 / L2-L3 review 候选集。"""
    if not 0.0 <= iou <= 1.0 or not 0.0 <= conf <= 1.0:
        typer.secho("--iou/--conf 须在 [0,1]", fg=typer.colors.RED, err=True); raise typer.Exit(1)
    vlist = [v.strip() for v in validators.split(",") if v.strip()]
    bad = [v for v in vlist if v not in VALIDATOR_BACKENDS]
    if bad:
        typer.secho(f"未知 validator: {bad}（可选 {VALIDATOR_BACKENDS}）", fg=typer.colors.RED, err=True); raise typer.Exit(1)
    if consensus > len(vlist):
        typer.secho(f"--consensus {consensus} > 校验器数 {len(vlist)}", fg=typer.colors.RED, err=True); raise typer.Exit(1)
    wmap = _parse_weights(weights)
    priority = sorted(wmap.keys(), key=lambda n: wmap[n], reverse=True)

    if not target_model.is_file():
        typer.secho(f"target 模型不存在: {target_model}", fg=typer.colors.RED, err=True); raise typer.Exit(1)
    imgs = gather_images(frames_dir)
    if not imgs:
        typer.secho(f"候选目录无图: {frames_dir}", fg=typer.colors.RED, err=True); raise typer.Exit(1)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    (out_dir / "review").mkdir(parents=True)

    typer.secho(f"det_mine target={target} validators={vlist} frames={len(imgs)}", fg=typer.colors.CYAN)
    target_map = _detect(YOLO(str(target_model)), imgs, conf, iou, device)
    vmaps: dict[str, dict[str, list[Box]]] = {}
    if "yoloe" in vlist:
        vmaps["yoloe"] = detect_yoloe(imgs, Path("/home/jiang/py/jxl/models/yoloe-11l-seg.pt"), conf, iou, device, target)
    if "gdino" in vlist:
        vmaps["gdino"] = detect_gdino(imgs, gdino_model, target, device)
    if "rfdetr" in vlist:
        from rfdetr import RFDETRBase, RFDETRLarge  # noqa: PLC0415
        cls = {"base": RFDETRBase, "large": RFDETRLarge}.get(rfdetr_variant, RFDETRBase)
        vmaps["rfdetr"] = detect_rfdetr(imgs, cls(), class_id=0, conf=conf)

    # 评分
    scored: list[tuple[Path, float, list[Box], int, int]] = []
    skipped = 0
    for img in imgs:
        stem = img.stem
        tb = target_map.get(stem)
        vs = {vn: vmaps[vn].get(stem, []) for vn in vlist}
        if tb is None and all(not vs[vn] for vn in vlist):
            skipped += 1; continue
        r = score_sample(tb or [], vs, wmap, iou, consensus, priority)
        scored.append((img, r.score, r.boxes, r.fp_count, r.fn_count))

    # cascade 分流: score==0 → L0 drop; score>0 按分数排序 top review_top → review; 余 → L1 auto-label
    nonzero = sorted([s for s in scored if s[1] > 0], key=lambda x: x[1], reverse=True)
    review_n = int(len(nonzero) * review_top)
    review_stems = {s[0].stem for s in nonzero[:review_n]}

    l1 = l2 = l0 = 0
    manifest_lines: list[str] = []
    for img, score, boxes, fpc, fnc in scored:
        if score == 0:
            l0 += 1; continue
        if img.stem in review_stems:
            # L2/L3 review
            shutil.copy2(img, out_dir / "review" / img.name)
            import orjson  # noqa: PLC0415
            rec = {"image": img.name, "score": score, "target": target_map.get(img.stem, []),
                   "validators": {vn: vmaps[vn].get(img.stem, []) for vn in vlist},
                   "fp_count": fpc, "fn_count": fnc}
            manifest_lines.append(orjson.dumps(rec).decode())
            l2 += 1
        else:
            # L1 auto-label
            shutil.copy2(img, out_dir / "images" / img.name)
            content = to_yolo_label(boxes, cls_id=cls_id)
            (out_dir / "labels" / (img.stem + ".txt")).write_text(content, encoding="utf-8")
            l1 += 1

    (out_dir / "review" / "manifest.jsonl").write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    import orjson  # noqa: PLC0415
    report = {"target": target, "total": len(imgs), "skipped": skipped, "L0_drop": l0, "L1_auto": l1, "review": l2,
              "validators": vlist, "weights": wmap, "iou": iou, "consensus": consensus, "review_top": review_top}
    (out_dir / "mining_report.json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    typer.secho(f"L0 丢 {l0} | L1 自动 {l1} | review {l2} | skip {skipped} → {out_dir}", fg=typer.colors.GREEN)
```

**注**：`detect_yoloe` 需加 `target` 参数（替代硬编码 "person"）。改 `detect_yoloe` 签名加 `classes_name: str`：
```python
def detect_yoloe(paths, model_path, conf, iou, device, classes_name: str = "person") -> dict[str, list[Box]]:
    model = YOLOE(str(model_path))
    model.set_classes([classes_name], model.get_text_pe([classes_name]))
    return _detect(model, paths, conf, iou, device)
```

- [ ] **Step 2: ruff + import 冒烟**

Run: `uv run ruff check src/jxl/bin/det_mine.py && uv run python -c "from jxl.bin.det_mine import app, run; print('ok')"`
Expected: All checks passed / ok（ruff 可能在 PGH003/ANN 上需 `# noqa`，按报错加）

- [ ] **Step 3: Commit**

```bash
git add src/jxl/bin/det_mine.py
git commit -m "feat(bin): det_mine category params + cascade (L0/L1/review) + score_sample"
```

---

### Task 8: 全量收尾（mypy + ruff + pytest + 集成）

- [ ] **Step 1: 全量 pytest + mypy lib + ruff 新文件**

Run: `uv run pytest tests/det/hardmine_test.py -v && uv run mypy src/jxl/det/hardmine.py && uv run ruff check src/jxl/det/hardmine.py src/jxl/bin/det_mine.py tests/det/hardmine_test.py`
Expected: all passed / no issues / All checks passed

- [ ] **Step 2: 集成冒烟（真实小样本，依赖权重下载 + GPU）**

Run:
```bash
uv run python -m jxl.bin.det_mine /tmp/person_frames_subset /tmp/detmine_out \
    --target person --target-model /opt/howell/iap/current/ias/model/person.pt \
    --validators yoloe,gdino,rfdetr --device cuda:0
```
（`/tmp/person_frames_subset` 取 dates/2025-07-07/images 的 50 张子集）
Expected: `<out>` 产出 `images/+labels/`（L1）+ `review/manifest.jsonl`（L2/L3）+ `mining_report.json`；report 含 L0/L1/review 计数。

- [ ] **Step 3: 确认 git 干净 + 旧 person_mine 引用清理**

Run: `git status --short && grep -rn "person_mine" src tests docs 2>/dev/null`
Expected: 无 person_mine 残留引用（rename 后全 det_mine）。

- [ ] **Step 4: 最终 commit（如有残留）**

```bash
git status  # 确认干净
```

---

## Self-Review 记录

- **Spec coverage**: §2 决策（K=2/权重/RF-DETR 优先/B+）→ Task 3/7；§5 争议分（fp/fn）→ Task 3；§6 cascade（L0/L1/review）→ Task 7；§7 输出（images+labels/review manifest）→ Task 7；§9 依赖 → Task 4；§10 测试 → Task 1-3；§11 P2/P3 TODO → 不在本 plan（spec §11 已记录）。✓
- **Type consistency**: `Box`=5-tuple 贯穿；`ScoreResult`(NamedTuple) 字段 score/boxes/fp_count/fn_count 在 Task 2 定义、Task 3 使用、Task 7 消费一致；`find_consensus_positions` 返回 `list[tuple[Box, dict[str,Box]]]` 在 Task 1 定义、Task 3 消费一致。✓
- **backend API 不确定**: GDINO `post_process_grounded_object_detection` 与 rfdetr `model.predict` 签名随版本，Task 5/6 给参考实现 + 实测验证步骤（诚实，非占位符）。✓
- **No placeholders**: 纯函数 Task 1-3 完整 TDD 代码；backend Task 5-6 参考实现 + 验证；Task 7 完整 run。✓
