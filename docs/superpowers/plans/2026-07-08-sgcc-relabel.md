# sgcc 多模型重标注 + P2 豆包 grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** sgcc（含 video-extract 合并 20124 图）det_mine 多模型共识重标注（L1）+ review 豆包 vision grounding（P2）复检，替旧 YOLOE。

**Architecture:** 合并 video-extract→sgcc → det_mine 4 模型跑 sgcc（L1+review）→ doubao_relabel（复用 rmb_ground 豆包 grounding）复检 review → 合并 L1+review-grounding 替旧标注。

**Tech Stack:** Python 3.12 / det_mine（ultralytics+transformers+rfdetr）/ doubao_relabel（httpx+pydantic，复用 rmb_ground）/ typer。

## Global Constraints

- Python `~=3.12.0`；lib mypy strict；`src/jxl/bin/` mypy exclude；ruff `ALL` + bin per-file-ignores。
- 测试 `uv run pytest`。
- 豆包 key 从配置文件读（`--cfg`，rmb_ground `load_backend` 模式），**绝不硬编码**（doubao key 是 secret）。
- 复用：`det_mine`（4 模型 cascade）、`rmb_ground`（`load_backend`/`parse_detections`/`Backend`）。
- det_mine/doubao 长任务（GDINO 慢、豆包 API 量）→ 后台 + Monitor。

---

## File Structure

| 文件 | 变化 | 职责 |
|------|------|------|
| `datasets/sgcc/` | Modify（扩到 20124）| 合并 video-extract + 重标注 |
| `src/jxl/bin/doubao_relabel.py` | Create | P2 豆包 grounding 复检 review 集 → YOLO labels |
| `datasets/sgcc-yoloe-bak/` | Create | 旧 YOLOE 标注备份 |

---

### Task 1: video-extract 合并 sgcc + 旧标注 backup

**Files:** 数据操作（`/home/jiang/ws/sgcc/person/datasets/`）

- [ ] **Step 1: 备份旧 sgcc YOLOE 标注**

```bash
cd /home/jiang/ws/sgcc/person/datasets
cp -rl sgcc sgcc-yoloe-bak  # hardlink 备份(images+labels)
echo "sgcc-yoloe-bak: img=$(ls sgcc-yoloe-bak/images | wc -l)"
```

- [ ] **Step 2: video-extract 图合并 sgcc**

```bash
cd /home/jiang/ws/sgcc/person/datasets
uv run python -c "
from pathlib import Path
import shutil
src = Path('video-extract'); dst = Path('sgcc')
n = 0
for img in (src/'images').glob('*.jpg'):
    shutil.move(str(img), str(dst/'images'/img.name)); n += 1
for lbl in (src/'labels').glob('*.txt'):
    shutil.move(str(lbl), str(dst/'labels'/lbl.name))
shutil.rmtree(src)
print(f'合并 {n} 图 → sgcc (总 {len(list((dst/\"images\").glob(\"*.jpg\")))})')
"
```

- [ ] **Step 3: 验证 + 更新 experiments/ toml（移除 video-extract 引用）**

```bash
echo "sgcc: $(ls /home/jiang/ws/sgcc/person/datasets/sgcc/images | wc -l) (应 20124)"
# experiments/non-standing-boost.toml 移除 video-extract（已并入 sgcc）
sed -i 's/"video-extract", //;s/, "video-extract"//' /home/jiang/py/jxl/experiments/non-standing-boost.toml
cat /home/jiang/py/jxl/experiments/non-standing-boost.toml
```

- [ ] **Step 4: Commit（experiments 更新；数据不入 git）**

```bash
cd /home/jiang/py/jxl
git add experiments/non-standing-boost.toml
git commit -m "refactor(data): merge video-extract into sgcc (20124), backup yoloe labels"
```

---

### Task 2: det_mine 跑 sgcc（L1 重标注 + review）

**Files:** 数据操作（det_mine 跑 sgcc → L1 + review）

- [ ] **Step 1: det_mine 跑 sgcc（后台，~2h GDINO 慢）**

```bash
LOG=/home/jiang/ws/sgcc/person/datasets/_sgcc_detmine.log
{
echo "$(date '+%F %T') START det_mine sgcc (20124, 4 models)"
uv run python -m jxl.bin.det_mine \
    /home/jiang/ws/sgcc/person/datasets/sgcc/images \
    /home/jiang/ws/sgcc/person/datasets/sgcc-detmine \
    --target person --target-model /opt/howell/iap/current/ias/model/person.pt \
    --validators yoloe,gdino,rfdetr --device cuda:0
echo "$(date '+%F %T') DONE det_mine sgcc"
} > "$LOG" 2>&1
```
后台跑（run_in_background）+ Monitor `_sgcc_detmine.log`（DONE/L0/Error）。

- [ ] **Step 2: 完成后验证 L1/review 分布**

```bash
cat /home/jiang/ws/sgcc/person/datasets/sgcc-detmine/mining_report.json | python3 -c "import json,sys; d=json.load(sys.stdin); print({k:d[k] for k in ['L0_drop','L1_auto','review','skipped']})"
echo "L1 img: $(ls /home/jiang/ws/sgcc/person/datasets/sgcc-detmine/images | wc -l)"
echo "review manifest: $(wc -l < /home/jiang/ws/sgcc/person/datasets/sgcc-detmine/review/manifest.jsonl)"
```

---

### Task 3: doubao_relabel bin（P2 豆包 grounding 复检 review）

**Files:**
- Create: `src/jxl/bin/doubao_relabel.py`

**Interfaces:**
- Consumes: `jxl.bin.rmb_ground.{Backend, load_backend, parse_detections}`；det_mine review/manifest.jsonl
- Produces: `doubao_relabel <manifest> <images> <out_labels> --cfg --model` CLI

- [ ] **Step 1: 实现 doubao_relabel.py**

Create `src/jxl/bin/doubao_relabel.py`:
```python
#!/home/jiang/py/jxl/.venv/bin/python
"""P2 豆包 vision grounding 复检 det_mine review 集 → YOLO labels。

读 det_mine review/manifest.jsonl，对每图豆包 grounding(text "person")→ bbox，
输出 YOLO labels(cls=0)。复用 rmb_ground 的 load_backend + parse_detections。
key 从配置文件读(--cfg, rmb_ground 模式), 绝不硬编码。

用法:
    doubao_relabel <review_manifest.jsonl> <review_images_dir> <out_labels_dir> \
        --cfg <llm.json> --model doubao-seed-2-0-lite-260215
"""
import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Annotated

import httpx
import typer
from PIL import Image

from jxl.bin.rmb_ground import Backend, load_backend, parse_detections  # noqa: E402

app = typer.Typer(add_completion=False, help="P2 豆包 grounding 复检 review 集 → YOLO labels。")

PROMPT = """检测图中所有 person 的位置。严格只输出一个 JSON 数组,不要任何其他文字、不要 markdown。
每个 person 一个对象: {"label":"person","bbox":[x1,y1,x2,y2],"conf":0-1}
- bbox 归一化坐标 [0,1], x1y1=左上, x2y2=右下
- 列出所有可见人(含部分遮挡)
- 若无人, 输出 []
仅输出 JSON 数组。"""

MAX_IMG_SIDE = 1024


def encode_image(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as im:
        img = im.convert("RGB")
        if max(img.size) > MAX_IMG_SIDE:
            img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), img.width, img.height


async def ground_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    path: Path,
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[Path, list, str | None]:
    async with sem:
        try:
            data_url, w, h = await asyncio.to_thread(encode_image, path)
        except OSError as e:
            return path, [], f"image_read: {e}"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}], "temperature": 0.1, "max_tokens": 1024}
        try:
            resp = await client.post(f"{base_url}chat/completions", headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            return path, [], f"http: {e}"
        except (KeyError, IndexError, ValueError) as e:
            return path, [], f"shape: {e}"
        return path, parse_detections(content, w, h), None


@app.command()
def main(  # noqa: PLR0913
    manifest: Annotated[Path, typer.Argument(help="det_mine review/manifest.jsonl")],
    images_dir: Annotated[Path, typer.Argument(help="review 图目录")],
    out_labels: Annotated[Path, typer.Argument(help="输出 YOLO labels 目录")],
    cfg: Annotated[str, typer.Option("--cfg", help="豆包配置文件(base_url+api_key+model)")] = "",
    model: Annotated[str, typer.Option("--model", help="覆盖模型名")] = "doubao-seed-2-0-lite-260215",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 6,
    limit: Annotated[int, typer.Option("--limit", help="只处理前N张(0=全部)")] = 0,
) -> None:
    """读 review manifest → 豆包 grounding → YOLO labels(cls=0)。"""
    base_url, api_key, use_model = load_backend(Backend.DOUBAO, model, cfg)
    out_labels.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = [images_dir / r["image"] for r in recs]
    if limit:
        paths = paths[:limit]
    if not paths:
        typer.secho("无 review 图", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.secho(f"豆包 grounding {len(paths)} review 图 @ {use_model}", fg=typer.colors.CYAN)

    sem = asyncio.Semaphore(concurrency)

    async def run() -> list:
        results = []
        async with httpx.AsyncClient() as client:
            tasks = [ground_one(client, sem, p, base_url, api_key, use_model) for p in paths]
            done = 0
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                done += 1
                if done % 50 == 0 or done == len(paths):
                    typer.echo(f"  进度 {done}/{len(paths)}")
        return results

    results = asyncio.run(run())
    results.sort(key=lambda r: str(r[0]))
    n_ok = n_empty = n_err = 0
    for path, dets, err in results:
        lbl = out_labels / (path.stem + ".txt")
        if err:
            n_err += 1
            continue  # 错误图: 不写 label(跳过)
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            lines.append(f"0 {cx:.6f} {cy:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            n_ok += 1
        else:
            n_empty += 1
    typer.secho(f"有框 {n_ok} | 空(无人) {n_empty} | 错误 {n_err} → {out_labels}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: ruff + import 冒烟**

Run: `uv run ruff check src/jxl/bin/doubao_relabel.py && uv run python -c "from jxl.bin.doubao_relabel import app, ground_one; print('ok')"`
Expected: All checks passed / ok

- [ ] **Step 3: chmod +x + Commit**

```bash
chmod +x src/jxl/bin/doubao_relabel.py
git add src/jxl/bin/doubao_relabel.py
git commit -m "feat(bin): doubao_relabel (P2 VLM grounding review -> YOLO labels)"
```

---

### Task 4: review 豆包 grounding 复检 + 合并 L1 → sgcc labels

**Files:** 数据操作（doubao_relabel 跑 + 合并标注）

**前置**：Task 2 det_mine 完成（sgcc-detmine/review/manifest.jsonl）+ 豆包配置文件（`--cfg`，含 base_url+api_key+model）。

- [ ] **Step 1: doubao_relabel 跑 review（后台，~5400 图豆包 API）**

```bash
# 准备豆包配置(用户提供的 doubao key, 不入 git)
cat > /tmp/doubao.json <<'EOF'
{"base_url":"https://ark.cn-beijing.volces.com/api/v3/","api_key":"<KEY>","model":"doubao-seed-2-0-lite-260215"}
EOF
LOG=/home/jiang/ws/sgcc/person/datasets/_sgcc_doubao.log
{
echo "$(date '+%F %T') START doubao_relabel review"
uv run python -m jxl.bin.doubao_relabel \
    /home/jiang/ws/sgcc/person/datasets/sgcc-detmine/review/manifest.jsonl \
    /home/jiang/ws/sgcc/person/datasets/sgcc-detmine/review \
    /home/jiang/ws/sgcc/person/datasets/sgcc-detmine/review-labels \
    --cfg /tmp/doubao.json --concurrency 6
echo "$(date '+%F %T') DONE doubao_relabel"
} > "$LOG" 2>&1
```
后台 + Monitor（DONE/有框/错误）。

- [ ] **Step 2: 合并 L1 + review-labels → sgcc labels（替旧 YOLOE）**

```bash
cd /home/jiang/ws/sgcc/person/datasets
uv run python -c "
from pathlib import Path
import shutil
# 清旧 sgcc labels（已 backup sgcc-yoloe-bak）
sgcc = Path('sgcc')
for lbl in (sgcc/'labels').glob('*.txt'):
    lbl.unlink()
# L1 labels（多模型共识）
dm = Path('sgcc-detmine')
n1 = 0
for lbl in (dm/'labels').glob('*.txt'):
    shutil.move(str(lbl), str(sgcc/'labels'/lbl.name)); n1 += 1
# review labels（豆包 grounding）— review 图名 = manifest image，label stem 对应
n2 = 0
for lbl in (dm/'review-labels').glob('*.txt'):
    shutil.move(str(lbl), str(sgcc/'labels'/lbl.name)); n2 += 1
print(f'L1 {n1} + review-grounding {n2} → sgcc/labels (总 {len(list((sgcc/\"labels\").glob(\"*.txt\")))})')
"
```

- [ ] **Step 3: 验证 sgcc labels 完整**

```bash
echo "sgcc images: $(ls /home/jiang/ws/sgcc/person/datasets/sgcc/images | wc -l) | labels: $(ls /home/jiang/ws/sgcc/person/datasets/sgcc/labels | wc -l)"
```

---

### Task 5: 验证（PIL 抽检新标注质量）+ 收尾

- [ ] **Step 1: PIL 预览网格抽检（新标注 vs 旧 YOLOE）**

```bash
uv run python <<'EOF'
import random
from pathlib import Path
from PIL import Image, ImageDraw
SGCC = Path("/home/jiang/ws/sgcc/person/datasets/sgcc")
BAK = Path("/home/jiang/ws/sgcc/person/datasets/sgcc-yoloe-bak")
PREVIEW = Path("/home/jiang/ws/sgcc/person/datasets/_preview_relabel")
PREVIEW.mkdir(exist_ok=True)
imgs = random.sample(list((SGCC/"images").glob("*.jpg")), 16)
THUMB = (320, 240)
def load(img, labels_dir, color):
    im = Image.open(img).convert("RGB"); im.thumbnail((640, 480))
    W, H = im.size; d = ImageDraw.Draw(im)
    lbl = Path(labels_dir) / (img.stem + ".txt")
    if lbl.is_file():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, w, h = map(float, p[1:5])
                d.rectangle([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H], outline=color, width=2)
    return im
# 新(红) vs 旧(绿) 拼
new = [load(i, SGCC/"labels", (255, 0, 0)) for i in imgs]
old = [load(i, BAK/"labels", (0, 255, 0)) for i in imgs]
def grid(images, cols):
    rows = (len(images) + cols - 1) // cols; cw, ch = THUMB
    canvas = Image.new("RGB", (cols*cw, rows*ch), (30, 30, 30))
    for i, im in enumerate(images):
        t = im.copy(); t.thumbnail(THUMB)
        canvas.paste(t, ((i%cols)*cw + (cw-t.width)//2, (i//cols)*ch))
    return canvas
grid(new, 4).save(PREVIEW/"new_multimodel.jpg", quality=85)
grid(old, 4).save(PREVIEW/"old_yoloe.jpg", quality=85)
print(f"生成 new_multimodel.jpg(红=新多模型) + old_yoloe.jpg(绿=旧YOLOE) → {PREVIEW}")
EOF
```

- [ ] **Step 2: 清理中间产物（sgcc-detmine 图，保留 review manifest 备查）**

```bash
# sgcc-detmine images（L1/review 图，已合并标注）可清理；labels 已移走
rm -rf /home/jiang/ws/sgcc/person/datasets/sgcc-detmine/images
rm -rf /home/jiang/ws/sgcc/person/datasets/sgcc-detmine/review/*.jpg
echo "清理完成; sgcc-detmine 保留 manifest/report 备查"
```

- [ ] **Step 3: 更新存档 + Commit**

```bash
# 更新 sgcc 状态存档（标注来源 YOLOE → 多模型+豆包）
cd /home/jiang/py/jxl
git add docs/  # 若有文档更新
git commit -m "docs: sgcc relabel 完成（多模型共识 + 豆包 grounding 替 YOLOE）" --allow-empty
```

---

## Self-Review 记录

- **Spec coverage**: §4 合并 → Task 1；§5 det_mine L1 → Task 2；§6 doubao_relabel → Task 3；§6 review 复检 + §7 合并 → Task 4；§7 验证 → Task 5。✓
- **Placeholder scan**: doubao_relabel 完整代码；其他 task 数据命令完整（Task 2/4 长任务后台 + Monitor）。Task 4 Step 1 `<KEY>` 是用户填 secret（不硬编码）。✓
- **Type consistency**: doubao_relabel 复用 rmb_ground `Backend/load_backend/parse_detections`（返回 Detection pydantic，d.bbox list[float]）；ground_one 返回 (Path, list, str|None) 一致。✓
- **依赖**: doubao_relabel import from rmb_ground（像 rmb_reannotate `from jxl.bin.rmb_ground import ...`）；豆包配置 --cfg（load_backend 读 json）。✓
