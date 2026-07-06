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

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def gather_images(src: Path) -> list[Path]:
    """递归收集候选帧图片。"""
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def _detect(
    model: YOLO | YOLOE,
    paths: list[Path],
    conf: float,
    iou: float,
    device: str,
) -> dict[str, list[Box]]:
    """通用 ultralytics 检测 → {stem: [Box]}。用 res.path 反查 stem，杜绝 stream 错位。

    stream 模式下 ultralytics 对损坏图可能静默跳过（yield 少于输入数）；
    用 res.path 反查保证 stem↔框 正确配对，未返回的图在 run 里计为 skipped。
    person.pt 单类 YOLO 直接 predict；YOLOE 由调用方先 set_classes 再传入。
    Box 坐标取 boxes.xyxyn（归一化）。
    """
    kwargs: dict[str, object] = {"conf": conf, "iou": iou, "verbose": False, "stream": True}
    if device:
        kwargs["device"] = device
    out: dict[str, list[Box]] = {}
    for res in model.predict([str(p) for p in paths], **kwargs):
        boxes: list[Box] = []
        if res.boxes is not None and len(res.boxes):
            xy = res.boxes.xyxyn
            cf = res.boxes.conf
            for i in range(len(xy)):
                b = xy[i].tolist()
                boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(cf[i])))
        out[Path(res.path).stem] = boxes
    return out


def detect_yoloe(
    paths: list[Path],
    model_path: Path,
    conf: float,
    iou: float,
    device: str,
) -> dict[str, list[Box]]:
    """YOLOE 开放词汇检测 person: set_classes('person') 后 predict。

    YOLOE 为 prompt-based 模型，必须先 set_classes + get_text_pe 提供文本提示，
    否则 predict 不输出目标（与普通 YOLO 固定 COCO 类不同）。参考 d2d_yoloe.py。
    """
    model = YOLOE(str(model_path))
    model.set_classes(["person"], model.get_text_pe(["person"]))
    return _detect(model, paths, conf, iou, device)


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
    if not 0.0 <= iou <= 1.0:
        typer.secho(f"--iou 须在 [0,1]: {iou}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not 0.0 <= conf <= 1.0:
        typer.secho(f"--conf 须在 [0,1]: {conf}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
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
    yoloe_map = detect_yoloe(imgs, yoloe_model, conf, iou, device)

    counts: Counter[str] = Counter()
    by_video: dict[str, Counter[str]] = {}
    skipped = 0
    for img in imgs:
        stem = img.stem
        pb = person_map.get(stem)
        yb = yoloe_map.get(stem)
        if pb is None and yb is None:
            # 两检测器都未返回（损坏图被 ultralytics 静默跳过）→ 计 skipped，不分类
            skipped += 1
            continue
        cls = classify_sample(pb or [], yb or [], iou)
        counts[cls.value] += 1
        # by_video 分组依赖 mkv_keyframes 的 {video_stem}_{idx:06d} 命名
        video = stem.rsplit("_", 1)[0] if "_" in stem else stem
        by_video.setdefault(video, Counter())[cls.value] += 1
        if cls is SampleClass.POSITIVE:
            write_yolo_sample(out_dir, img, yb or [])
        elif cls is SampleClass.NEGATIVE:
            write_yolo_sample(out_dir, img, None)

    report = {
        "total_frames": len(imgs),
        "skipped": skipped,
        "positive": counts.get("positive", 0),
        "negative": counts.get("negative", 0),
        "dropped_empty": counts.get("drop_empty", 0),
        "dropped_agree": counts.get("drop_agree", 0),
        "by_video": {k: dict(v) for k, v in by_video.items()},
    }
    (out_dir / "mining_report.json").write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    typer.secho(
        f"正样本 {report['positive']} | 负样本 {report['negative']} | "
        f"丢弃(空/一致) {report['dropped_empty']}/{report['dropped_agree']} | "
        f"跳过 {skipped} → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
