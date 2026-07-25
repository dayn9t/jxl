#!/usr/bin/env python3
"""MOT(MOT17/MOT20) -> YOLO 格式转换.

读 MOT train 序列(gt/gt.txt + img1/*.jpg), 筛 class 1/7(pedestrian + static person)
作为单一 person 类, MOT 绝对像素 xywh(左上角) -> YOLO 归一化中心 xywh,
输出 YOLO images/ + labels/. 文件名 {seq}_{frame:06d} 防多序列冲突.

class 1=pedestrian, 7=static person -> YOLO class 0(person).
其余(2骑车/3车/4自行车/8 distractor/12 crowd 等)丢弃.

MOT17 每序列有 DPM/FRCNN/SDP 三检测器变体(**图片相同**仅 gt 来源异), 用 --detector FRCNN
取一份避免重复; MOT20 序列名无检测器后缀(MOT20-01), detector 留空匹配所有 MOT 序列.

用法:
    mot_to_yolo /path/MOT17/train /path/out --detector FRCNN   # MOT17
    mot_to_yolo /path/MOT20/train /path/out                    # MOT20(无变体后缀)
"""

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from jcx.sys.fs import files_in
from loguru import logger
from PIL import Image

# typer CLI 惯用模式; assert 校验豁免噪声规则
app = typer.Typer(help="MOT17 -> YOLO 转换(person)")

PERSON_CLASSES = frozenset({1, 7})
"""MOT class 1=pedestrian / 7=static person -> YOLO person(0); 其余丢弃"""

IMG_EXT = ".jpg"


def parse_gt(gt_path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """读 MOT gt.txt, 按帧聚合 (x,y,w,h), 只留 PERSON_CLASSES."""
    by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        if int(p[7]) not in PERSON_CLASSES:
            continue
        frame = int(p[0])
        x, y, w, h = (float(v) for v in p[2:6])
        by_frame[frame].append((x, y, w, h))
    return by_frame


def to_yolo_line(
    box: tuple[float, float, float, float], iw: int, ih: int
) -> str | None:
    """MOT (x,y,w,h) 左上角绝对像素 -> YOLO '0 cx cy w h' 归一化中心.

    先 clamp 到图像边界(MOT gt 有少量越界标注), 全越界返回 None 丢弃.
    """
    x, y, w, h = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(iw), x + w), min(float(ih), y + h)
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:  # 全越界, 丢弃
        return None
    cx = (x1 + cw / 2) / iw
    cy = (y1 + ch / 2) / ih
    return f"0 {cx:.6f} {cy:.6f} {cw / iw:.6f} {ch / ih:.6f}"


@app.command()
def main(
    src_dir: Annotated[Path, typer.Argument(help="MOT train 目录(含 MOT*-*/序列)")],
    out_dir: Annotated[Path, typer.Argument(help="输出 YOLO 目录(images/+labels/)")],
    detector: Annotated[
        str | None,
        typer.Option(
            help="检测器变体(DPM/FRCNN/SDP, MOT17用); 留空匹配所有MOT序列(MOT20)"
        ),
    ] = None,
) -> None:
    """遍历 MOT train 序列, 转 YOLO person 检测格式."""
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    suffix = f"-{detector}" if detector else ""
    seqs = sorted(
        d
        for d in src_dir.iterdir()
        if d.is_dir() and d.name.startswith("MOT") and d.name.endswith(suffix)
    )
    assert seqs, f"未在 {src_dir} 找到 MOT-*{suffix} 序列目录"
    logger.info("序列 {} 个({}) -> {}", len(seqs), detector, out_dir)

    n_img = n_lbl = n_box = 0
    for seq in seqs:
        gt = parse_gt(seq / "gt" / "gt.txt")
        imgs = sorted(files_in(seq / "img1", IMG_EXT))
        if not imgs:
            logger.warning("{} 无图片, 跳过", seq.name)
            continue
        iw, ih = Image.open(imgs[0]).size  # 序列内尺寸固定, 读首帧

        seq_box = 0
        for img in imgs:
            frame = int(img.stem)  # 000001 -> 1
            boxes = gt.get(frame, [])
            label_name = f"{seq.name}_{frame:06d}.txt"
            if boxes:  # 有标注才写 label(无标注帧不写, YOLO 视为负样本)
                lines = [ln for b in boxes if (ln := to_yolo_line(b, iw, ih))]
                if lines:
                    (out_lbl / label_name).write_text(
                        "\n".join(lines) + "\n", encoding="utf-8"
                    )
                    n_lbl += 1
                    seq_box += len(lines)
            dst_img = out_img / f"{seq.name}_{frame:06d}{IMG_EXT}"
            if not dst_img.exists():
                shutil.copy(img, dst_img)
            n_img += 1
        n_box += seq_box
        logger.info("{}: {} 帧, {} 框, {}x{}", seq.name, len(imgs), seq_box, iw, ih)

    logger.info(
        "完成: {} 序列 -> {} 图 / {} label / {} person 框 -> {}",
        len(seqs),
        n_img,
        n_lbl,
        n_box,
        out_dir,
    )


if __name__ == "__main__":
    app()
