#!/usr/bin/env python3
"""视频抽帧数据集时序下采样(去相邻帧冗余).

MOT17 等连续视频抽帧数据集, 相邻帧相似度极高, 作检测样本冗余大.
按文件名序列前缀分组, 组内按帧号排序, 每 stride 帧保留 1(其余删 images+labels).

文件名约定: {seq}_{frame}.jpg (如 MOT17-02-FRCNN_000001), rsplit('_',1) 分组.
缺帧时按组内序号 index%stride 采样(不依赖帧号连续, 均匀).

用法:
    frame_sample /path/yolo_dataset --stride 5
    frame_sample /path/yolo_dataset --stride 5 --dry-run  # 只看不删
"""

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from jcx.sys.fs import files_in
from loguru import logger

# typer CLI 惯用模式
app = typer.Typer(help="视频抽帧时序下采样(相邻帧去冗余)")

IMG_EXT = ".jpg"


def parse_name(stem: str) -> tuple[str, int] | None:
    """{seq}_{frame} -> (seq, frame); 帧尾非数字返回 None(不参与采样)."""
    idx = stem.rfind("_")
    if idx < 0:
        return None
    tail = stem[idx + 1 :]
    return (stem[:idx], int(tail)) if tail.isdigit() else None


@app.command()
def main(
    src_dir: Annotated[Path, typer.Argument(help="YOLO 数据集目录(images/+labels/)")],
    stride: Annotated[int, typer.Option(help="采样步长(每 N 帧留 1)")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="只统计不删除")] = False,
) -> None:
    """按序列分组 stride 采样, 删除冗余相邻帧(images + 对应 labels)."""
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"
    assert images_dir.is_dir(), f"无 {images_dir}"

    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    skipped = 0
    for img in files_in(images_dir, IMG_EXT):
        parsed = parse_name(img.stem)
        if parsed is None:
            skipped += 1
            continue
        seq, frame = parsed
        groups[seq].append((frame, img))

    keep_total = drop_total = 0
    for seq in sorted(groups):
        items = sorted(groups[seq])  # 按帧号 [(frame, img), ...]
        keep_frames = {items[i][0] for i in range(0, len(items), stride)}
        k = len(keep_frames)
        d = len(items) - k
        keep_total += k
        drop_total += d
        logger.info("{}: {} 帧 -> 留 {} 删 {}", seq, len(items), k, d)
        if dry_run:
            continue
        for frame, img in items:
            if frame in keep_frames:
                continue
            img.unlink()
            lbl = labels_dir / f"{img.stem}.txt"
            if lbl.is_file():
                lbl.unlink()

    action = "[dry-run] " if dry_run else "已删除 "
    logger.info(
        "{}{} 序列 | 留 {} 删 {} (stride={}){}",
        action,
        len(groups),
        keep_total,
        drop_total,
        stride,
        f" | 跳过 {skipped} 无帧号文件" if skipped else "",
    )


if __name__ == "__main__":
    app()
