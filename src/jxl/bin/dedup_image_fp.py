#!/usr/bin/env python3
"""整图姿态指纹去重(场景A).

监控背景完全相同, 整图 pHash/CLIP 被背景主导. 改用前景姿态指纹:
每张图 = 其前景 crop 的 SemDeDup 簇 id 集合(frozenset). 相同姿态组合的图视为重复
(同实例同姿态的相邻帧), 留一张代表. 簇 id 来自场景B dedup_sem 的 sem_cluster_map.
整图 + .txt 标注一起留.

无前景 crop 的图(负样本/空标, 或退化极小 bbox)原样保留: 前景去重只作用于有前景
的正样本, 负样本对训练(假阳抑制)重要, 不应丢弃.

典型用法:
    dedup_image_fp /path/crops_dedup /path/samples /path/samples_dedup
"""

import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger

# typer CLI 惯用模式: 参数校验异常消息豁免噪声规则
# ruff: noqa: TRY003, EM102

app = typer.Typer(help="整图姿态指纹去重(忽略相同背景)")

CROP_RE = re.compile(r"(.*)_p\d+\.jpg$")
"""前景 crop 文件名: <原图stem>_p<序号>.jpg"""


@app.command()
def main(
    dedup_dir: Annotated[
        Path,
        typer.Argument(
            help="crops_dedup 目录(含 sem_cluster_map.npy + sem_cluster_files.txt)"
        ),
    ],
    samples_dir: Annotated[Path, typer.Argument(help="samples 目录(images + labels)")],
    out_dir: Annotated[Path, typer.Argument(help="去重后输出(samples_dedup)")],
) -> None:
    """用前景姿态簇集合做图指纹, 相同指纹图留 1 代表."""
    sem_path = dedup_dir / "sem_cluster_map.npy"
    files_path = dedup_dir / "sem_cluster_files.txt"
    if not sem_path.is_file() or not files_path.is_file():
        raise typer.BadParameter(f"缺 sem_cluster_map.npy/.txt, 先跑 person_dedup: {dedup_dir}")

    sem = np.load(sem_path)
    files = files_path.read_text(encoding="utf-8").splitlines()
    assert len(files) == len(sem), f"sem {len(sem)} != files {len(files)}"
    logger.info("加载 sem_cluster_map: {} crop", len(files))

    # 原图 stem -> {SemDeDup 簇 id}
    stem2sem: dict[str, set[int]] = defaultdict(set)
    for fname, cid in zip(files, sem, strict=False):
        m = CROP_RE.match(fname)
        if m:
            stem2sem[m.group(1)].add(int(cid))
    logger.info("覆盖原图 stem: {}", len(stem2sem))

    image_dir = samples_dir / "images"
    label_dir = samples_dir / "labels"
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    seen_fp: set[frozenset] = set()
    n_imgs = kept = no_crop = 0
    for img in sorted(image_dir.glob("*.jpg")):
        n_imgs += 1
        cids = stem2sem.get(img.stem)
        lbl = label_dir / f"{img.stem}.txt"
        if not cids:
            # 无 crop: 负样本(空标/无人)或退化标注(bbox 极小无法截).
            # 前景去重只作用于有 person 的正样本; 无前景图原样保留(负样本对训练重要).
            shutil.copy(img, out_img / img.name)
            if lbl.is_file():
                shutil.copy(lbl, out_lbl / lbl.name)
            no_crop += 1
            continue
        fp = frozenset(cids)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        shutil.copy(img, out_img / img.name)
        if lbl.is_file():
            shutil.copy(lbl, out_lbl / lbl.name)
        kept += 1

    logger.info(
        "完成: {} 图 -> 去重正样本 {} + 无前景保留 {} | -> {}",
        n_imgs,
        kept,
        no_crop,
        out_dir,
    )


if __name__ == "__main__":
    app()
