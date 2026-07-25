#!/usr/bin/env python3
"""将 CrowdHuman odgt 标注转为 YOLO person 检测集。

读 annotation.odgt（每行一图 JSON: ID + gtboxes[].{tag,fbox,vbox,hbox,head_attr}），
筛 tag=person + head_attr.ignore=0 + unsure=0，fbox 归一化(/img_w,/img_h) → YOLO cls。
输出 images/(symlink) + labels/*.txt。

CrowdHuman 密集人群，含 sitting/crouching/bending 等非站立姿态 + 严重遮挡，
补 person.pt（sgcc+MOT+COCO，站立/行走为主）的姿态与密集多样性。

用法:
    crowdhuman_to_yolo <odgt> <images_dir> <out_dir> [--cls-id 0]
"""

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

app = typer.Typer(add_completion=False, help="CrowdHuman odgt → YOLO person 检测集。")


@app.command()
def main(
    odgt: Annotated[Path, typer.Argument(help="annotation.odgt")],
    images_dir: Annotated[Path, typer.Argument(help="解压后的 Images 目录")],
    out_dir: Annotated[Path, typer.Argument(help="输出 YOLO 集目录")],
    cls_id: Annotated[int, typer.Option("--cls-id", help="YOLO 类 id")] = 0,
    copy: Annotated[bool, typer.Option("--copy", help="复制图(默认 symlink)")] = False,
) -> None:
    """读 odgt + Images，输出 YOLO images/+labels/（fbox 归一化，筛 person+ignore0+unsure0）。"""
    if not odgt.is_file():
        typer.secho(f"odgt 不存在: {odgt}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    n_img = n_box = n_skip = 0
    for line in odgt.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        img_id = rec["ID"]
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.is_file():
            n_skip += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        label_lines: list[str] = []
        for gb in rec.get("gtboxes", []):
            if gb.get("tag") != "person":
                continue
            ha = gb.get("head_attr", {})
            if ha.get("ignore", 0) or ha.get("unsure", 0):
                continue
            x, y, bw, bh = gb["fbox"]  # xywh 像素
            cx = max(0.0, min(1.0, (x + bw / 2) / w))
            cy = max(0.0, min(1.0, (y + bh / 2) / h))
            nw = max(0.0, min(1.0, bw / w))
            nh = max(0.0, min(1.0, bh / h))
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1
        if not label_lines:
            continue
        dst_img = out_dir / "images" / f"{img_id}.jpg"
        if copy:
            shutil.copy2(img_path, dst_img)
        elif not dst_img.exists():
            dst_img.symlink_to(img_path.resolve())
        (out_dir / "labels" / f"{img_id}.txt").write_text(
            "\n".join(label_lines) + "\n", encoding="utf-8"
        )
        n_img += 1
    typer.secho(
        f"图 {n_img} | 框 {n_box} | 跳过(图缺失) {n_skip} → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
