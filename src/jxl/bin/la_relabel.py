#!/usr/bin/env python3
"""La-Relabel: LocateAnything 全量重标注 → YOLO labels（评估用途）。

复用 det_mine.detect_la 逐批检测 → labels/<stem>.txt(cls=0)，产出独立评估目录
（不替换训练 labels）。断点续跑: 已有 label 的 stem 跳过; 批为最小进度单位,
服务中断重跑仅损失当前批。坏图(单图失败)记录 _errors.jsonl, 重跑自动重试。

NVIDIA License 非商用——产出仅研究/评估链路, 不进训练管线。

用法:
    la_relabel <images_dir> <out_dir> --target person
    # 续跑: 同命令再执行(已标注 stem 自动跳过, _errors.jsonl 为本次运行失败清单)
"""

import time
from pathlib import Path
from typing import Annotated

import orjson
import typer

from jxl.bin.det_mine import detect_la, gather_images
from jxl.det.hardmine import to_yolo_label
from jxl.det.locateanything.client import LA_DEFAULT_URL, LaServerDownError

app = typer.Typer(
    add_completion=False, help="LocateAnything 全量重标注(评估用)."
)


@app.command()
def run(
    images_dir: Annotated[Path, typer.Argument(help="图片目录(递归)")],
    out_dir: Annotated[Path, typer.Argument(help="输出目录(labels/ + report)")],
    target: Annotated[str, typer.Option("--target", help="检测目标文本")] = "person",
    la_url: Annotated[
        str,
        typer.Option("--la-url", help="LocateAnything 服务地址(script/la-serve.sh)"),
    ] = LA_DEFAULT_URL,
    batch: Annotated[int, typer.Option("--batch", help="批大小(断点粒度)")] = 200,
    cls_id: Annotated[int, typer.Option("--cls-id", help="YOLO 标注类 id")] = 0,
) -> None:
    """LocateAnything 全量检测 → YOLO labels(独立评估目录)."""
    if batch < 1:
        typer.secho(f"--batch 须 >=1: {batch}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    images_dir = images_dir.expanduser().resolve()
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    imgs = gather_images(images_dir)
    if not imgs:
        typer.secho(f"目录无图: {images_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    todo = [p for p in imgs if not (labels_dir / (p.stem + ".txt")).exists()]
    resume_skipped = len(imgs) - len(todo)
    typer.secho(
        f"la_relabel target={target} total={len(imgs)} "
        f"resume_skip={resume_skipped} todo={len(todo)}",
        fg=typer.colors.CYAN,
    )

    t0 = time.monotonic()
    n_written = 0
    n_empty = 0
    errors: list[str] = []
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        try:
            detected = detect_la(chunk, la_url, target)
        except LaServerDownError:
            typer.secho(
                "la 服务中断: 已写进度保留, 重跑同命令续跑",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(2) from None
        for p in chunk:
            boxes = detected.get(p.stem)
            if boxes is None:  # 坏图/单图失败: 无 label, 下次续跑自动重试
                errors.append(p.name)
                continue
            (labels_dir / (p.stem + ".txt")).write_text(
                to_yolo_label(boxes, cls_id=cls_id), encoding="utf-8"
            )
            n_written += 1
            if not boxes:
                n_empty += 1
        done = resume_skipped + i + len(chunk)
        speed = (i + len(chunk)) / (time.monotonic() - t0)
        remain = len(todo) - (i + len(chunk))
        typer.secho(
            f"[{done}/{len(imgs)}] written={n_written} empty={n_empty} err={len(errors)} "
            f"speed={speed:.2f}img/s eta={remain / speed / 3600:.2f}h",
            fg=typer.colors.GREEN,
        )

    (out_dir / "_errors.jsonl").write_text(
        "\n".join(orjson.dumps({"image": e}).decode() for e in errors)
        + ("\n" if errors else ""),
        encoding="utf-8",
    )
    report = {
        "target": target,
        "la_url": la_url,
        "total": len(imgs),
        "resume_skipped": resume_skipped,
        "written": n_written,
        "empty_labels": n_empty,
        "errors": len(errors),
        "batch": batch,
    }
    (out_dir / "relabel_report.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2)
    )
    typer.secho(
        f"完成: written={n_written}(空标 {n_empty}) errors={len(errors)} → {out_dir}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
