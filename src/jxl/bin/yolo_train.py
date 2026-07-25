#!/usr/bin/env python3
"""YOLO 目标检测训练(ultralytics, 含最新 YOLO26).

项目通用训练入口: 调用 ultralytics 的 YOLO.train() 训练目标检测器.
首次使用某预训练权重(如 yolo26n.pt)时 ultralytics 自动下载.

project 默认绝对化(基于 cwd), 避免相对路径被 ultralytics 拼到 SETTINGS runs_dir
导致 save_dir 跑偏(曾出现输出误存到无关目录).

典型用法:
    # 从 COCO 预训练训 person 检测器(标准迁移, lr0=0.01 安全)
    yolo_train /path/to/dataset.yaml --model yolo26n.pt --epochs 200 --device 0

    # 微调已收敛 person.pt(必须降 lr0 + freeze backbone, 否则大 lr 破坏已学特征)
    yolo_train /path/to/dataset.yaml --model best.pt --lr0 0.001 --freeze 10 --epochs 100

    # 续训
    yolo_train /path/to/dataset.yaml --model runs/detect/train/weights/last.pt --resume
"""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from ultralytics import YOLO

# typer CLI 惯用模式: bool Option 默认值与参数校验异常消息豁免噪声规则

app = typer.Typer(help="YOLO 目标检测训练(ultralytics, 含 YOLO26)")


@app.command()
def main(
    data: Annotated[Path, typer.Argument(help="dataset.yaml 路径")],
    model: Annotated[
        str,
        typer.Option(help="预训练权重(yolo26n.pt/yolo26s.pt)或 .yaml 配置或 best.pt"),
    ] = "yolo26n.pt",
    epochs: Annotated[int, typer.Option(help="训练轮数")] = 100,
    imgsz: Annotated[int, typer.Option(help="输入图像尺寸")] = 640,
    batch: Annotated[int, typer.Option(help="batch size, -1=AutoBatch")] = -1,
    device: Annotated[str, typer.Option(help="设备: 0 / cpu / 0,1")] = "0",
    workers: Annotated[int, typer.Option(help="数据加载进程数")] = 8,
    lr0: Annotated[float, typer.Option(help="初始学习率(微调已收敛模型用 0.001)")] = 0.01,
    lrf: Annotated[float, typer.Option(help="最终 lr 系数(最终 lr = lr0*lrf)")] = 0.01,
    freeze: Annotated[int, typer.Option(help="冻结 backbone 前 N 层(微调用 10), 0=不冻")] = 0,
    optimizer: Annotated[str, typer.Option(help="优化器: auto/SGD/Adam/AdamW")] = "auto",
    cos_lr: Annotated[bool, typer.Option("--cos-lr", help="余弦学习率调度")] = False,
    close_mosaic: Annotated[int, typer.Option(help="最后 N 轮关闭 mosaic, 0=全程开")] = 10,
    project: Annotated[str, typer.Option(help="输出项目目录(自动绝对化)")] = "runs/detect",
    name: Annotated[str, typer.Option(help="实验名(输出子目录)")] = "train",
    patience: Annotated[int, typer.Option(help="早停耐心值, 0=关闭")] = 30,
    resume: Annotated[bool, typer.Option("--resume", help="从 last.pt 续训")] = False,
    exist_ok: Annotated[
        bool, typer.Option("--exist-ok", help="允许覆盖同名实验目录")
    ] = False,
) -> None:
    """用 ultralytics 训练 YOLO 目标检测模型."""
    if not data.is_file():
        raise typer.BadParameter(f"dataset.yaml 不存在: {data}")
    # 绝对化 project: 避免 ultralytics 把相对路径拼到 SETTINGS runs_dir 导致 save_dir 跑偏
    project_abs = str(Path(project).resolve())

    logger.info(
        "开始训练: model={} data={} epochs={} imgsz={} batch={} lr0={} freeze={} opt={} device={} -> {}",
        model,
        data,
        epochs,
        imgsz,
        batch,
        lr0,
        freeze,
        optimizer,
        device,
        project_abs,
    )

    model_obj = YOLO(model)
    model_obj.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        lr0=lr0,
        lrf=lrf,
        freeze=freeze or None,  # 0 -> None(不冻结, 对齐 ultralytics 默认)
        optimizer=optimizer,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        project=project_abs,
        name=name,
        patience=patience,
        resume=resume,
        exist_ok=exist_ok,
    )

    best = Path(project_abs) / name / "weights" / "best.pt"
    logger.info("训练完成, best.pt: {}", best)


if __name__ == "__main__":
    app()
