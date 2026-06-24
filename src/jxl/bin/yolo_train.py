#!/home/jiang/py/jxl/.venv/bin/python
"""YOLO 目标检测训练(ultralytics, 含最新 YOLO26).

项目通用训练入口: 调用 ultralytics 的 YOLO.train() 训练目标检测器.
首次使用某预训练权重(如 yolo26s.pt)时 ultralytics 自动下载.

典型用法:
    # 用 YOLO26s 训练 person 检测器
    yolo_train /path/to/dataset.yaml --model yolo26s.pt --epochs 100 --device 0

    # 续训
    yolo_train /path/to/dataset.yaml --model runs/detect/train/weights/last.pt --resume
"""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from ultralytics import YOLO

# typer CLI 惯用模式: bool Option 默认值与参数校验异常消息豁免噪声规则
# ruff: noqa: FBT002, TRY003, EM102

app = typer.Typer(help="YOLO 目标检测训练(ultralytics, 含 YOLO26)")


@app.command()
def main(  # noqa: PLR0913
    data: Annotated[Path, typer.Argument(help="dataset.yaml 路径")],
    model: Annotated[
        str,
        typer.Option(help="预训练权重(yolo26s.pt/yolo26n.pt)或模型 .yaml 配置"),
    ] = "yolo26s.pt",
    epochs: Annotated[int, typer.Option(help="训练轮数")] = 100,
    imgsz: Annotated[int, typer.Option(help="输入图像尺寸")] = 640,
    batch: Annotated[int, typer.Option(help="batch size, -1=自动")] = -1,
    device: Annotated[str, typer.Option(help="设备: 0 / cpu / 0,1")] = "0",
    workers: Annotated[int, typer.Option(help="数据加载进程数")] = 8,
    project: Annotated[str, typer.Option(help="输出项目目录")] = "runs/detect",
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

    logger.info(
        "开始训练: model={} data={} epochs={} imgsz={} batch={} device={}",
        model,
        data,
        epochs,
        imgsz,
        batch,
        device,
    )

    model_obj = YOLO(model)
    model_obj.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=name,
        patience=patience,
        resume=resume,
        exist_ok=exist_ok,
    )

    best = Path(project) / name / "weights" / "best.pt"
    logger.info("训练完成, best.pt: {}", best)


if __name__ == "__main__":
    app()
