from pathlib import Path
from typing import Annotated

import typer
from jvi.geo.size2d import SIZE_FHD, Size
from jvi.video.capture import Capture
from loguru import logger

from jxl.det.a2d import from_d2d
from jxl.det.d2d import D2dOpt
from jxl.det.yolo.d2d_yoloe import D2dYoloE
from jxl.label.meta_dataset import MetaDataset
from jxl.yolo.util import yolo_set_weights_dir

app = typer.Typer(help="使用SAM模型从视频中提取标注")


def process_video(
    video_file: Path,
    dataset_dir: Path,
    names: str,
    meta_id: int = 0,
    fps: float = 1,
    size: Size = SIZE_FHD,
    min_conf: float = 0.4,
    max_conf: float = 0.8,
    iou_thr: float = 0.5,
    model_name: str = "yoloe-11l-seg.pt",
    weights_dir: Path | None = None,
) -> None:
    """处理视频文件, 提取帧并使用SAM模型进行目标检测和分割.

    该函数从指定视频文件中按给定帧率提取帧, 使用YOLO-E模型进行目标检测和分割,
    然后将检测和分割结果保存到指定的数据集目录中.

    Args:
        video_file: 输入视频文件路径
        dataset_dir: 输出数据集目录路径
        names: 目标类别名称, 多个类别用逗号分隔
        meta_id: 元数据ID, 默认为0
        fps: 提取视频帧的帧率, 默认为1fps
        size: 处理图像的大小, 默认为全高清(1920x1080)
        min_conf: 目标检测的最小置信度阈值, 默认为0.4
        max_conf: 最大置信度阈值, 超过此值的帧会被跳过, 默认为0.8
        iou_thr: 非极大值抑制的IOU阈值, 默认为0.5
        model_name: YOLO-E模型文件名, 默认为"yoloe-11l-seg.pt"
        weights_dir: 模型权重文件目录, 如果为None则使用默认目录

    Returns:
        None

    """
    video_file = video_file.resolve()
    logger.info("video_file: {}", video_file)

    weights_dir = weights_dir or Path("/home/jiang/py/jxl/models")
    yolo_set_weights_dir(str(weights_dir))

    capture = Capture(video_file)
    assert capture.is_opened()

    capture.set_fps(fps).unwrap()
    logger.info("fps: {}", fps)

    capture.set_size(size).unwrap()
    logger.info("size: {}", size)

    opt = D2dOpt(conf_thr=min_conf, iou_thr=iou_thr)

    model_file = Path(weights_dir, model_name)
    name_arr = names.split(",")
    model = D2dYoloE(model_file, opt, name_arr)

    a2d_set = MetaDataset(dataset_dir, "a2d", meta_id)

    n = 1
    total = 0
    while True:
        r = capture.read_frame()
        if r.is_null():
            break
        total += 1
        frame = r.unwrap()

        d2d_ret = model.detect(frame.data)

        min_conf = d2d_ret.min_conf()
        if min_conf > max_conf:
            logger.warning(
                f"frame_{frame.number} skipped, min_conf={min_conf:.2f} > {max_conf:.2f}"
            )
            continue

        a2d_ret = from_d2d(d2d_ret)

        p = video_file.parent
        name = f"{p.parent.name}_{p.name}_{video_file.stem}_{frame.number:04d}"
        logger.info(f"#{n} {name} {min_conf:.2f}")
        n += 1
        a2d_set.add_sample(name, frame.data, a2d_ret)
    logger.info(f"Done! add samples: {n}/{total}")


@app.command()
def main(  # noqa: PLR0913
    video_file: Annotated[Path, typer.Argument(help="输入视频文件路径")],
    dataset_dir: Annotated[Path, typer.Argument(help="输出数据集目录")],
    names: Annotated[str, typer.Argument(help="目标类别名称, 多个类别用逗号分隔")],
    meta_id: Annotated[int, typer.Option(help="元数据ID")] = 0,
    fps: Annotated[float, typer.Option(help="提取视频帧的帧率")] = 0.2,
    width: Annotated[int, typer.Option(help="处理图像宽度")] = SIZE_FHD.width,
    height: Annotated[int, typer.Option(help="处理图像高度")] = SIZE_FHD.height,
    min_conf: Annotated[float, typer.Option(help="最小置信度阈值")] = 0.3,
    max_conf: Annotated[
        float, typer.Option(help="最大置信度阈值, 超过此值的帧会被跳过")
    ] = 0.8,
    iou_thr: Annotated[float, typer.Option(help="IOU阈值")] = 0.5,
    model_name: Annotated[str, typer.Option(help="模型文件名")] = "yoloe-11l-seg.pt",
    weights_dir: Annotated[Path | None, typer.Option(help="权重文件目录")] = None,
) -> None:
    """使用SAM模型从视频中提取标注数据.

    此工具从视频中提取帧, 使用YOLO-E模型进行目标检测和分割,
    然后将结果保存到指定目录的数据集中.
    """
    size = Size(width=width, height=height)
    process_video(
        video_file=video_file,
        dataset_dir=dataset_dir,
        names=names,
        meta_id=meta_id,
        fps=fps,
        size=size,
        min_conf=min_conf,
        max_conf=max_conf,
        iou_thr=iou_thr,
        model_name=model_name,
        weights_dir=weights_dir,
    )


if __name__ == "__main__":
    app()
