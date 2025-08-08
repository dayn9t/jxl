from pathlib import Path
from typing import List, Optional, Annotated

from jvi.geo.size2d import Size, SIZE_FHD
from jvi.video.capture import Capture
from loguru import logger
import typer

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
    weights_dir: Optional[Path] = None,
) -> None:
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

        name = f"{video_file.parent.parent.name}_{video_file.parent.name}_{video_file.stem}_{frame.number:04d}"
        logger.info(f"#{n} {name} {min_conf:.2f}")
        n += 1
        a2d_set.add_sample(name, frame.data, a2d_ret)
    logger.info(f"Done! add samples: {n}/{total}")


@app.command()
def main(
    video_file: Annotated[Path, typer.Argument(help="输入视频文件路径")],
    dataset_dir: Annotated[Path, typer.Argument(help="输出数据集目录")],
    names: Annotated[str, typer.Argument(help="目标类别名称，多个类别用逗号分隔")],
    meta_id: Annotated[int, typer.Option(help="元数据ID")] = 0,
    fps: Annotated[float, typer.Option(help="提取视频帧的帧率")] = 1.0,
    width: Annotated[int, typer.Option(help="处理图像宽度")] = SIZE_FHD.width,
    height: Annotated[int, typer.Option(help="处理图像高度")] = SIZE_FHD.height,
    min_conf: Annotated[float, typer.Option(help="最小置信度阈值")] = 0.4,
    max_conf: Annotated[
        float, typer.Option(help="最大置信度阈值，超过此值的帧会被跳过")
    ] = 0.8,
    iou_thr: Annotated[float, typer.Option(help="IOU阈值")] = 0.5,
    model_name: Annotated[str, typer.Option(help="模型文件名")] = "yoloe-11l-seg.pt",
    weights_dir: Annotated[Optional[Path], typer.Option(help="权重文件目录")] = None,
) -> None:
    """
    使用SAM模型从视频中提取标注数据。

    此工具从视频中提取帧，使用YOLO-E模型进行目标检测和分割，
    然后将结果保存到指定目录的数据集中。
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
