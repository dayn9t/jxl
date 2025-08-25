from pathlib import Path

import typer
from jvi.geo.size2d import SIZE_FHD, Size
from jvi.video.capture import Capture
from loguru import logger

from jxl.det.d2d import D2dOpt

app = typer.Typer(help="使用SAM模型从视频中提取标注")


def main(
    video_file: Path,
    model_file: Path,
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
    video_file = video_file.resolve()
    logger.info("video_file: {}", video_file)

    capture = Capture(video_file)
    assert capture.is_opened()

    capture.set_fps(fps).unwrap()
    logger.info("fps: {}", fps)

    capture.set_size(size).unwrap()
    logger.info("size: {}", size)

    det_opt = D2dOpt(conf_thr=min_conf, iou_thr=iou_thr)

    detector = D2dYolo(model_file, det_opt)

    while True:
        r = capture.read_frame()
        if r.is_null():
            break
        frame = r.unwrap()
        logger.info(f"frame_{frame.number} pos={frame.position}")

        d2d_ret = model.detect(frame.data)

        min_conf = d2d_ret.min_conf()
        if min_conf > max_conf:
            continue

    logger.info("Done!")


if __name__ == "__main__":
    main()
