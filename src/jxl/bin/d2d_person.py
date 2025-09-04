from pathlib import Path

import typer
from jcx.text.txt_json import save_json
from jvi.geo.point2d import Point
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import SIZE_HD, Size
from jvi.video.capture import Capture
from loguru import logger
from pydantic import BaseModel

from jxl.det.d2d import D2dOpt
from jxl.det.yolo.d2d_yolo import D2dYolo

app = typer.Typer(help="使用SAM模型从视频中提取标注")


class TimestampList(BaseModel):
    periods: list[tuple[int, int]]


def main(
    video_file: Path,
    model_file: Path,
    fps: float = 1,
    size: Size = SIZE_HD,
    min_conf: float = 0.4,
    iou_thr: float = 0.5,
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

    p1 = Point(x=0.2869791666666667, y=0.21428571428571427)
    p2 = Point(x=0.7645833333333333, y=0.43214285714285716)
    roi = Rect.from_ltrb(p1, p2)

    periods = []

    start = None
    while True:
        r = capture.read_frame()
        if r.is_null():
            break
        frame = r.unwrap()
        # logger.info(f"frame_{frame.number} pos={frame.position}")

        d2d_ret = detector.detect(frame.data)
        found_person = False
        for ob in d2d_ret.objects:
            if roi.contains(ob.rect.center()):
                found_person = True
        if found_person:
            logger.info(f" Found {ob}")
            if start is None:
                start = frame.position
        else:
            logger.info("Not Found")
            if start is not None:
                periods.append((start, frame.position))
                start = None
        # trace_image(frame.data)

    meta_data = TimestampList(periods=periods)
    meta_file = video_file.with_suffix(".json")
    save_json(meta_data, meta_file).unwrap()
    logger.info("Done!")


if __name__ == "__main__":
    f = Path("/home/jiang/ws/sgcc/video/top/09-56-14.mp4")
    m = Path("/home/jiang/ws/sgcc/person/model_dir/person.pt")
    main(f, m)
