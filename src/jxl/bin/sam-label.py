from pathlib import Path

from jvi.geo.size2d import Size, SIZE_FHD
from jvi.video.capture import Capture
from loguru import logger

from jxl.det.a2d import from_d2d
from jxl.det.d2d import D2dOpt
from jxl.det.yolo.d2d_yoloe import D2dYoloE
from jxl.label.meta_dataset import MetaDataset


def main(
    video_file: Path,
    dataset_dir: Path,
    names: str,
    meta_id: int = 0,
    fps: float = 1,
    size: Size = SIZE_FHD,
    min_conf: float = 0.4,
    max_conf: float = 0.8,
    iou_thr: float = 0.5,
) -> None:
    logger.info("video_file: {}", video_file)

    capture = Capture(video_file)
    assert capture.is_opened()

    capture.set_fps(fps).unwrap()
    logger.info("fps: {}", fps)

    capture.set_size(size).unwrap()
    logger.info("size: {}", size)

    opt = D2dOpt(conf_thr=min_conf, iou_thr=iou_thr)

    model_name = "yoloe-11l-seg.pt"
    model_file = Path("/home/jiang/py/jxl/models/yoloe", model_name)
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

        name = f"{video_file.parent.name}_{video_file.stem}_{frame.number:04d}"
        logger.info(f"#{n} {name} {min_conf:.2f}")
        n += 1
        a2d_set.add_sample(name, frame.data, a2d_ret)
    logger.info(f"Done! add samples: {n}/{total}")


if __name__ == "__main__":
    video_file = Path(
        "/var/www/static/projects/sgcc/video/柜员后视角/2024-03-15/08-30-57.mp4"
    )
    dataset_dir = Path("/home/jiang/py/jxl/dist/2024-03-15")
    names = "person"
    main(video_file, dataset_dir, names, fps=0.1)
