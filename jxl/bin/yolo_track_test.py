from pathlib import Path
from time import sleep, time
from typing import Final

import cv2
from jcx.text.txt_json import to_json, load_json
from js4.task import TaskDb, D2dParams
from loguru import logger

from jxl.yolo_track_service import track_videos

START_PROGRESS: Final = 1
END_PROGRESS: Final = 10


def main(wait: int):
    d2d_cfg_file = Path("/home/jiang/py/jxl/assets/meta/d2d_cfg.json")
    # model_dir = Path("/opt/howell/s4/current/ias/model/")
    model_dir = Path("/home/jiang/1/model")
    dst_dir = Path("/home/jiang/1/track")
    fps = 1.25
    urls = [
        "file:///home/jiang/1/2025_03_31/C3_2025_03_31T10_47_18_R.mkv",
        "file:///home/jiang/1/2025_03_31/C3_2025_03_31T10_42_17_R.mkv",
        "file:///home/jiang/1/2025_03_31/C2_2025_03_31T10_47_18_L.mkv",
        "file:///home/jiang/1/2025_03_31/C2_2025_03_31T10_37_16_L.mkv",
        "file:///home/jiang/1/2025_03_31/C3_2025_03_31T10_17_12_R.mkv",
        "file:///home/jiang/1/2025_03_31/C2_2025_03_31T10_27_14_L.mkv",
        "file:///home/jiang/1/2025_03_31/C2_2025_03_31T10_17_12_L.mkv",
        "file:///home/jiang/1/2025_03_31/C3_2025_03_31T10_37_16_R.mkv",
        "file:///home/jiang/1/2025_03_31/C2_2025_03_31T10_32_15_L.mkv",
        "file:///home/jiang/1/2025_03_31/C3_2025_03_31T10_32_15_R.mkv",
        "file:///home/jiang/1/2025_03_31/C3_2025_03_31T10_22_13_R.mkv",
        "file:///home/jiang/1/2025_03_31/C2_2025_03_31T10_42_17_L.mkv",
        "file:///home/jiang/1/2025_03_31/C3_2025_03_31T10_27_14_R.mkv",
        "file:///home/jiang/1/2025_03_31/C2_2025_03_31T10_22_13_L.mkv",
    ]

    logger.info("d2d_cfg_file: {}", d2d_cfg_file)
    logger.info("dst_dir: {}", dst_dir)
    # db.show()
    d2d_cfg = load_json(d2d_cfg_file, D2dParams).unwrap()

    start_time = time()
    for _ in range(3):
        track_videos(urls, dst_dir, d2d_cfg, model_dir, fps, wait=wait)
    elapsed_time = time() - start_time
    logger.info("track_videos函数执行时间: {:.2f}秒", elapsed_time)


if __name__ == "__main__":
    # catch_show_err(main)
    # typer.run(main)
    main(-1)
