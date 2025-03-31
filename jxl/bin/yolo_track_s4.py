from pathlib import Path
from time import sleep
from typing import Final

import cv2  # type: ignore
from jcx.text.txt_json import to_json, load_json
from js4.task import TaskDb, D2dParams
from loguru import logger

from jxl.yolo_track_service import track_videos

START_PROGRESS: Final = 1
END_PROGRESS: Final = 10


def main(wait: int):
    db_dir = Path("/opt/howell/s4/current/ias/domain/d1/n1/db")
    d2d_cfg_file = db_dir / "analyzer_d2d/111.json"
    dst_dir = Path("/var/howell/s4/ias/track")
    fps = 1.25

    logger.info("db_dir: {}", db_dir)
    logger.info("dst_dir: {}", dst_dir)
    db = TaskDb(db_dir, START_PROGRESS, END_PROGRESS)
    # db.show()
    d2d_cfg = load_json(d2d_cfg_file, D2dParams).unwrap()

    while True:
        res = db.find_task()
        if res.is_some():
            task, status = res.unwrap()
            logger.info("find task: {}", to_json(task))
            if track_videos(
                task.data_urls, dst_dir / str(task.id), d2d_cfg, fps, wait=wait
            ):
                db.task_done(task.id)
        else:
            logger.info("no task")

        sleep(5)


if __name__ == "__main__":
    # catch_show_err(main)
    # typer.run(main)
    main(-1)
