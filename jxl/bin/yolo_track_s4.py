from pathlib import Path
from time import sleep
from typing import List

import cv2  # type: ignore
import cv2
from jcx.text.txt_json import save_json, to_json
from jcx.ui.key import Key
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import SIZE_HD, SIZE_FHD
from jvi.image.image_nda import ImageNda
from jvi.image.proc import resize
from jvi.image.util import make_mask, copy_mask
from loguru import logger

from jxl.det.a2d import A2dOpt
from jxl.det.d2d import D2dOpt
from jxl.det.d2d import draw_d2d_objects
from jxl.det.yolo.a2d_yolo import A2dYolo
from jxl.task_s4 import TaskDb, TaskInfo


def track_videos(src_files: List[str], dst_dir: Path, wait: int = -1):
    model_dir = Path("/opt/howell/s4/current/ias/model/")

    img_size = 640
    conf_thr = 0.5
    iou_thr = 0.7
    roi = Rect(x=0.0, y=0.0, width=1.0, height=0.94).vertexes()

    src_files.sort()

    d2d_opt = D2dOpt(
        input_shape=(img_size, img_size), conf_thr=conf_thr, iou_thr=iou_thr, track=True
    )
    a2d_opt = A2dOpt(d2d=d2d_opt, d2d_name="sign.pt", props={0: "sign-valid.pt"})
    analyzer = A2dYolo(model_dir, a2d_opt)

    out_size = SIZE_HD
    canvas = ImageNda(out_size)
    mask = make_mask(SIZE_FHD, roi)

    for src_file in src_files:
        logger.info("Open file: {}", src_file)
        cap = cv2.VideoCapture(src_file)

        out_dir = dst_dir / Path(src_file).name
        out_dir.mkdir(parents=True, exist_ok=True)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            # timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # if number % 4 != 0:
            #    continue
            image_ori = ImageNda(data=frame)
            # logger.debug(f"#{number} size: {image_in.size()}")
            assert image_ori.size() == SIZE_FHD
            image_in = copy_mask(image_ori, mask)
            res = analyzer.detect(image_in)

            resize(image_in, canvas)
            if len(res.objects) > 0:
                draw_d2d_objects(canvas, res.objects)

            if wait >= 0:
                cv2.imshow("win", canvas.data())
                if cv2.waitKey(wait) == Key.ESC.value:
                    break
            if number % 20 == 0:
                n = number // 20
                name = f"{n:04d}"
                image_path = Path(out_dir, f"{name}.jpg")
                meta_path = image_path.with_suffix(".json")
                logger.info(f"#{number} {image_path}")
                image_in.save(image_path)
                res.roi = roi
                save_json(res, meta_path)

        cap.release()
    cv2.destroyAllWindows()


def main():
    db_dir = Path("/opt/howell/s4/current/ias/domain/d1/n1/db")
    dst_dir = Path("/var/howell/s4/ias/track")
    logger.info("db_dir: {}", db_dir)
    logger.info("dst_dir: {}", dst_dir)
    db = TaskDb(db_dir)
    # db.show()

    while True:
        res = db.find_task()
        if res.is_some():
            task: TaskInfo = res.unwrap()
            logger.info("find task: {}", to_json(task))
            track_videos(task.data_urls, dst_dir / str(task.id))
            db.task_done(task.id)
        else:
            logger.info("no task")

        sleep(5)


if __name__ == "__main__":
    # catch_show_err(main)
    # typer.run(main)
    main()
