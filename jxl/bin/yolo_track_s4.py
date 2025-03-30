from pathlib import Path
from time import sleep
from typing import List

import cv2  # type: ignore
import cv2
from jcx.text.txt_json import save_json, to_json, load_json
from jcx.ui.key import Key
from jvi.drawing.color import YOLO_GRAY
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import SIZE_HD, SIZE_FHD
from jvi.image.image_nda import ImageNda
from jvi.image.proc import resize
from jvi.image.util import make_mask, copy_mask
from loguru import logger

from jxl.det.a2d import A2dOpt
from jxl.det.d2d import D2dOpt, D2dResult
from jxl.det.d2d import draw_d2d_objects
from jxl.det.yolo.a2d_yolo import A2dYolo
from jxl.task_s4 import TaskDb, TaskInfo, D2dParams


def display_and_input(image_in: ImageNda, canvas: ImageNda, res: D2dResult, wait: int):
    """处理图像显示和键盘输入

    Args:
        image_in: 输入图像
        canvas: 用于显示的画布
        res: 检测结果
        wait: 等待时间(毫秒)

    Returns:
        tuple: (是否退出, 跳帧数)
    """
    resize(image_in, canvas)
    if len(res.objects) > 0:
        draw_d2d_objects(canvas, res.objects)
    cv2.imshow("win", canvas.data())
    key = cv2.waitKey(wait)

    skip_frames = 0
    if key == Key.ESC.value:
        return True, skip_frames  # 用户按ESC退出
    elif key == ord("w"):
        skip_frames = 200
        logger.info("跳过10帧")

    return False, skip_frames


def process_video(
    src_file: str, dst_dir: Path, analyzer, mask, canvas, roi, wait: int = -1
) -> bool:
    """处理单个视频文件，提取帧并生成元数据

    Returns:
        bool: 用户是否按ESC退出
    """
    cap = cv2.VideoCapture(src_file)

    out_dir = dst_dir / Path(src_file).name
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_frames = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if skip_frames > 0:
            skip_frames -= 1
            continue

        number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        image_ori = ImageNda(data=frame)
        assert image_ori.size() == SIZE_FHD
        image_det = ImageNda(size=SIZE_FHD, color=YOLO_GRAY)
        copy_mask(image_ori, mask, image_det)
        res = analyzer.detect(image_det)

        if number % 20 == 0:
            if wait >= 0:
                should_exit, frames_to_skip = display_and_input(
                    image_det, canvas, res, wait
                )
                if should_exit:
                    cap.release()
                    return True
                skip_frames = frames_to_skip

            n = number // 20
            name = f"{n:04d}"
            image_path = Path(out_dir, f"{name}.jpg")
            meta_path = image_path.with_suffix(".json")
            logger.info(f"#{number} {image_path}")
            image_ori.save(image_path)
            res.roi = roi
            save_json(res, meta_path)

    cap.release()
    return False  # 正常处理完毕


def track_videos(
    src_files: List[str], dst_dir: Path, d2d_cfg: D2dParams, wait: int = -1
) -> bool:
    model_dir = Path("/opt/howell/s4/current/ias/model/")

    img_size = 640
    conf_thr = 0.5
    iou_thr = 0.7

    src_files.sort()

    d2d_opt = D2dOpt(
        input_shape=(img_size, img_size), conf_thr=conf_thr, iou_thr=iou_thr, track=True
    )
    a2d_opt = A2dOpt(d2d=d2d_opt, d2d_name="sign.pt", props={0: "sign-valid.pt"})
    analyzer = A2dYolo(model_dir, a2d_opt)

    out_size = SIZE_HD
    canvas = ImageNda(out_size)
    mask = make_mask(SIZE_FHD, d2d_cfg.roi)

    for src_file in src_files:
        logger.info("Open file: {}", src_file)

        should_exit = process_video(
            src_file, dst_dir, analyzer, mask, canvas, d2d_cfg.roi, wait
        )
        if should_exit:
            return False
    return True


cv2.destroyAllWindows()


def main(wait: int):
    db_dir = Path("/opt/howell/s4/current/ias/domain/d1/n1/db")
    d2d_cfg_file = db_dir / "analyzer_d2d/111.json"
    dst_dir = Path("/var/howell/s4/ias/track")
    logger.info("db_dir: {}", db_dir)
    logger.info("dst_dir: {}", dst_dir)
    db = TaskDb(db_dir)
    # db.show()
    d2d_cfg = load_json(d2d_cfg_file, D2dParams).unwrap()

    while True:
        res = db.find_task()
        if res.is_some():
            task: TaskInfo = res.unwrap()
            logger.info("find task: {}", to_json(task))
            if track_videos(task.data_urls, dst_dir / str(task.id), d2d_cfg, wait=wait):
                db.task_done(task.id)
        else:
            logger.info("no task")

        sleep(5)


if __name__ == "__main__":
    # catch_show_err(main)
    # typer.run(main)
    main(-1)
