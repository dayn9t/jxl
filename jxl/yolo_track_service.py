from pathlib import Path
from time import sleep
from typing import List, Final

import cv2  # type: ignore
import cv2
from jcx.text.txt_json import save_json, to_json, load_json
from jcx.ui.key import Key
from jvi.drawing.color import YOLO_GRAY
from jvi.geo.point2d import Points
from jvi.geo.size2d import SIZE_HD, SIZE_FHD
from jvi.image.image_nda import ImageNda
from jvi.image.proc import resize
from jvi.image.util import make_mask, copy_mask
from loguru import logger

from jxl.det.a2d import A2dOpt, A2dResult
from jxl.det.d2d import D2dOpt
from jxl.det.d2d import draw_d2d_objects
from jxl.det.yolo.a2d_yolo import A2dYolo
from js4.task import TaskDb, D2dParams


def display_and_input(image_in: ImageNda, canvas: ImageNda, res: A2dResult, wait: int):
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
    src_file: str,
    dst_dir: Path,
    analyzer: A2dYolo,
    mask: ImageNda,
    canvas: ImageNda,
    roi: Points,
    fps: float,
    wait: int = -1,
) -> bool:
    """处理单个视频文件，提取帧并生成元数据

    Args:
        src_file: 视频文件路径
        dst_dir: 输出目录
        analyzer: 视频分析器对象
        mask: 用于处理的图像蒙版
        canvas: 用于显示的画布
        roi: 感兴趣区域坐标列表
        wait: 帧间等待时间(毫秒)，-1表示不等待

    Returns:
        bool: 用户是否按ESC退出
    """
    cap = cv2.VideoCapture(src_file)

    out_dir = dst_dir / Path(src_file).name
    out_dir.mkdir(parents=True, exist_ok=True)

    interval = 1000 / fps
    number = 1
    skip_frames = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if skip_frames > 0:
            skip_frames -= 1
            continue

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)  # 获取当前帧的时间戳，单位为毫秒
        image_ori = ImageNda(data=frame)
        assert image_ori.size() == SIZE_FHD
        image_det = ImageNda(size=SIZE_FHD, color=YOLO_GRAY)
        copy_mask(image_ori, mask, image_det)
        res = analyzer.detect(image_det)

        if pos_msec >= number * interval:
            name = f"{number:04d}"
            image_path = Path(out_dir, f"{name}.jpg")
            meta_path = image_path.with_suffix(".json")
            logger.info(f"#{number} {image_path}")
            image_ori.save(image_path)
            res.roi = roi
            save_json(res, meta_path)
            number += 1
            if wait >= 0:
                should_exit, frames_to_skip = display_and_input(
                    image_det, canvas, res, wait
                )
                if should_exit:
                    cap.release()
                    return True
                skip_frames = frames_to_skip
    cap.release()
    return False  # 正常处理完毕


def track_videos(
    src_files: List[str],
    dst_dir: Path,
    d2d_cfg: D2dParams,
    model_dir: Path,
    fps: float,
    wait: int = -1,
) -> bool:

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
            src_file, dst_dir, analyzer, mask, canvas, d2d_cfg.roi, fps, wait
        )
        if should_exit:
            return False
    return True


def main(wait: int):
    db_dir = Path("/opt/howell/s4/current/ias/domain/d1/n1/db")
    d2d_cfg_file = db_dir / "analyzer_d2d/111.json"
    dst_dir = Path("/home/jiang/py/jxl/tmp")
    fps = 1.25  # 每秒1.25帧

    logger.info("db_dir: {}", db_dir)
    logger.info("dst_dir: {}", dst_dir)
    d2d_cfg = load_json(d2d_cfg_file, D2dParams).unwrap()

    files = ["file:///mnt/temp/2025_03_31/C2_2025_03_24T10_24_32_L.mkv"]

    track_videos(files, dst_dir, d2d_cfg, fps, wait=wait)
    cv2.destroyAllWindows()
    # "ffmpeg -i /mnt/temp/2025_03_31/C2_2025_03_21T10_12_44_L.mkv -vf fps=1.25 -q:v 2 /tmp/.tmprJ2LvU/%04d.jpg"


if __name__ == "__main__":
    # catch_show_err(main)
    # typer.run(main)
    main(-1)
