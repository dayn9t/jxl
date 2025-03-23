from pathlib import Path

import cv2  # type: ignore
# params:/home/jiang/ws/trash/c
import cv2
import typer
from jcx.text.txt_json import save_json
from jcx.ui.key import Key
from jvi.geo.size2d import SIZE_HD
from jvi.image.image_nda import ImageNda
from jvi.image.proc import resize

from jxl.det.d2d import D2dOpt
from jxl.det.d2d import draw_d2d_objects
from jxl.det.yolo.d2d_yolo import D2dYolo


# video_path = "/mnt/temp/C2_2025_03_05T10_09_47_L.mkv"
# video_path = "/mnt/temp/C2_2025_03_05T10_14_48_L.mkv"

def main(src_file_list: str, dst_dir: str, wait: int = -1):
    model_file = Path("/opt/howell/s4/current/ias/model/2025-03-05_sign.pt")

    src_files = src_file_list.split(',')
    src_files.sort()

    img_size = 640
    conf_thr = 0.5
    iou_thr = 0.7
    det_opt = D2dOpt(input_shape=(img_size, img_size), conf_thr=conf_thr, iou_thr=iou_thr, track=True)
    detector = D2dYolo(model_file, det_opt)

    out_size = SIZE_HD
    canvas = ImageNda(out_size)

    for src_file in src_files:
        print("Open file:", src_file)
        cap = cv2.VideoCapture(src_file)
        out_dir = Path(dst_dir, Path(src_file).name)
        # Loop through the video frames
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            # timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            #if number % 4 != 0:
            #    continue
            image_in = ImageNda(data=frame)
            print(f"#{number} size: {image_in.size()}")

            res = detector.detect(image_in)

            resize(image_in, canvas)
            if len(res.objects) > 0:
                draw_d2d_objects(canvas, res.objects)
            else:
                print("Invalid det")

            if wait >= 0:
                cv2.imshow("win", canvas.data())
                if cv2.waitKey(wait) == Key.ESC.value:
                    break
            if number % 20 == 0:
                n = number // 20
                name = f"{n:04d}"
                image_path = Path(out_dir, f"{name}.jpg")
                meta_path = image_path.with_suffix(".json")
                print(f"#{number} {image_path}")
                image_in.save(image_path)
                save_json(res, meta_path)

        cap.release()
    cv2.destroyAllWindows()


def main_demo():
    files3 = [
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_42_50_L.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_17_45_L.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_12_44_R.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_37_49_L.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_47_55_R.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_57_53_L.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_12_44_L.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_42_54_R.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_17_45_R.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_52_55_R.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_52_52_L.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_32_52_R.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_57_56_R.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_22_46_L.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_27_47_L.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_47_51_L.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_22_46_R.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_27_46_R.mkv",
    "file:///mnt/temp/2025_03_21/C2_2025_03_21T10_32_48_L.mkv",
    "file:///mnt/temp/2025_03_21/C3_2025_03_21T10_37_53_R.mkv"
    ]

    files_str = ','.join(files3)
    dst_dir3 = "/home/jiang/1/s4/6"

    # file = "file:///mnt/temp/2025_03_17/C2_2025_03_14T10_13_58_L.mkv"
    main(files_str, dst_dir3)


if __name__ == "__main__":
    # catch_show_err(main)
    # typer.run(main)
    main_demo()
