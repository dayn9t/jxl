import cv2

from ultralytics import YOLO

# !/opt/ias/env/bin/python


from jcx.sys.fs import find
from jvi.image.image_nda import ImageNda


def main() -> None:
    model_file = "/opt/howell/s4/current/ias/model/2025-03-05_sign.pt"
    # video_path = "/mnt/temp/C2_2025_03_05T10_09_47_L.mkv"
    video_path = "/mnt/temp/C2_2025_03_05T10_14_48_L.mkv"

    src_dir = "/home/jiang/py/jxl/assets/s4/snapshots"

    model = YOLO(model_file)

    files = find(src_dir, ".jpg")

    # Loop through the video frames
    for i, file in enumerate(files):
        # Read a frame from the video
        image_in = ImageNda.load(file)
        results = model.track(image_in.data(), persist=True)

        result = results[0]

        # Print bounding boxes, classes, and probabilities
        print("Bounding Boxes:", result.boxes.xyxy.tolist())
        print("Classes:", result.boxes.cls)
        print("Probabilities:", result.boxes.conf)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(800) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # catch_show_err(main)
    main()
