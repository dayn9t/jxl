import cv2
import typer

from ultralytics import YOLO


def main(src_file: str, dst_dir: str):
    model_file = "/opt/howell/s4/current/ias/model/2025-03-05_sign.pt"
    # video_path = "/mnt/temp/C2_2025_03_05T10_09_47_L.mkv"
    # video_path = "/mnt/temp/C2_2025_03_05T10_14_48_L.mkv"

    model = YOLO(model_file, task="detect")

    cap = cv2.VideoCapture(src_file)

    n = 0
    # Loop through the video frames
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        n += 1
        #if n % 5 != 0:
        #    continue
        #timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        #print(f"#{n} Timestamp: {timestamp} ms")
        number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if number % 4 != 0:
            continue

        print(f"#{number} frame_number: {number} ms")
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        result = results[0]

        # Print bounding boxes, classes, and probabilities
        #print("Bounding Boxes:", result.boxes.xyxy.tolist())
        #print("Classes:", result.boxes.cls)
        #print("Probabilities:", result.boxes.conf)

        annotated_frame = results[0].plot()
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
        if number % 20 == 0:
            print("save frame", number)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # catch_show_err(main)
    typer.run(main)
