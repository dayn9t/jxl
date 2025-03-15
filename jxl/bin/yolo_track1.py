import cv2

from ultralytics import YOLO


def main():

    model_file = "/opt/howell/s4/current/ias/model/2025-03-05_sign.pt"
    #video_path = "/mnt/temp/C2_2025_03_05T10_09_47_L.mkv"
    video_path = "/mnt/temp/C2_2025_03_05T10_14_48_L.mkv"

    model = YOLO(model_file)

    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # catch_show_err(main)
    main()
