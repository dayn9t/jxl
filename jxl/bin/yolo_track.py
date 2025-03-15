from ultralytics import YOLO

def main():

    model_file = "/opt/howell/s4/current/ias/model/2025-03-05_sign.pt"
    #video_file = "/mnt/temp/C2_2025_03_05T10_09_47_L.mkv"
    video_file = "/mnt/temp/C2_2025_03_05T10_14_48_L.mkv"

    #model = YOLO("yolo11n.pt")  # Load an official Detect model
    #model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
    #model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
    model = YOLO(model_file)  # Load a custom trained model

    # Perform tracking with the model
    #results = model.track(video_file, show=True)  # Tracking with default tracker
    results = model.track(video_file, show=True, tracker="bytetrack.yaml")  # with ByteTrack


if __name__ == "__main__":
    # catch_show_err(main)
    main()