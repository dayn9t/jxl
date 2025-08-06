from ultralytics import NAS


def main():
    # Load a COCO-pretrained YOLO-NAS-s model
    model = NAS("yolo_nas_s.pt")

    # Validate the model on the COCO8 example dataset
    results = model.val(data="coco8.yaml")

    # Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
    results = model("path/to/bus.jpg")


if __name__ == "__main__":
    main()
