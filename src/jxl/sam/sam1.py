from ultralytics.models.nas.model import NAS


def main() -> None:
    # Load a COCO-pretrained YOLO-NAS-s model
    model = NAS("yolo_nas_s.pt")

    # Validate the model on the COCO8 example dataset
    _results = model.val(data="coco8.yaml")

    # Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
    _results = model("path/to/bus.jpg")


if __name__ == "__main__":
    main()
