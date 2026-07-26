from ultralytics import YOLOE
from ultralytics.engine.results import Results

from jxl.det.d2d import D2dResult
from jxl.det.yolo.adapter import boxes_to_d2d


def main() -> None:
    # Initialize a YOLOE model
    model = YOLOE(
        "yoloe-11l-seg.pt"
    )  # or select yoloe-11s/m-seg.pt for different sizes

    # Set text prompt to detect person and bus. You only need to do this once after you load the model.
    names = ["bus", "bike", "person"]
    model.set_classes(names, model.get_text_pe(names))

    # Run detection on the given image
    rs = model.predict("/home/jiang/cc/py/jxl/assets/person/p2.jpg")

    assert isinstance(rs, list)
    assert len(rs) == 1
    assert isinstance(rs[0], Results)
    # print('YOLO result:', type(rs[0]))
    if rs[0].boxes is None:
        raise ValueError("rs[0].boxes is None")
    objects = boxes_to_d2d(rs[0].boxes)
    D2dResult(objects=objects)  # demo: 检测框转 D2dResult
    # Show results
    rs[0].show()


if __name__ == "__main__":
    main()
