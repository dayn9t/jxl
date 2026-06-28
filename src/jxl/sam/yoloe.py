from typing import TYPE_CHECKING, cast

from ultralytics.engine.results import Results
from ultralytics.models.yolo.model import YOLOE

from jxl.det.d2d import D2dResult
from jxl.det.yolo.adapter import boxes_to_d2d

if TYPE_CHECKING:
    from collections.abc import Callable


def main() -> None:
    # Initialize a YOLOE model
    model = YOLOE(
        "yoloe-11l-seg.pt"
    )  # or select yoloe-11s/m-seg.pt for different sizes

    # Set text prompt to detect person and bus. You only need to do this once after you load the model.
    names = ["bus", "bike", "person"]
    # ultralytics 动态方法无类型签名, 经 cast 显式标注以通过 no-untyped-call。
    get_text_pe = cast("Callable[[list[str]], object]", model.get_text_pe)
    set_classes = cast("Callable[[list[str], object], None]", model.set_classes)
    set_classes(names, get_text_pe(names))

    # Run detection on the given image
    rs = model.predict("/home/jiang/py/jxl/assets/person/p2.jpg")

    assert isinstance(rs, list)
    assert len(rs) == 1
    assert isinstance(rs[0], Results)
    boxes = rs[0].boxes
    assert boxes is not None
    objects = boxes_to_d2d(boxes)
    _r = D2dResult(objects=objects)
    # Show results
    show = cast("Callable[[], None]", rs[0].show)
    show()


if __name__ == "__main__":
    main()
