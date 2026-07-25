from jvi.geo.rectangle import Rect
from ultralytics.engine.results import Boxes

from jxl.det.d2d import D2dObject, D2dObjects


def boxes_to_d2d(boxes: Boxes) -> D2dObjects:
    """boxes => objects"""
    xyxyn_arr = boxes.xyxyn.tolist()
    conf_arr = boxes.conf.tolist()
    # TODO(dayn9t): ultralytics stubs 把 boxes.cls/id 标注为 Tensor | ndarray，
    # 运行时恒为 Tensor；库 stub 修正前此处显式忽略 .int() 的 union-attr。
    cls_arr = boxes.cls.int().tolist()  # type: ignore[union-attr]
    id_arr = (
        boxes.id.int().tolist()  # type: ignore[union-attr]
        if boxes.id is not None
        else [0] * len(xyxyn_arr)
    )

    objects = []
    for i in range(len(xyxyn_arr)):
        rect = Rect.from_ltrb_list(xyxyn_arr[i])
        conf = conf_arr[i]
        cls = cls_arr[i]
        id_ = id_arr[i]
        objects.append(D2dObject(id=id_, cls=cls, conf=conf, rect=rect))
    return objects
