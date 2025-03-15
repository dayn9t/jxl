from jvi.geo.rectangle import Rect

from jvi.geo.rectangle import Rect
from ultralytics.engine.results import Boxes

from jxl.det.d2d import D2dObject, D2dObjects


def boxes_to_d2d(boxes: Boxes) -> D2dObjects:
    """boxes => objects"""
    xyxyn_arr = boxes.xyxyn.tolist()
    conf_arr = boxes.conf.tolist()
    cls_arr = boxes.cls.int().tolist()
    id_arr = boxes.id.int().tolist() if boxes.id else [0] * len(xyxyn_arr)

    objects = []
    for i in range(len(xyxyn_arr)):
        rect = Rect.from_ltrb_list(xyxyn_arr[i])
        conf = conf_arr[i]
        cls = cls_arr[i]
        id_ = id_arr[i]
        objects.append(D2dObject(id=id_, cls=cls, conf=conf, rect=rect))
    return objects
