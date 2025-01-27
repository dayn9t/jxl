from pathlib import Path

from ias.sensor_type import SensorMsg
from jcx.text.txt_json import save_json
from jvi.geo import Size
from jxl.det.detector import DetectedObject

_class_tab = ["dry_can", "dry_can_lid", "refuse_dump", "wet_can", "wet_can_lid"]


def _convert_object(o: DetectedObject, size: Size):
    """转换目标"""
    r = o.rect.absolutize(size)
    points = [[p.x, p.y] for p in r.vertexes()]

    m = {
        "label": _class_tab[o.class_index],
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
    }
    return m


def save_labelme_json(msg: SensorMsg, size: Size) -> Path:
    """传感器消息保存为JSON"""
    file = Path(msg.image)
    objects = [_convert_object(o, size) for o in msg.objects]
    m = {
        "version": "4.5.6",
        "flags": {},
        "shapes": objects,
        "imagePath": file.name,
        "imageData": None,
        "imageHeight": size.height,
        "imageWidth": size.width,
    }
    json_file = file.with_suffix(".json")
    save_json(m, json_file)
    return json_file
