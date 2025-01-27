from pathlib import Path

from jxl.label.darknet import DarknetSet
from jxl.label.hop import HopSet
from jxl.label.label_set import LabelSet, LabelFormat
from rustshed import Result, Err, Ok

_class_map = {
    LabelFormat.HOP: HopSet,
    LabelFormat.DARKNET: DarknetSet,
    LabelFormat.IMAGENET: HopSet,
    LabelFormat.KITTI: HopSet,
}


def open_label_set(
    folder: Path, label_format: LabelFormat, meta_id: int, pattern: str
) -> Result[LabelSet, str]:
    """打开标注数据集合"""

    fmt_cls = guess_format_cls(folder, label_format, meta_id)

    if fmt_cls is None:
        return Err(f'Format "{label_format}" not found')

    if not fmt_cls.valid_set(folder, meta_id=meta_id):
        return Err(f'Invalid folder for format "{label_format}"')

    return Ok(fmt_cls(folder, meta_id, pattern))


def guess_format_cls(folder: Path, label_format: LabelFormat, meta_id: int):
    """猜测数据集格式"""
    fmt_cls = None
    if label_format:
        fmt_cls = _class_map[label_format]
    else:
        for name, cls in _class_map.items():
            if cls.valid_set(folder, meta_id):  # type: ignore
                fmt_cls = cls
    return fmt_cls
