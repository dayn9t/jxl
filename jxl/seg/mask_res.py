from jvi.image.image_nda import ImageNda
from jvi.image.struct import find_polygons
from jxl.common import ProbValue
from jxl.det.idetector import DetObjects, DetObject
from jxl.seg.iseg import ISegRes


class MaskRes(ISegRes):
    """Mask分割器返回结果"""

    def __init__(self, mask: ImageNda, min_area: float):
        self._mask = mask
        self._min_area = min_area

    def foreground(self) -> ImageNda:
        """获取前景Mask"""
        return self._mask

    def objects(self) -> DetObjects:
        """获取结果中的对象集合"""
        boxes = find_polygons(self._mask, self._min_area)
        obs = [DetObject(0, ProbValue(0, 0.5), polygon=box.polygon) for box in boxes]
        return obs
