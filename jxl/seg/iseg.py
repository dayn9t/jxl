from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jiv.image.image_nda import ImageNda, ImageNdas
from jml.det.idetector import DetObjects
from jml.model.types import ModelInfo


@dataclass(frozen=True)
class SegOpt:
    """分割器选项"""
    input_shape: tuple
    """输入形状"""
    conf_thr: float
    """置信度阈值"""
    iou_thr: float
    """非极大值抑制中的重叠率阈值"""
    min_area: float = 0
    """目标最小面积"""
    verbose: bool = False
    """显示细节信息"""


class ISegRes(Protocol):
    """分割器返回结果"""

    def __len__(self):
        """包含对象数量"""
        return len(self.objects())

    def foreground(self) -> ImageNda:
        """获取前景Mask"""
        pass

    def objects(self) -> DetObjects:
        """获取结果中的对象集合"""
        pass


class ISeg(ABC):
    """分割器"""

    model_class = 'segment'

    @classmethod
    def new(cls, info: ModelInfo, model_root: Path):
        """创建检测器-根据信息"""
        file = model_root / info.file
        d = cls(file, info.opt, info.device)
        return d

    @abstractmethod
    def __init__(self, model_path: Path, opt: SegOpt, device_name: str):
        """创建分类器, 为 cls.new 提供模板"""
        self._model_path = model_path
        self._opt = opt
        self._device_name = device_name

    def __call__(self, images: ImageNdas) -> ISegRes:
        """分割图像"""
        return self.forward(images)

    @abstractmethod
    def forward(self, ims: ImageNdas) -> ISegRes:
        """分割图像"""
        pass
