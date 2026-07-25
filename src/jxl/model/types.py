from dataclasses import dataclass
from typing import TypeVar

OptT = TypeVar("OptT")


@dataclass(frozen=True, slots=True)
class ModelInfo[OptT]:
    """模型信息"""

    model_class: str
    """模型类"""
    file: str
    """所在文件"""
    opt: OptT
    """选项"""
    device: str = ""
    """设备"""
