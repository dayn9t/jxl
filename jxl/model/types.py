from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelInfo(object):
    """模型信息"""
    model_class: str
    """模型类"""
    file: str
    """所在文件"""
    opt: Any
    """选项"""
    device: str = ''
    """设备"""
