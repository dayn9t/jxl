from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, TypeVar, Type, Protocol

import numpy as np
from jml.label.info import ProbValue
from jml.model.types import ModelInfo
from pydantic import BaseModel


# 参考: [torch模型存取](https://blog.csdn.net/dialogueeeee/article/details/122714704)
class ModelFormat(IntEnum):
    """模型格式"""

    FULL_MODEL = 0
    """完整模型，模型+参数"""
    PARAM_TAR = 1
    """参数包, 无模型结构"""


@dataclass(frozen=True)
class ClassifierOpt:
    """分类器器选项"""

    input_shape: tuple
    """输入形状, 1/2/3维"""
    num_classes: int
    """类别数量"""
    normalized: bool = True
    """数据归一化，ImageNet标准归一化"""
    out_layer: bool = True
    """输出层是否增加 softmax，TODO"""
    data_format: ModelFormat = ModelFormat.FULL_MODEL
    """模型数据格式"""
    verbose: bool = False
    """显示细节信息"""


class ClassifierRes(Protocol):
    """分类器返回结果"""

    def top(self) -> ProbValue:
        """最可能类别"""
        pass

    def top_index(self) -> int:
        """最可能类别索引"""
        return self.top().value

    def top_confidence(self) -> float:
        """最可能类别置信度"""
        return self.top().confidence

    def confidences(self) -> list[float]:
        """获取各个分类的置信度"""
        pass

    def at(self, idx: int) -> float:
        """获取指定分类的置信度"""
        return self.confidences()[idx]

    def __len__(self) -> int:
        """分类器结果包含对象数量"""
        pass

    def bin(self) -> 'ClassifierRes':
        """二值化, 0/非0各一组"""
        probs = self.confidences()
        p0 = probs[0]
        p1 = sum(probs) - p0

        return ClassifierResList(probs=[p0, p1])

    def nozero_ratio(self) -> float:
        """非零分类所占比率"""
        probs = self.confidences()
        return 1 - probs[0]


def vote_weighted(arr: list[ClassifierRes]) -> ClassifierRes:
    """根据概率投票获取最终分类结果"""
    mat = np.array([a.confidences() for a in arr])
    v = np.sum(mat, axis=0)
    v /= np.sum(v)
    return ClassifierResList(probs=v.tolist())


def vote_bin(arr: list[ClassifierRes]) -> ClassifierRes:
    """投票进行二分类"""
    zero_votes = 0
    for r in arr:
        if r.top_index() == 0:
            zero_votes += 1
    p0 = zero_votes / len(arr)
    p1 = 1 - p0
    return ClassifierResList(probs=[p0, p1])


@dataclass
class ClassifierResList(ClassifierRes):
    """分类器返回结果"""

    probs: list[float]

    def top(self) -> ProbValue:
        """最可能类别"""
        i = np.argmax(self.probs)
        return ProbValue(int(i), self.probs[i])

    def top_index(self) -> int:
        """最可能类别索引"""
        return self.top().value

    def top_confidence(self) -> float:
        """最可能类别置信度"""
        return self.top().confidence

    def confidences(self) -> list[float]:
        """获取各个分类的置信度"""
        return self.probs

    def __len__(self) -> int:
        """分类器结果包含对象数量"""
        return len(self.probs)


Self = TypeVar("Self", bound='IClassifier')


class IClassifier(ABC):
    """分类器"""

    model_class = 'classifier'

    @classmethod
    def new(cls: Type[Self], info: ModelInfo, model_root: Path) -> Self:
        """创建分类器-根据信息"""
        file = model_root / info.file
        d = cls(file, info.opt, info.device)
        return d

    @abstractmethod
    def __init__(self, model_path: Path, opt: ClassifierOpt, device_name: str):
        """创建分类器"""
        self._model_path = model_path
        self._opt = opt
        self._device_name = device_name

    @abstractmethod
    def __call__(self, item: Any) -> ClassifierRes:
        """对数据条目分类, 支持类型:图像, 数组"""
        pass
