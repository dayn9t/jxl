from typing import Iterable

from ias.app.afs import afs
from jxl.det.idetector import DetOpt
from jxl.model.types import ModelInfo


class ModelPool:
    """模型池.

    深度学习模型是超大对象，需要在不同场景共享，避免重复。
    """

    def __init__(self) -> None:
        self.models = {}
        self.classes = {}

    def register(self, c) -> None:
        """注册模型类"""
        self.classes[c.model_class] = c

    def register_all(self, classes: Iterable) -> None:
        """注册模型类列表"""
        for c in classes:
            self.register(c)

    def get(self, info: ModelInfo):
        """根据信息获取模型"""

        if info in self.models:
            return self.models[info]

        if info.model_class in self.classes:
            m = self.classes[info.model_class].new(info, afs().model_dir)
            self.models[info] = m
            return m
        return None

    def __len__(self) -> int:
        """获取模型数量"""
        return len(self.models)


def a_test() -> None:
    opt1 = DetOpt((640, 640), 0.5, 0.5)
    opt2 = DetOpt((640, 640), 0.6, 0.5)

    folder = '/opt/ias/model/work/disaster/fire_smoke/'
    file = folder + 'weights.pt'
    info1 = ModelInfo('y5_detector', file, opt1)
    info2 = ModelInfo('y5_detector', file, opt2)

    pool = ModelPool()

    _model1 = pool.get(info1)
    print(len(pool))
    _model2 = pool.get(info1)
    print(len(pool))
    _model2 = pool.get(info2)
    print(len(pool))


if __name__ == '__main__':
    a_test()
