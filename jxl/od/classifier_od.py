from pathlib import Path

import numpy as np
from jxl.cls.classifier import ClassifierRes, IClassifier, ClassifierOpt, ClassifierResList
from joblib import load


class ClassifierOd(IClassifier):
    """异常检测分类器"""

    model_class = 'outlier'

    def __init__(self, model_path: Path, opt: ClassifierOpt, device_name: str = ''):
        super().__init__(model_path, opt, device_name)

        self._model = load(str(model_path))

    def __str__(self) -> str:
        s = self._model.__str__()
        assert isinstance(s, str)
        return s

    def __call__(self, vec: list[float]) -> ClassifierRes:
        """分类输入向量"""
        assert self._opt.input_shape == (len(vec),)
        arr = np.array([vec])
        res = self._model.predict_proba(arr)
        assert isinstance(res, np.ndarray) and len(res) == 1
        assert len(res[0]) == 2

        return ClassifierResList(probs=res[0].tolist())
