from pathlib import Path

from autogluon.tabular import TabularPredictor
from jxl.cls.classifier import ClassifierRes, IClassifier, ClassifierOpt, ClassifierResList
from jxl.label.extractor import mat_to_df
from pandas import DataFrame


class ClassifierTab(IClassifier):
    """表格数据分类器"""

    model_class = 'tabular'

    def __init__(self, model_path: Path, opt: ClassifierOpt, device_name: str = ''):
        super().__init__(model_path, opt, device_name)

        self._model = TabularPredictor.load(str(model_path))
        self._model.compile_models()

    def __str__(self) -> str:
        s = self._model.__str__()
        assert isinstance(s, str)
        return s

    def __call__(self, vec: list[float]) -> ClassifierRes:
        """分类输入向量"""
        df = mat_to_df([vec])
        r = self._model.predict_proba(df)
        assert isinstance(r, DataFrame)
        return ClassifierResList(probs=r.loc[0].tolist())
