from pathlib import Path

from autogluon.tabular import TabularPredictor
from pandas import DataFrame

from jxl.cls.classifier import (
    ClassifierOpt,
    ClassifierRes,
    ClassifierResList,
    IClassifier,
)
from jxl.label.extractor import mat_to_df


class ClassifierTab(IClassifier[list[float]]):
    """表格数据分类器"""

    model_class = "tabular"

    def __init__(self, model_path: Path, opt: ClassifierOpt, device_name: str = "") -> None:
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
