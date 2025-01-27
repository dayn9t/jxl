#!/opt/ias/env/bin/python

from pathlib import Path

import fire  # type: ignore
from autogluon.tabular import TabularDataset, TabularPredictor  # type: ignore


# label = 'signature'

def train(data_dir: str, save_dir: str, label: str = 'label') -> None:
    p = Path(data_dir)
    train_file = p / 'train.csv'
    val_file = p / 'val.csv'
    test_dir = p / 'test.csv'

    train_data = TabularDataset(str(train_file))
    # train_data.head()
    r = train_data[label].describe()
    print(f'describe of {label}:')
    print(r)

    val_data = TabularDataset(str(val_file)) if val_file.is_file() else None

    print('\ntraining...\n')
    predictor = TabularPredictor(label=label, path=save_dir, verbosity=2) \
        .fit(train_data, val_data)

    # 最准确的集成预测器提取为更简单/更快且需要更少内存/计算的单个模型
    # predictor.distill()

    # predictor.print_interpretable_rules() # FIXME: 会报错
    # 依然保留了其他3个, 可能被依赖?
    predictor.delete_models(models_to_keep='best', dry_run=False)
    predictor.compile_models()

    test_data = TabularDataset(str(test_dir))

    print('\npredict...\n')
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    r = y_pred.head()
    print(r)

    print('\nevaluate...\n')
    r = predictor.evaluate(test_data, silent=True)
    print(r)

    print('\nleaderboard...\n')
    r = predictor.leaderboard(test_data, silent=True)
    print(r)

    # print('\nexplain...\n')
    # 通过将基于规则的模型拟合到分类错误来解释分类错误
    # predictor.explain_classification_errors(test_data) # FIXME: ModuleNotFoundError: No module named 'imodels'


if __name__ == '__main__':
    fire.Fire(train)
