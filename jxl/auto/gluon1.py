from autogluon.tabular import TabularDataset, TabularPredictor  # type: ignore


def main(train_new: bool) -> None:
    # data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
    data_url = './tab_data/'
    path = data_url + 'models'

    # TabularDataset 本质上是一个 pandas DataFrame
    train_data = TabularDataset(f'{data_url}train.csv')
    train_data.head()

    label = 'signature'
    r = train_data[label].describe()
    print(f'describe of {label}:')
    print(r)

    if train_new:
        print('\ntraining...\n')
        predictor = TabularPredictor(label=label, path=path, verbosity=2).fit(train_data)
        # 最准确的集成预测器提取为更简单/更快且需要更少内存/计算的单个模型
        predictor.distill()

    else:
        print('\nloading...\n')
        predictor = TabularPredictor.load("./tab_data/models/")
        predictor.print_interpretable_rules()
        # 依然保留了其他3个, 可能被依赖?
        predictor.delete_models(models_to_keep='best', dry_run=False)
        predictor.compile_models()

    test_data = TabularDataset(f'{data_url}test.csv')

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

    print('\nexplain...\n')
    # 通过将基于规则的模型拟合到分类错误来解释分类错误
    predictor.explain_classification_errors(test_data)


if __name__ == '__main__':
    main(True)
