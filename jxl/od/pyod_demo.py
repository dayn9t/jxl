from pyod.models.base import BaseDetector
from pyod.models.iforest import IForest
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.xgbod import XGBOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize


def show_od(detector: BaseDetector, name: str, contamination: float = 0.1, n_train: int = 60, n_test: int = 30):
    """
    contamination: 异常值占比
    n_train: 训练集大小
    n_test: 测试集大小
    """
    # 生产样本数据
    x_train, x_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42,
                      behaviour="new")

    # print('x_train:\n', x_train)
    # print('y_train:\n', y_train)

    # print('x_test:\n', x_test)
    # print('y_test:\n', y_test)

    detector.fit(x_train)

    # 获得训练集的预测标签和异常分数
    y_train_pred = detector.labels_  # (0: 正常, 1: 异常)
    y_train_scores = detector.decision_scores_

    print('y_train_pred:\n', y_train_pred)
    print('y_train_scores:\n', y_train_scores)

    # 获得测试集的预测结果
    y_test_pred = detector.predict(x_test)
    y_test_scores = detector.decision_function(x_test)  # 返回未知数据上的异常值 (分值越大越异常)

    print('y_test_pred:\n', y_test_pred)
    print('y_test_scores:\n', y_test_scores)

    # 评估并且打印结果
    print("\nOn Training Data:")
    evaluate_print(name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(name, y_test, y_test_scores)

    # 注意，为了实现可视化，原始维度必须是 2 维
    visualize(name, x_train, y_train, x_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)


if __name__ == "__main__":

    clf_name = 'IForest'
    clf = None

    if clf_name == 'PCA':
        clf = PCA(n_components=2)
    elif clf_name == 'MCD':
        clf = MCD()
    elif clf_name == 'IForest':
        clf = IForest()
    elif clf_name == 'XGBOD':
        clf = XGBOD()

    show_od(clf, clf_name)
