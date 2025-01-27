#!/opt/ias/env/bin/python
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from jcx.ui.key import Key
from jvi.geo.size2d import Size
from jvi.image.image_nda import ImageNda
from jvi.image.trace import trace_images, close_all_windows
from joblib import dump
from pandas import DataFrame
from pyod.models.iforest import IForest
from pyod.utils.data import evaluate_print


def train(data_file: str, data_dir: str, n_estimators: int = 500, contamination: float = 0.002,
          save_file: str = 'od.joblib',
          show: bool = False) -> None:
    origin_data = pd.read_csv(data_file)
    train_data = origin_data.drop(columns=['a', 'b']).to_numpy()

    detector = IForest(n_estimators=n_estimators, contamination=contamination)
    name = 'IForest'

    print('开始训练模型 ...')
    detector.fit(train_data)

    dump(detector, save_file)
    print('模型保存为:', save_file)

    # 获得训练集的预测标签和异常分数
    res = detector.labels_  # (0: 正常, 1: 异常)
    res_scores = detector.decision_scores_

    print('y:', res)
    print('y_scores:', res_scores)

    # 评估并且打印结果
    print("\nOn Training Data:")
    evaluate_print(name, res, res_scores)

    assert len(origin_data) == len(res)

    err = sum(res)
    total = len(res)
    radio = round(err / total, 4)
    print(f'错误率: {err}/{total}({radio * 100}%)')

    if show:
        show_error(data_dir, origin_data, res)


def show_error(data_dir: str, origin_data: DataFrame, res: np.ndarray) -> None:
    dir_ = Path(data_dir)
    j = 0
    for i, row in origin_data.iterrows():
        if res[i]:
            j += 1
            print(f'#{j}', row['a'], row['b'])
            im_a = ImageNda.load(dir_ / row['a'])
            im_b = ImageNda.load(dir_ / row['b'])
            key, image = trace_images([im_a, im_b], 'Unmatched Image', box_size=Size(1920, 1080), auto_close=False)
            match key:
                case Key.ESC:
                    break
                case Key.F2:
                    file = f'od-snapshot{j}.jpg'
                    print(f'保存快照: {file}')
                    image.save(file)
    close_all_windows()


if __name__ == '__main__':
    fire.Fire(train)
