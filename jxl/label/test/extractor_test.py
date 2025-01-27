from random import random

from jxl.iqa.diag_extractor import sharpness
from jxl.label.extractor import *
from pandas._testing import assert_frame_equal  # type: ignore


def test_num_columns() -> None:
    assert num_columns(4) == ['c0', 'c1', 'c2', 'c3']


def test_mat_to_df() -> None:
    df1 = mat_to_df([[1, 2]])
    df2 = DataFrame({'c0': [1], 'c1': [2]})
    assert_frame_equal(df1, df2)


def test_extractor() -> None:
    folder = Path('/home/jiang/ws/scene/diagnosis/clearness/dataset')

    e = Extractor(fun=sharpness, vec_size=256)
    m = e.extract_classes(folder / 'train', folder / 'train.csv')

    s = 0
    for cc in m:
        s += cc.count
        print(f'cls({cc.cls}), {cc.count})')
    print(f'total: {s}')


def not_test_del_random(ratio: float = 0.5) -> None:
    folder = 'NOT /home/jiang/ws/scene/diagnosis/chroma/_old/2023-02-05/0'

    files = files_in(folder, '.jpg')
    for f in files:
        if random() < ratio:
            f.unlink()
