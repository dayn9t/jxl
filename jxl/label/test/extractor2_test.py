from random import random

from jvi.geo.rectangle import Rect
from jvi.geo.size2d import SIZE_QVGA
from jvi.match.match import ImageMatcher
from jxl.iqa.diag_extractor import sharpness, DiagExtractor
from jxl.label.extractor import *
from jxl.label.extractor2 import Extractor2
from pandas._testing import assert_frame_equal  # type: ignore

COLS = 4
ROWS = 3


def test_extractor2() -> None:
    folder = Path('/home/jiang/ws/trash/outside/2022/n1/n')

    e = Extractor2(fun=extract_matches, vec_size=COLS * ROWS)
    m = e.extract_cameras(folder, folder / 'train.csv')

    s = 0
    for cc in m:
        s += cc.count
        print(f'cls({cc.cls}), {cc.count})')
    print(f'total: {s}')
