from pathlib import Path

import pytest
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda

from jxl.iqa.diag_extractor import DIV_COLS, DIV_ROWS, DIV_SIZE, DiagExtractor

# 依赖本地图像数据集, 数据缺失时跳过.
_DATA_DIR = Path("/home/jiang/ws/scene/diagnosis")
pytestmark = pytest.mark.skipif(
    not _DATA_DIR.is_dir(),
    reason=f"local image dataset not available: {_DATA_DIR}",
)


def test_extractor() -> None:
    folder = Path("/home/jiang/ws/scene/diagnosis/dates/2023-04-10/image")
    f1 = "n1_31010900901900301_2023-04-10_10-07-36.331.jpg"
    f2 = "n1_31011513700200301_2023-04-10_10-04-12.290.jpg"
    roi = Rect.new(0.0625, 0.03333333333333333, 0.875, 0.9333333333333333)
    extractor = DiagExtractor(roi, DIV_COLS, DIV_ROWS, DIV_SIZE)

    im1 = ImageNda.load(folder / f1)
    im2 = ImageNda.load(folder / f2)

    extractor.extract(im2, im1)

