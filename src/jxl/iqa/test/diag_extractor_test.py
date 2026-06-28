from pathlib import Path

import pytest
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda

from jxl.iqa.diag_extractor import DIV_SIZE, DiagExtractor


# FIXME(pre-existing, jvi-migration): 依赖外部数据目录 /home/jiang/ws/...。待数据自包含后启用。
@pytest.mark.skip(reason="pre-existing: external data dir")
def test_extractor() -> None:
    folder = Path("/home/jiang/ws/scene/diagnosis/dates/2023-04-10/image")
    f1 = "n1_31010900901900301_2023-04-10_10-07-36.331.jpg"
    f2 = "n1_31011513700200301_2023-04-10_10-04-12.290.jpg"
    roi = Rect.new(0.0625, 0.03333333333333333, 0.875, 0.9333333333333333)
    extractor = DiagExtractor(roi, 5, 3, DIV_SIZE)

    im1 = ImageNda.load(folder / f1)
    im2 = ImageNda.load(folder / f2)

    extractor.extract(im2, im1)


    im1 = ImageNda.load(folder / f1)
    im2 = ImageNda.load(folder / f2)

    extractor.extract(im2, im1)


def test_extract_dir() -> None:
    Path("/home/jiang/ws/scene/diagnosis/clearness/dataset/train/0")

    # im1 = ImageNda.load(folder / f1)
    # im2 = ImageNda.load(folder / f2)

    # extractor.extract(im2, im1)
