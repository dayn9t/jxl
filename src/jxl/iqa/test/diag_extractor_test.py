from pathlib import Path

import pytest

from jxl.iqa.diag_extractor import *


# FIXME(pre-existing, jvi-migration): DiagExtractor 签名现为 (roi, cols, rows, tile_size),
# 此处少传 tile_size; 且依赖外部数据目录 /home/jiang/ws/...。待 PR-3 修复。
@pytest.mark.skip(reason="pre-existing: stale DiagExtractor signature + external data")
def test_extractor() -> None:
    folder = Path("/home/jiang/ws/scene/diagnosis/dates/2023-04-10/image")
    f1 = "n1_31010900901900301_2023-04-10_10-07-36.331.jpg"
    f2 = "n1_31011513700200301_2023-04-10_10-04-12.290.jpg"
    roi = Rect.new(0.0625, 0.03333333333333333, 0.875, 0.9333333333333333)
    extractor = DiagExtractor(roi, 5, 3)


    im1 = ImageNda.load(folder / f1)
    im2 = ImageNda.load(folder / f2)

    extractor.extract(im2, im1)


def test_extract_dir() -> None:
    folder = Path("/home/jiang/ws/scene/diagnosis/clearness/dataset/train/0")

    # im1 = ImageNda.load(folder / f1)
    # im2 = ImageNda.load(folder / f2)

    # extractor.extract(im2, im1)
