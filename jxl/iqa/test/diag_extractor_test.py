from pathlib import Path

from jxl.iqa.diag_extractor import *


def test_extractor() -> None:
    folder = Path('/home/jiang/ws/scene/diagnosis/dates/2023-04-10/image')
    f1 = 'n1_31010900901900301_2023-04-10_10-07-36.331.jpg'
    f2 = 'n1_31011513700200301_2023-04-10_10-04-12.290.jpg'
    roi = Rect(0.0625, 0.03333333333333333, 0.875, 0.9333333333333333)
    extractor = DiagExtractor(roi, 5, 3)

    im1 = ImageNda.load(folder / f1)
    im2 = ImageNda.load(folder / f2)

    extractor.extract(im2, im1)


def test_extract_dir() -> None:
    folder = Path('/home/jiang/ws/scene/diagnosis/clearness/dataset/train/0')

    # im1 = ImageNda.load(folder / f1)
    # im2 = ImageNda.load(folder / f2)

    # extractor.extract(im2, im1)
