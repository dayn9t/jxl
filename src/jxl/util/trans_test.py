import numpy as np
from jvi.geo.size2d import SIZE_HD
from jvi.image.image_nda import ImageNda
from jvi.image.trans import bgr_to_pil, pil_to_bgr
from PIL.Image import Image as PilImage


def test_bgr_to_pil() -> None:
    im1 = ImageNda(SIZE_HD)
    im2 = bgr_to_pil(im1.data())
    assert isinstance(im2, PilImage)
    assert im2.size == SIZE_HD.to_tuple_int()

    im3 = pil_to_bgr(im2)
    assert isinstance(im3, np.ndarray)

    assert ImageNda(data=im3) == im1
