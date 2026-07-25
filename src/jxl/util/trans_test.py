import numpy as np
import torchvision.transforms as transforms
from jvi.geo.size2d import SIZE_HD
from jvi.image.image_nda import ImageNda
from jvi.image.trans import PilImage, bgr_to_pil, fromarray, pil_to_bgr


def test_bgr_to_pil() -> None:
    im1 = ImageNda(SIZE_HD)
    im2 = bgr_to_pil(im1.data())
    assert isinstance(im2, PilImage)
    assert im2.size == SIZE_HD.to_tuple_int()

    im3 = pil_to_bgr(im2)
    assert isinstance(im3, np.ndarray)

    assert ImageNda(data=im3) == im1


def main() -> None:
    """比较ndarray和PIL Image"""
    np_image = np.array([[[0, 100, 200]]], dtype=np.uint8).repeat(4, 0).repeat(4, 1)

    print(type(np_image))

    print(np_image)
    print(np_image.shape)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    pil_image = fromarray(np_image)
    print(type(pil_image))

    trans1 = transforms.Compose(
        [
            transforms.Resize((4, 4)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    image1 = trans1(pil_image)

    print("image1:", image1)

    print(np.array(image1))


if __name__ == "__main__":
    main()
