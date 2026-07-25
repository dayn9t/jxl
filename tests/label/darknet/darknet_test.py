from pathlib import Path

from jxl.label.darknet.darknet_set import img2label


def test_img2label() -> None:
    """从图片文件获取标注文件路径 (images/x.jpg -> labels/x.txt)。"""
    image = Path("sign/3.14.jpg")
    label = Path("labels/3.14.txt")
    assert img2label(image) == label
