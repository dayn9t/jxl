import cv2
import numpy as np
from jvi.image.image_nda import ImageNda


def estimate_clearness(image: ImageNda, ratio: float) -> int:
    """获取图像边缘直方图中指值v，另[0,v]区间内的颜色占据比例不低于ratio"""
    src = image.data()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # print(gray.dtype)
    edge = cv2.Sobel(gray, -1, 1, 1, ksize=3)
    # print('edge:', edge.shape, edge.dtype)
    hist = cv2.calcHist([edge], [0], None, [256], [0, 256])
    # print('hist:', hist.shape)
    hist = hist.flatten()
    # print('hist:', hist.shape)
    thr = ratio * np.sum(hist)
    s = 0.0
    for i, v in enumerate(hist):
        s += v
        if s > thr:
            return i
    return len(hist) - 1
