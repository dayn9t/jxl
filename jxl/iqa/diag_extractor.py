from jiv.geo.rectangle import Rect
from jiv.geo.size2d import Size
from jiv.image.image_nda import ImageNda
from jiv.image.proc import resize_roi
from jiv.image.proc import to_gray, laplacian_edge
from jiv.image.stat import hist_vector, hue_hist
from jiv.match.match import ImageMatcher
from pydantic import BaseModel

DIV_COLS = 5
"""HD图像建议分块列数"""
DIV_ROWS = 3
"""HD图像建议分块行数"""
DIV_SIZE = Size(256, 240)
"""HD图像建议分块尺寸"""


def sharpness(image: ImageNda) -> list[float]:
    """获取锐度分布向量"""
    assert image.channel_num() == 3
    lap = laplacian_edge(to_gray(image))
    # calc_show_hist(lap, 16)
    return hist_vector(lap).tolist()  # type: ignore


def match(im1: ImageNda, im2: ImageNda) -> float:
    """图像匹配, 返回匹配的平均距离"""
    matcher = ImageMatcher()
    dist = matcher.match(im1, im2)  # FIXME, 图像匹配在平滑地面没法工作,
    return round(dist, 2)


class MatchVec:
    """提取两副图像的匹配向量"""

    def __init__(self, cols: int, rows: int, size: Size) -> None:
        self.de = DiagExtractor(Rect.one(), cols, rows, size)

    def __call__(self, im_a: ImageNda, im_b: ImageNda) -> list[float]:
        """提取两副图像的匹配向量"""
        features = self.de.extract(im_a, im_b)
        return features.match_vec


def chroma(image: ImageNda) -> list[float]:
    """获取主颜色"""
    assert image.channel_num() == 3
    hist, _ = hue_hist(image)
    arr = hist.tolist()
    assert isinstance(arr, list)
    return arr


def brightness(_im1: ImageNda) -> float:
    """获取亮度"""
    return 0


def interference(_im1: ImageNda) -> float:
    """检查干扰"""
    return 0


class TileFeatures(BaseModel):
    """图像瓦片特征集合"""
    rect: Rect
    """所在图像区域"""
    sharpness_vec: list[float]
    """锐度分布向量 (256,)"""


class ImageFeatures(BaseModel):
    """图像特征集合"""
    tiles: list[TileFeatures]
    """图像分块特征(n,)"""
    match_vec: list[float]
    """分块匹配向量 (n,)"""
    chroma_vec: list[float]
    """分块色度向量 (256,)"""


class DiagExtractor:
    """视频诊断特镇提取.

    功能：1-视频清晰度，2-视频角度，3-视频亮度，4-视频颜色，5-视频干扰
    """

    def __init__(self, roi: Rect, cols: int, rows: int, tile_size: Size) -> None:
        assert not tile_size.is_normalized()
        assert roi.is_normalized()

        self.tile_size = tile_size
        self.rects = list(roi.to_tiles(cols, rows))

    def extract(self, image1: ImageNda, image2: ImageNda) -> ImageFeatures:
        """获取匹配向量"""

        tile1 = ImageNda(self.tile_size)
        tile2 = ImageNda(self.tile_size)

        tiles = []
        match_vec = []
        for r in self.rects:
            # print('r:', r)
            resize_roi(image1, r, self.tile_size, tile1)
            resize_roi(image2, r, self.tile_size, tile2)
            # trace_images([tile1, tile2])

            tile = TileFeatures(
                rect=r,
                sharpness_vec=sharpness(tile1)
            )
            tiles.append(tile)
            match_vec.append(match(tile1, tile2))

        return ImageFeatures(
            tiles=tiles,
            match_vec=match_vec,
            chroma_vec=chroma(image1)
        )
