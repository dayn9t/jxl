"""图像序列"""
from pathlib import Path
from typing import List

from jcx.sys.fs import name_with_parents, stem_append
from jvi.geo.rectangle import Rect, Rects
from jvi.geo.size2d import Size
from jvi.image.image_nda import ImageNda
from jvi.image.trace import trace_image
from jvi.match.match import ImageMatcher


class Cutter2:
    """时间序列图像切割, 用以提供图图像场景样本"""

    def __init__(self, dst_dir: Path, sensitivity: int = 50, tile_size: Size = Size(224, 224)):
        self.tile_size = tile_size
        self.threshold = 20 * sensitivity / 50  # 距离阈值
        self.dst_dir = dst_dir
        self.matcher = ImageMatcher()

    def _check_moved(self, im1: ImageNda, im2: ImageNda) -> bool:
        """检测镜头移动"""
        dist = self.matcher.match(im1, im2)
        dist = round(dist, 2)
        print('dist:', dist)
        return dist > self.threshold

    def _cut_tile(self, ims: List[ImageNda], src_file: Path):
        """图像切割成块"""
        assert len(ims) > 1
        for im in ims:
            assert im.same_shape_type(ims[0])

        size = ims[0].size()

        dst_size = self.tile_size.scale((len(ims), 1))
        dst_im = ImageNda(dst_size)  # W*n 的图像
        rs = Rect.from_size(dst_size).to_tiles(size=self.tile_size)
        dst_rois = [dst_im.roi(r) for r in rs]

        rects: Rects = Rect.from_size(size).to_center_tiles(self.tile_size)

        for i, r in enumerate(rects):
            for j, im in enumerate(ims):
                src_roi: ImageNda = im.roi(r)
                src_roi.copy_to(dst_rois[j])

            trace_image(dst_im)
            if self._check_moved(dst_rois[0], dst_rois[1]):
                name = name_with_parents(src_file, 3)  # 文件名=节点_设备_日期_名称
                file = stem_append(self.dst_dir / name, f'_{i}')
                dst_im.save(file)

    def cut_files(self, files: list[Path]):
        """切割图片里表"""
        if len(files) < 2:
            return
        n = 2  # W*2 瓦片图
        images = [ImageNda.load(f) for f in files]

        for i in range(len(images) - n + 1):
            ims = images[i:i + n]
            self._cut_tile(ims, files[i + n - 1])
