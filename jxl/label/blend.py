from pathlib import Path
from random import choices

from jcx.m.fun import Linear
from jcx.m.rand import random_choices
from jcx.sys.fs import StrPath
from jvi.geo.rectangle import random_point
from jvi.image.io import load_images_in, load_image_pairs_in
from jvi.image.proc import *


def random_object_pos(size: Size, size_ob: Size) -> Rect:
    """获取目标随即位置, 近大远小"""
    thr_side_n = 0.1  # 边长阈值, 放弃边长低于该值的边长, 避免出现太小目标
    y0_side_n = 0.08  # y=0时的边长
    y1_side_n = 0.3  # y=1时的边长
    times = 1000  # 尝试次数

    fun = Linear.from_xyxy(0, y0_side_n, size.height, y1_side_n)

    rect = Rect.from_size(size)
    for i in range(times):
        p = random_point(size).round()
        side_n = fun(p.y)  # 归一化边长
        if side_n < thr_side_n:
            continue
        box_size = size.scale(side_n)
        size_ob = size_ob.scale_in(box_size)
        rect_ob = Rect.from_ps(p, size_ob).round()
        if rect.contains(rect_ob):
            return rect_ob

    raise RuntimeError("random_object_pos failed")


def random_blend(objects: ImageNdas, im_bg: ImageNda, fg_mask: ImageNda) -> None:
    """把目标图像混合到背景图的随即位置"""
    ksize = (5, 5)
    for im_ob in objects:
        rect = random_object_pos(im_bg.size(), im_ob.size())
        assert im_ob.channel_num() == 4
        ico_c4 = resize(im_ob, dst_size=rect.size())
        ico_c4 = blur(ico_c4, ksize=ksize)
        alpha: ImageNda = ico_c4.channel_at(3)
        ico_c3 = ImageNda(data=ico_c4.data()[..., 0:3])
        ico_c3 = equalize_hist(ico_c3)  # 直方图均衡

        alpha_blend(ico_c3, im_bg.roi(rect), alpha)

        dst = fg_mask.roi(rect).data()
        cv2.bitwise_or(alpha.data(), dst, dst)

        threshold(fg_mask, 128, fg_mask)
        erode(fg_mask, fg_mask, (5, 5))


class ObjectBlender:
    """目标混合器"""

    def __init__(
        self,
        multiple: int = 10,
        ob_count: int = 3,
        ext: str = ".png",
        size: Size = Size(1024, 1024),
        verbose: bool = False,
    ):
        self._multiple = multiple
        self._ob_num = ob_count
        self._ext = ext
        self._verbose = verbose
        self._objects: ImageNdas = []
        self._count = 0

        self._im_a = ImageNda(size)
        self._im_b = ImageNda(size)
        self._im_label = ImageNda(size, 1)

    def load_objects(self, folder: StrPath, ext: str = ".png") -> int:
        self._objects = load_images_in(folder, ext)
        return len(self._objects)

    def _make_sample(
        self, im1: ImageNda, im2: ImageNda, dst_dir: Path, prefix: str
    ) -> None:
        assert im1.same_shape_type(im2), "图片尺寸/数据类型必须一致"
        assert im1.same_shape_type(self._im_a), "图片尺寸/数据类型必须一致"

        self._im_label.set_to(0)

        im1.copy_to(self._im_a)
        im2.copy_to(self._im_b)

        obs = choices(self._objects, k=self._ob_num)
        random_blend(obs, self._im_b, self._im_label)

        self._count += 1
        name = f"{prefix}_{self._count:04}{self._ext}"

        m = {
            dst_dir / "A": self._im_a,
            dst_dir / "B": self._im_b,
            dst_dir / "label": self._im_label,
        }

        if self._verbose:
            print(f"#{self._count} save:", dst_dir / "x" / name)

        for p, im in m.items():
            im.save(p / name)
            im1 = ImageNda.load(p / name)
            assert im1.same_shape_type(im)  # 验证, 出现过两次坏的PNG

    def make_samples(self, src_dir: StrPath, dst_dir: StrPath) -> int:
        """制作样本集合"""
        pairs = load_image_pairs_in(src_dir, self._ext)
        prefix = Path(src_dir).name
        n = min(len(pairs) - 1, self._multiple)
        for path, im1 in pairs:
            print(f"draw: {path}")
            pairs1 = random_choices(pairs, n, (path, im1))
            for _, im2 in pairs1:
                self._make_sample(im1, im2, Path(dst_dir), prefix)
        return self._count
