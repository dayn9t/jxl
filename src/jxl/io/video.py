from jcx.sys.fs import StrPath
from jvi.drawing.color import random_color, Colors, COLORS7
from jvi.drawing.shape import polylines
from jvi.geo.point2d import Points
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size, SIZE_VGA
from jvi.image.image_nda import ImageNda, ImageNdas
from jvi.image.io import save_images
from jxl.det.idetector import DetObjects, DetObject
from pydantic import BaseModel


class CmpVideoMaker(BaseModel):
    """制作图像对比视频制作器"""

    repeat = 5
    fps = 0.5
    colors: Colors = COLORS7
    size = Size()
    line_thickness = 2

    def make(
        self, images: ImageNdas, objects: DetObjects, file: StrPath, roi: Points
    ) -> None:
        """制作图像对比视频"""
        images = [im.clone() for im in images]

        for im, color in zip(images, self.colors):
            if roi:
                polylines(im, roi, color, self.line_thickness)
            for ob in objects:
                polylines(im, ob.polygon, random_color(), self.line_thickness)
            # trace_image(im)

        save_images(images * self.repeat, file, fps=self.fps)


def demo_maker() -> None:
    maker = CmpVideoMaker()
    image = ImageNda(SIZE_VGA)
    r1 = Rect(0.0, 0.0, 0.5, 0.5)
    r2 = Rect(0.5, 0.5, 0.5, 0.5)
    ob = DetObject.new(0, 1.0, r1)
    maker.make([image] * 7, [ob], "/tmp/a_maker.mp4", r2.vertexes())


if __name__ == "__main__":
    demo_maker()
