from collections.abc import Sequence
from pathlib import Path

import cv2
from jcx.sys.fs import remake_subdir
from jcx.text.txt_json import load_json, to_json
from jcx.ui.key import Key
from jcx.util.algo import list_index
from jvi.drawing.color import COLORS7
from jvi.geo.rectangle import Rect
from jvi.image.image_nda import ImageNda
from jvi.image.util import ndarray_rect
from loguru import logger
from pydantic import BaseModel


class CocoCategory(BaseModel):
    """COCO 类别定义（仅建模代码实际访问的字段）。"""

    id: int
    name: str


class CocoImage(BaseModel):
    """COCO 图片记录（仅建模代码实际访问的字段）。"""

    id: int
    file_name: str
    width: int
    height: int


class CocoAnnotation(BaseModel):
    """COCO 标注框（仅建模代码实际访问的字段；标注 id 未被使用故不建模）。"""

    image_id: int
    category_id: int
    bbox: list[float]


class CocoDataset(BaseModel):
    """COCO 数据集合的顶层结构（仅建模代码实际访问的三个列表）。"""

    categories: list[CocoCategory]
    images: list[CocoImage]
    annotations: list[CocoAnnotation]


class CocoLabel(BaseModel):
    """单张图片聚合后的运行时标签（image_info 产物，非 COCO JSON 原生结构）。

    引入此模型以彻底消除 orjson 加载链下游的裸 dict 访问（file/size/annotations）。
    """

    file: str
    size: list[float]
    annotations: list[CocoAnnotation]


def rect2pp(r: Sequence[float]) -> tuple[tuple[int, int], tuple[int, int]]:
    """矩形化两点"""
    p1 = (int(r[0]), int(r[1]))
    p2 = (int(r[0] + r[2]), int(r[1] + r[3]))
    return p1, p2


def rect2ncr(r: Rect, size: Sequence[float]) -> tuple[float, float, float, float]:
    """矩形转归一化中心矩形"""
    x = (r.x + r.width / 2) / size[0]
    y = (r.y + r.height / 2) / size[1]
    w = r.width / size[0]
    h = r.height / size[1]
    return x, y, w, h


def show_label(label: CocoLabel) -> bool:
    """展示标签"""
    img = ImageNda.load(label.file)
    thickness = 2

    for a in label.annotations:
        p1, p2 = rect2pp(a.bbox)
        color = COLORS7[a.category_id - 1]
        # TODO(dayn9t): cv2 stubs 不识别 jvi ImageNda，需 ImageNda.data() 转 ndarray。
        # show_label 为遗留可视化函数，后续重构时统一走 jvi.drawing.shape。
        img = cv2.rectangle(img, p1, p2, color, thickness)  # type: ignore[call-overload]

    # cv2.imshow(label.file, img)
    logger.info(f"file: {label.file}")
    cv2.imshow("coco label viewer", img)  # type: ignore[call-overload]
    return cv2.waitKey(0) != Key.ESC.value  # ESC 退出


def image_info(c: CocoImage) -> CocoLabel:
    """提取图片信息"""
    return CocoLabel(
        file=c.file_name,
        size=[c.width, c.height],
        annotations=[],
    )


class DataCoco:
    """Coco数据集合"""

    def __init__(self, coco_json: Path) -> None:
        coco = load_json(coco_json, CocoDataset).unwrap()

        self.cats: dict[int, str] = {c.id: c.name for c in coco.categories}
        self.labels: dict[int, CocoLabel] = {c.id: image_info(c) for c in coco.images}

        for a in coco.annotations:
            self.labels[a.image_id].annotations.append(a)
        logger.info(f"COCO cats: {self.cats}")

    def find_cat(self, name: str) -> int | None:
        """查找指定名称的类别ID"""
        for k, v in self.cats.items():
            if v == name:
                return k
        return None

    def show(self, i: int) -> None:
        logger.info(f"COCO images: {to_json(self.labels[i])}")

        show_label(self.labels[i])

    def dump_darknet(
        self,
        output_dir: Path,
        cat_names: list[str] | None = None,
        rect: Rect | None = None,
        verbose: bool = False,
    ) -> None:
        """保存"""
        if cat_names is None:
            cat_map = {c: c - 1 for c in self.cats}
        else:
            cat_map = {
                c: list_index(cat_names, name).unwrap() for c, name in self.cats.items()
            }

        logger.info(f"cat_map: {cat_map}")
        # return

        image_dir = remake_subdir(output_dir, "images")
        label_dir = remake_subdir(output_dir, "labels")
        pending = self.find_cat("pending")
        for k, v in self.labels.items():
            skip = False
            for a in v.annotations:
                if a.category_id == pending:
                    skip = True
                    break
            if skip:
                logger.info(f"  skip file: {v.file}")
                continue

            image_file = Path(image_dir, f"{k:04d}.jpg")
            label_file = Path(label_dir, f"{k:04d}.txt")
            if verbose:
                logger.info(f"image: {image_file}")

            if rect:
                image = ImageNda.load(v.file)
                roi = ndarray_rect(image.data(), rect)
                cv2.imwrite(str(image_file), roi)
            else:
                image_file.symlink_to(v.file)

            with open(label_file, "w") as f:
                for a in v.annotations:
                    cat = cat_map[a.category_id]
                    r = Rect(*a.bbox)
                    if rect:
                        if not rect.contains(r):
                            logger.error(f"invalid annotation: {v.file}")
                        r.x -= rect.x
                        r.y -= rect.y
                    xywh = rect2ncr(r, v.size)
                    # print('\t', c, a.bbox)
                    f.write(("%g " * 5 + "\n") % (cat, *xywh))
