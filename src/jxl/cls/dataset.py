from pathlib import Path

from jcx.data.split import random_split
from jcx.sys.fs import StrPath, dirs_in, files_in, link_files, remake_subdir
from jvi.image.image_nda import is_image
from loguru import logger

from jxl.label.darknet.darknet_set import img2label


def remake_dirs(dst: StrPath) -> list[str]:
    dst_dirs = ["train", "val", "test"]
    # print('重建目录:')
    for _i, d in enumerate(dst_dirs):
        # dst_dir = dst / d
        # print('  #%d' % i, dst_dir)
        remake_subdir(dst, d)
    return dst_dirs


def dataset_split(src: Path, dst: Path, radio: list[int], ext: str = ".jpg") -> None:
    """分个样本集合，可处理样本集：darknet检测 & 图片分类"""
    dir_images = src / "images"
    dir_labels = src / "labels"
    dir_0 = src / "0"
    dir_1 = src / "1"
    dir_a = src / "A"
    dir_b = src / "B"

    if dir_images.is_dir() and dir_labels.is_dir():
        logger.info(f"Darknet样本集：{src} => {dst}")
        total = darknet_split(src, dst, radio, ext)  # FIXME: 移动到Darknet
    elif dir_0.is_dir() and dir_1.is_dir():  # classification
        logger.info(f"图片分类样本集：{src} => {dst}")
        total = class_split(src, dst, radio, ext)
    elif dir_a.is_dir() and dir_b.is_dir():
        logger.info(f"变更检测样本集：{src} => {dst}")
        total = cd_split(src, dst, radio, ext)
    else:
        raise RuntimeError("Invalid dataset")
    logger.info(f"样本总数：{total}\n")


def class_split(src: Path, dst: Path, radio: list[int], ext: str = ".jpg") -> int:
    """分割分类样本集"""

    dst_dirs = remake_dirs(dst)
    class_dirs = dirs_in(src)
    total = 0
    for class_dir in class_dirs:
        files = files_in(class_dir, ext)
        count = len(files)
        total += count
        logger.info(f"链接类别：{class_dir.name} ({count})")

        # TODO(dayn9t): jcx.random_split 签名用 list[object]（不变性，list[Path] 不兼容），
        # 应在 jcx 侧改用 TypeVar；此处暂显式转换。
        file_groups = random_split(files, radio)  # type: ignore[arg-type]
        for i, file_group in enumerate(file_groups):
            show_dir = Path(dst_dirs[i], class_dir.name)
            dst_dir = dst / show_dir
            logger.info(f"- {show_dir!s:<12}\t{len(file_group): 6}")
            link_files(file_group, dst_dir)  # type: ignore[arg-type]

    return total


def darknet_split(src: Path, dst: Path, radio: list[int], ext: str = ".jpg") -> int:
    """分割darknet检测样本集"""

    dst_dirs = remake_dirs(dst)
    images = files_in(src / "images", ext)

    image_groups = random_split(images, radio)  # type: ignore[arg-type]
    logger.info("链接分组：")
    for i, image_group in enumerate(image_groups):
        label_group = list(map(img2label, image_group))  # type: ignore[arg-type]
        show_dir = Path(dst_dirs[i], "images")
        image_dir = dst / dst_dirs[i] / "images"
        label_dir = dst / dst_dirs[i] / "labels"

        logger.info(f"- {show_dir!s:<12}\t{len(image_group): 6}")
        link_files(image_group, image_dir)  # type: ignore[arg-type]
        link_files(label_group, label_dir)

    return len(images)


def path_replace(src: Path, parent: str, ext: str) -> Path:
    """替换路径部分"""
    dst = (src.parent.parent / parent / src.name).with_suffix(ext)
    assert src.exists()
    assert dst.exists()
    return dst


def cd_split(src: Path, dst: Path, radio: list[int], ext: str = ".png") -> int:
    """分割变化检测样本集"""

    dst_dirs = remake_dirs(dst)
    labels = files_in(src / "label", ext)

    label_groups = random_split(labels, radio)  # type: ignore[arg-type]
    logger.info("链接分组：")
    for i, label_group in enumerate(label_groups):
        a_group = list(map(lambda f: path_replace(f, "A", ext), label_group))  # type: ignore[arg-type]
        b_group = list(map(lambda f: path_replace(f, "B", ext), label_group))  # type: ignore[arg-type]
        show_dir = Path(dst_dirs[i], "label")
        a_dir = dst / dst_dirs[i] / "A"
        b_dir = dst / dst_dirs[i] / "B"
        label_dir = dst / dst_dirs[i] / "label"

        logger.info(f"- {show_dir!s:<12}\t{len(label_group): 6}")
        link_files(label_group, label_dir, is_image)  # type: ignore[arg-type]
        link_files(a_group, a_dir, is_image)
        link_files(b_group, b_dir, is_image)

    return len(labels)
