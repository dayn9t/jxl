from pathlib import Path

from jcx.data.split import random_split
from jcx.sys.fs import StrPath, dirs_in, files_in, link_files, remake_subdir
from jvi.image.image_nda import is_image

from jxl.label.darknet.darknet_set import img2label


def remake_dirs(dst: StrPath) -> list[str]:
    dst_dirs = ["train", "val", "test"]
    # print('重建目录:')
    for _i, d in enumerate(dst_dirs):
        # dst_dir = dst / d
        # print('  #%d' % i, dst_dir)
        remake_subdir(dst, d)
    return dst_dirs


def dataset_split(src: Path, dst: Path, radio: list[float], ext: str = ".jpg") -> None:
    """分个样本集合，可处理样本集：darknet检测 & 图片分类"""
    dir_images = src / "images"
    dir_labels = src / "labels"
    dir_0 = src / "0"
    dir_1 = src / "1"
    dir_a = src / "A"
    dir_b = src / "B"

    if dir_images.is_dir() and dir_labels.is_dir():
        darknet_split(src, dst, radio, ext)  # FIXME: 移动到Darknet
    elif dir_0.is_dir() and dir_1.is_dir():  # classification
        class_split(src, dst, radio, ext)
    elif dir_a.is_dir() and dir_b.is_dir():
        cd_split(src, dst, radio, ext)
    else:
        msg = "Invalid dataset"
        raise RuntimeError(msg)


def class_split(src: Path, dst: Path, radio: list[float], ext: str = ".jpg") -> int:
    """分割分类样本集"""

    dst_dirs = remake_dirs(dst)
    class_dirs = dirs_in(src)
    total = 0
    for class_dir in class_dirs:
        files = files_in(class_dir, ext)
        count = len(files)
        total += count

        file_groups = random_split(list(files), radio)
        for i, file_group in enumerate(file_groups):
            show_dir = Path(dst_dirs[i], class_dir.name)
            dst_dir = dst / show_dir
            link_files([Path(str(f)) for f in file_group], dst_dir)

    return total


def darknet_split(src: Path, dst: Path, radio: list[float], ext: str = ".jpg") -> int:
    """分割darknet检测样本集"""

    dst_dirs = remake_dirs(dst)
    images = files_in(src / "images", ext)

    image_groups = random_split(list(images), radio)
    for i, image_group in enumerate(image_groups):
        image_paths = [Path(str(f)) for f in image_group]
        label_group = list(map(img2label, image_paths))
        Path(dst_dirs[i], "images")
        image_dir = dst / dst_dirs[i] / "images"
        label_dir = dst / dst_dirs[i] / "labels"

        link_files(image_paths, image_dir)
        link_files(label_group, label_dir)

    return len(images)


def path_replace(src: Path, parent: str, ext: str) -> Path:
    """替换路径部分"""
    dst = (src.parent.parent / parent / src.name).with_suffix(ext)
    assert src.exists()
    assert dst.exists()
    return dst


def cd_split(src: Path, dst: Path, radio: list[float], ext: str = ".png") -> int:
    """分割变化检测样本集"""

    dst_dirs = remake_dirs(dst)
    labels = files_in(src / "label", ext)

    label_groups = random_split(list(labels), radio)
    for i, label_group in enumerate(label_groups):
        label_paths = [Path(str(f)) for f in label_group]
        a_group = [path_replace(f, "A", ext) for f in label_paths]
        b_group = [path_replace(f, "B", ext) for f in label_paths]
        Path(dst_dirs[i], "label")
        a_dir = dst / dst_dirs[i] / "A"
        b_dir = dst / dst_dirs[i] / "B"
        label_dir = dst / dst_dirs[i] / "label"

        link_files(label_paths, label_dir, is_image)
        link_files(a_group, a_dir, is_image)
        link_files(b_group, b_dir, is_image)

    return len(labels)
