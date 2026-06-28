from pathlib import Path
from typing import Final

from jcx.sys.fs import StrPath, files_in, make_parents, remake_dir, with_parent
from jcx.text.txt_json import load_json
from jvi.image.image_nda import ImageNda

from jxl.label.a2d.dd import IMG_EXT, A2dImageLabel, A2dImageLabelPairs, A2dImageLabels
from jxl.label.meta import meta_fix


def label_tail(meta_id: int, ext: str) -> str:
    """获取标签文件尾部"""
    return f"_{meta_fix(meta_id)}{ext}"


def label_path_of(img_file: StrPath, format_name: str, meta_id: int, ext: str) -> Path:
    """获取图像对应的标注文件路径"""
    file = Path(img_file).with_suffix(ext)
    return with_parent(file, f"{format_name}_{meta_fix(meta_id)}")


LABEL_EXT: Final[str] = ".json"
"""标注文件扩展名"""


def load_image_label_pairs(
    folder: StrPath, meta_id: int, format_name: str
) -> A2dImageLabelPairs:
    """加载目录下指定格式的图像-标注对.

    读取 ``MetaDataset`` 写入的结构::

        {folder}/image/{stem}.jpg
        {folder}/{format_name}_m{meta_id}/{stem}.json

    Args:
        folder: 数据集根目录
        meta_id: 元数据 ID
        format_name: 标注格式名(目录前缀), 如 ``hop`` / ``a2d``

    Returns:
        图像与标注对集

    Raises:
        FileNotFoundError: 标注目录不存在时
    """
    root = Path(folder)
    image_dir = root / "image"
    label_dir = root / f"{format_name}_{meta_fix(meta_id)}"

    if not label_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {label_dir}")

    pairs: A2dImageLabelPairs = []
    for label_file in files_in(label_dir, LABEL_EXT):
        label = load_json(label_file, A2dImageLabel).unwrap()
        image_file = image_dir / (label_file.stem + IMG_EXT)
        pairs.append((image_file, label))
    return pairs


def load_label_dir(folder: StrPath, meta_id: int) -> A2dImageLabels:
    """加载目录下的图片标注记录"""
    folder = Path(folder)
    rs = []
    tail = label_tail(meta_id, ".todo")

    files = sorted(folder.rglob("*" + tail))
    for lbl_file in files:
        label = load_json(lbl_file, A2dImageLabel).unwrap()
        rs.append(label)
    return rs


def dump_label_prop(  # noqa: PLR0913
    label_pairs: A2dImageLabelPairs,
    dst: Path,
    category_id: int,
    prop_id: int,
    keep_dst_dir: bool,
    prefix: str,
) -> int:
    """保存标注多项属性, 生成分类样本"""

    if not keep_dst_dir:
        remake_dir(dst)

    total = 0
    for file, label in label_pairs:
        image = ImageNda.load(file)
        n = 0
        for o in label.objects:
            if o.prob_class.value == category_id:
                cat = o.prop(prop_id).value
                if cat < 0:
                    continue
                n += 1
                path = dst / str(cat) / f"{prefix}{file.stem}_{n:04}{IMG_EXT}"
                # print('[INFO] dump %d:' % i, path)
                obj_img = image.roi(o.rect())
                obj_img.save(path)
        total += n
    return total


def dump_label_prop_demo(label_pairs: A2dImageLabelPairs, dst_dir: Path) -> int:
    """保存标注多项属性, 生成分类样本"""

    remake_dir(dst_dir)

    total = 0
    for file, label in label_pairs:
        image = ImageNda.load(file)
        n = 0
        for o in label.objects:
            if o.prob_class.value >= 0:
                n += 1
                path = dst_dir / file.stem / f"{o.id:02}{IMG_EXT}"
                make_parents(path)
                # print('[INFO] dump %d:' % i, path)
                obj_img = image.roi(o.rect())
                obj_img.save(path)
        total += n
    return total


if __name__ == "__main__":
    # path_test()
    # load_label_records_test()
    # dump_label_prop_test()
    # image_path_of_label_test()
    pass
