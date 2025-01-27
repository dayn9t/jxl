from pathlib import Path

from jcx.sys.fs import StrPath, with_parent, remake_dir, files_in, make_parents
from jcx.text.txt_json import load_json
from jvi.image.image_nda import ImageNda
from jxl.label.info import ImageLabelInfo, ImageLabelInfos, IMG_EXT, ImageLabelPairs
from jxl.label.meta import meta_fix


def label_tail(meta_id: int, ext: str) -> str:
    """获取标签文件尾部"""
    return f'_{meta_fix(meta_id)}{ext}'


def label_path_of(img_file: StrPath, format_name: str, meta_id: int, ext: str) -> Path:
    """获取图像对应的标注文件路径"""
    file = Path(img_file).with_suffix(ext)
    return with_parent(file, f'{format_name}_{meta_fix(meta_id)}')


def load_label_dir(folder: StrPath, meta_id: int) -> ImageLabelInfos:
    """加载目录下的图片标注记录"""
    folder = Path(folder)
    rs = []
    tail = label_tail(meta_id, '.todo')
    print('tail:', tail)

    files = sorted(folder.rglob("*" + tail))
    for lbl_file in files:
        label = load_json(lbl_file, ImageLabelInfo)
        rs.append(label)
    return rs


def dump_label_prop(label_pairs: ImageLabelPairs, dst: Path, category_id: int, prop_name: str, keep_dst_dir: bool,
                    prefix: str) -> int:
    """保存标注多项属性, 生成分类样本"""

    if not keep_dst_dir:
        remake_dir(dst)

    total = 0
    for file, label in label_pairs:
        image = ImageNda.load(file)
        n = 0
        for o in label.objects:
            if o.prob_class.value == category_id:
                cat = o.prop(prop_name).value
                if cat < 0:
                    continue
                n += 1
                path = dst / str(cat) / f'{prefix}{file.stem}_{n:04}{IMG_EXT}'
                # print('[INFO] dump %d:' % i, path)
                obj_img = image.roi(o.rect())
                obj_img.save(path)
        total += n
    return total


def dump_label_prop_demo(label_pairs: ImageLabelPairs, dst_dir: Path) -> int:
    """保存标注多项属性, 生成分类样本"""

    remake_dir(dst_dir)

    total = 0
    for file, label in label_pairs:
        image = ImageNda.load(file)
        n = 0
        for o in label.objects:
            if o.prob_class.value >= 0:
                n += 1
                path = dst_dir / file.stem / f'{o.id:02}{IMG_EXT}'
                make_parents(path)
                # print('[INFO] dump %d:' % i, path)
                obj_img = image.roi(o.rect())
                obj_img.save(path)
        total += n
    return total


if __name__ == '__main__':
    # path_test()
    # load_label_records_test()
    # dump_label_prop_test()
    # image_path_of_label_test()
    pass
