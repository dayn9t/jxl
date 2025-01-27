from ias.app.afs import init_afs, afs
from jxl.label.darknet import *
from jxl.label.meta import find_meta

label_file = Path("n1_31015111120700111_2023-03-09_15-12-34.054.txt")
folder = Path("/home/jiang/ws/trash/cabin/dates/2023-03-09")


def test_file():
    print('darknet_dir:', folder)

    os1 = load_objects(folder / 'labels' / label_file)
    print('objects:', os1[0].rect())

    txt_file = Path('/tmp') / label_file
    print('txt_file:', txt_file)
    save_objects(os1, txt_file)

    os2 = load_objects(txt_file)
    print('objects:', os2[0].rect())


def test_img2label():
    """从图片文件获取标注文件"""
    f = Path('images/3.14.jpg')
    l = Path('labels/3.14.txt')
    assert img2label(f) == l


def test_darknet_set():
    init_afs('shtm', 'n1')

    assert DarknetSet.valid_set(folder, 0)

    dset = DarknetSet(folder)
    pairs = dset.load_pairs()

    # print(len(pairs))
    assert len(pairs) > 0


def test_labels():
    meta = find_meta(31, afs().meta_dir).unwrap()

    folder0 = Path("/home/jiang/ws/fire-smoke/darknet1")
    folder1 = Path("/home/jiang/ws/fire-smoke/darknet4")
    print('darknet_dir:', folder0)

    labels = darknet_load_labels(folder0)
    print('labels:', len(labels))

    darknet_dump_labels(labels, folder1, meta)

    labels = darknet_load_labels(folder1)
    print('labels:', len(labels))
