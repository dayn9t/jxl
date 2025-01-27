import json
import shutil
from pathlib import Path

import cv2

from jcx.text.txt_json import to_json
from jcx.ui.key import Key
from jcx.util.algo import list_index
from jiv.drawing.color import colors7
from jiv.geo import Rect
from jiv.image.util import ndarray_rect


def rect2pp(r):
    """矩形化两点"""
    p1 = (int(r[0]), int(r[1]))
    p2 = (int(r[0] + r[2]), int(r[1] + r[3]))
    return p1, p2


def rect2ncr(r: Rect, size):
    """矩形转归一化中心矩形"""
    x = (r.x + r.width / 2) / size[0]
    y = (r.y + r.height / 2) / size[1]
    w = r.width / size[0]
    h = r.height / size[1]
    return x, y, w, h


def show_label(label):
    """展示标签"""
    img = ImageNda.load(label['file'])
    thickness = 2

    for a in label['annotations']:
        p1, p2 = rect2pp(a['bbox'])
        color = colors7[a['category_id'] - 1]
        img = cv2.rectangle(img, p1, p2, color, thickness)

    # cv2.imshow(label['file'], img)
    print('file:', label['file'])
    cv2.imshow('coco label viewer', img)
    if cv2.waitKey(0) == Key.ESC.value:  # q to quit
        return False
    return True


def image_info(c):
    """提取图片信息"""
    return {'file': c['file_name'], 'size': [c['width'], c['height']], 'annotations': []}





class DataCoco:
    """Coco数据集合"""

    def __init__(self, coco_json: Path):
        with open(coco_json) as f:
            coco = json.load(f)

        self.cats = {c['id']: c['name'] for c in coco['categories']}
        self.labels = {c['id']: image_info(c) for c in coco['images']}

        for a in coco['annotations']:
            a1 = {'category_id': a['category_id'], 'bbox': a['bbox']}
            self.labels[a['image_id']]['annotations'].append(a1)
        print('COCO cats: ', self.cats)

    def find_cat(self, name: str):
        """查找指定名称的类别ID"""
        for k, v in self.cats.items():
            if v == name:
                return k
        return None

    def show(self, i):
        print('COCO images: ', to_json(self.labels[i]))

        show_label(self.labels[i])

    def dump_darknet(self, output_dir: Path, cat_names=None, rect=None, verbose=False):
        """保存"""
        if cat_names is None:
            cat_map = {c: c - 1 for c in self.cats.keys()}
        else:
            cat_map = {c: list_index(cat_names, name) for c, name in self.cats.items()}

        print('cat_map:', cat_map)
        # return

        image_dir = remake_subdir(output_dir, 'images')
        label_dir = remake_subdir(output_dir, 'labels')
        pending = self.find_cat('pending')
        for k, v in self.labels.items():
            skip = False
            for a in v['annotations']:
                if a['category_id'] == pending:
                    skip = True
                    break
            if skip:
                print('  skip file:', v['file'])
                continue

            image_file = Path(image_dir, '%04d.jpg' % k)
            label_file = Path(label_dir, '%04d.txt' % k)
            if verbose:
                print('image:', image_file)

            if rect:
                image = ImageNda.load(v['file'])
                roi = ndarray_rect(image, rect)
                cv2.imwrite(str(image_file), roi)
            else:
                image_file.symlink_to(v['file'])

            with open(label_file, 'w') as f:
                for a in v['annotations']:
                    cat = cat_map[a['category_id']]
                    if cat is not None:
                        r = Rect(*a['bbox'])
                        if rect:
                            if not rect.contains(r):
                                print('ERROR: invalid annotation', v['file'])
                            r.x -= rect.x
                            r.y -= rect.y
                        xywh = rect2ncr(r, v['size'])
                        # print('\t', c, a['bbox'])
                        f.write(('%g ' * 5 + '\n') % (cat, *xywh))
