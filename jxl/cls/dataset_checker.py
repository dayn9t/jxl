import shutil
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from colorama import Fore, Style
from jcx.sys.fs import dirs_in, files_in
from jcx.ui.key import Key
from jiv.geo.size2d import Size
from jiv.image.image_nda import ImageNda
from jiv.image.trace import trace_image, close_all_windows
from jml.cls.classifier import ClassifierOpt, IClassifier
from jml.cls.classifier_y8 import ClassifierY8


@dataclass(frozen=True, order=True)
class ConfFile:
    """分类器器选项"""
    conf: float
    """置信度"""
    index: int
    """类别索引 TODO: 修改颜色?"""
    file: Path
    """文件路径"""

    def __str__(self) -> str:
        return 'conf=%.2f, class=%d file=%s' % (self.conf, self.index, str(self.file))


class DatasetChecker:
    """数据集审核工具"""

    def __init__(self, model: Path, opt: Namespace, max_conf: float, top_num: int = 10, ext: str = '.jpg'):

        self.top_num = top_num
        self.ext = ext
        self.max_conf = max_conf
        self.review = opt.review
        self.verbose = opt.verbose
        cls_opt = ClassifierOpt(
            (opt.img_size, opt.img_size),
            opt.num_classes,
            not opt.non_normalized
        )
        # print('classifier opt:', cls_opt)
        self.classifier = ClassifierY8(model, cls_opt)

    def check(self, dataset: Path, class_id: Optional[int]) -> None:
        """数据审核"""

        if self.review:
            print('数据集审核工具，可以用按键[0~9]，修改错误分类, [DEL] 删除错误样本, [ESC] 退出')

        print('样本来源: %s\n' % dataset)
        if class_id is None:
            class_dirs = dirs_in(dataset)
        else:
            class_dirs = [dataset / str(class_id)]

        total = 0
        err = 0
        for class_dir in class_dirs:
            n, e = self.deal_class(class_dir, self.classifier, self.max_conf)
            if n < 1:
                print('[ERROR] %s 下不存在样本' % class_dir)
            else:
                r = 100 * e / n
                color = Fore.RED if r > 10 else ""
                print(f'  - 错误率：{color}{e}/{n} {r:.2f}%' + Style.RESET_ALL)
            total += n
            err += e
        if total > 0:
            r = 100 * err / total
            print('\n整体错误率：%d/%d %.2f%%\n' % (err, total, r))

        close_all_windows()

    def deal_class(self, class_dir: Path, classifier: IClassifier, max_conf: float) -> tuple[int, int]:
        """计算低置信度的样本"""
        files = files_in(class_dir, self.ext)

        count = len(files)

        class_id = int(class_dir.name)
        print('类别：%d (%d)' % (class_id, count))
        conf_files: List[ConfFile] = []
        err = 0
        for file in files:
            im = ImageNda.try_load(file)  # BGR
            if im.is_err():
                print('[ERROR] 无法加载:', file)
                continue
            ret = classifier(im.unwrap())
            # print('ret:', ret)
            if len(ret) < 1:
                print('[ERROR] 无效分类结果:', file)
                continue
            if ret.top_index() != class_id:
                if self.verbose:
                    print('[WARN] 分类结果错误:', ret.top_index(), file)
                err += 1
                conf_files.append(ConfFile(-ret.top_confidence(), ret.top_index(), file))
            elif ret.top_confidence() < max_conf:
                conf_files.append(ConfFile(ret.top_confidence(), ret.top_index(), file))

        conf_files.sort()

        if self.review:
            self.view_wrongs(conf_files)

        return count, err

    def view_wrongs(self, conf_files: List[ConfFile]) -> None:
        for i in range(min(self.top_num, len(conf_files))):
            file = conf_files[i].file
            print('#%d' % i, conf_files[i])
            im = ImageNda.load(str(file))
            key, _ = trace_image(im, 'IC_WIN', auto_close=False, box_size=Size(512, 512))
            if key == Key.ESC:
                exit(0)
            elif key == Key.DEL:
                file.unlink(missing_ok=True)
                print('  删除：%s' % str(file))
            elif ord('0') <= key <= ord('9'):
                dst_class = key - ord('0')
                dst_dir = file.parent.parent / str(dst_class)
                print('  移动：%s -> %s' % (file, dst_dir))
                try:
                    shutil.move(str(file), str(dst_dir))
                except shutil.Error:
                    file.unlink(missing_ok=True)
