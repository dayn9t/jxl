import shutil
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from jcx.sys.fs import dirs_in, files_in
from jcx.ui.key import Key
from jvi.geo.size2d import Size
from jvi.image.image_nda import ImageNda
from jvi.image.trace import close_all_windows, trace_image

from jxl.cls.classifier import ClassifierOpt, IClassifier
from jxl.cls.classifier_y8 import ClassifierY8


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
        return f"conf={self.conf:.2f}, class={self.index} file={self.file}"


class DatasetChecker:
    """数据集审核工具"""

    def __init__(
        self,
        model: Path,
        opt: Namespace,
        max_conf: float,
        top_num: int = 10,
        ext: str = ".jpg",
    ) -> None:
        self.top_num = top_num
        self.ext = ext
        self.max_conf = max_conf
        self.review = opt.review
        self.verbose = opt.verbose
        cls_opt = ClassifierOpt(
            (opt.img_size, opt.img_size), opt.num_classes, not opt.non_normalized
        )
        # print('classifier opt:', cls_opt)
        self.classifier = ClassifierY8(model, cls_opt)

    def check(self, dataset: Path, class_id: int | None) -> None:
        """数据审核"""

        if self.review:
            pass

        class_dirs = dirs_in(dataset) if class_id is None else [dataset / str(class_id)]

        total = 0
        err = 0
        for class_dir in class_dirs:
            n, e = self.deal_class(class_dir, self.classifier, self.max_conf)
            if n < 1:
                pass
            else:
                100 * e / n
            total += n
            err += e
        if total > 0:
            100 * err / total

        close_all_windows()

    def deal_class(
        self, class_dir: Path, classifier: IClassifier, max_conf: float
    ) -> tuple[int, int]:
        """计算低置信度的样本"""
        files = files_in(class_dir, self.ext)

        count = len(files)

        class_id = int(class_dir.name)
        conf_files: list[ConfFile] = []
        err = 0
        for file in files:
            im = ImageNda.try_load(file)  # BGR
            if im.is_err():
                continue
            ret = classifier(im.unwrap())
            # print('ret:', ret)
            if len(ret) < 1:
                continue
            if ret.top_index() != class_id:
                if self.verbose:
                    pass
                err += 1
                conf_files.append(
                    ConfFile(-ret.top_confidence(), ret.top_index(), file)
                )
            elif ret.top_confidence() < max_conf:
                conf_files.append(ConfFile(ret.top_confidence(), ret.top_index(), file))

        conf_files.sort()

        if self.review:
            self.view_wrongs(conf_files)

        return count, err

    def view_wrongs(self, conf_files: list[ConfFile]) -> None:
        for i in range(min(self.top_num, len(conf_files))):
            file = conf_files[i].file
            im = ImageNda.load(str(file))
            key, _ = trace_image(
                im, "IC_WIN", auto_close=False, box_size=Size.new(512, 512)
            )
            if key == Key.ESC:
                sys.exit(0)
            elif key == Key.DEL:
                file.unlink(missing_ok=True)
            elif ord("0") <= key <= ord("9"):
                dst_class = key - ord("0")
                dst_dir = file.parent.parent / str(dst_class)
                try:
                    shutil.move(str(file), str(dst_dir))
                except shutil.Error:
                    file.unlink(missing_ok=True)
