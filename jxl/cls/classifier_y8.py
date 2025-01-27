import logging
from pathlib import Path

import torch
from jiv.image.image_nda import ImageNda
from jxl.cls.classifier import ClassifierRes, IClassifier, ClassifierOpt
from jxl.label.info import ProbValue
from ultralytics import YOLO
from ultralytics.engine.results import Results


class ClassifierResY8(ClassifierRes):
    """图片分类器结果"""

    def __init__(self, rs: Results) -> None:
        assert isinstance(rs.probs.data, torch.Tensor)
        self.probs = rs.probs.data

        m = torch.max(self.probs, 0)
        self.top1 = ProbValue(m.indices.item(), m.values.item())
        self.top1_name = rs.names[self.top1.value]

    def top(self) -> ProbValue:
        """最可能类别"""
        return self.top1

    def confidences(self) -> list[float]:
        """获取各个分类的置信度"""
        return self.probs.tolist()

    def __len__(self) -> int:
        return len(self.probs)


class ClassifierY8(IClassifier):
    """图片分类器"""

    model_class = 'image_net2'

    def __init__(self, model_path: Path, opt: ClassifierOpt, device_name: str = ''):
        super().__init__(model_path, opt, device_name)

        self._model = YOLO(model_path, task='classify')
        # opt 用不到, 也许可以考虑验证
        assert len(self._model.names) == opt.num_classes
        t = self._model.transforms.transforms[0]
        assert (t.size, t.size) == opt.input_shape

    def __str__(self) -> str:
        s = self._model.__str__()
        assert isinstance(s, str)
        return s

    def __call__(self, image: ImageNda) -> ClassifierResY8:
        """分类输入图像, 图像尺寸无限制"""

        # data = image.data()[:, :, ::-1]  # BGR => RGB
        data = image.data()
        rss = self._model(data, verbose=self._opt.verbose)
        assert isinstance(rss, list)
        assert len(rss) == 1
        rs = rss[0]
        assert isinstance(rs, Results)
        assert len(rs.names) == self._opt.num_classes

        return ClassifierResY8(rs)


def main() -> None:
    opt = ClassifierOpt((224, 224), 1000)
    path = Path('yolov8n-cls.pt')
    image: ImageNda = ImageNda.load('../../jiv/static/lena.jpg')
    # image = ImageNda.load('bus.jpg')

    model = ClassifierY8(path, opt, 'cuda')
    res = model(image)

    print(model)
    print(res.top(), res.confidences())


if __name__ == '__main__':
    main()
