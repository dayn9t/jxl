from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from jvi.image.image_nda import ImageNda
from jvi.image.trans import bgr_to_pil
from loguru import logger
from torch import Tensor
from torch.nn import functional
from torchsummary import summary
from torchvision import transforms

from jxl.cls.arch.torch_image import load_pth_tar
from jxl.cls.classifier import ClassifierOpt, ClassifierRes, IClassifier, ModelFormat
from jxl.label.prop import ProbValue

if TYPE_CHECKING:
    from PIL.Image import Image as PilImage

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@dataclass(frozen=True)
class ClassifierTchRes(ClassifierRes):
    """图片分类器结果"""

    output: Tensor

    def top(self) -> ProbValue:
        """最可能类别"""
        m = torch.max(self.output.data, 0)
        return ProbValue(int(m.indices.item()), float(m.values.item()))

    def confidences(self) -> list[float]:
        """获取各个分类的置信度"""
        return self.output.tolist()

    def __len__(self) -> int:
        return len(self.output)


class ClassifierTch(IClassifier):
    """图片分类器"""

    model_class = "image_net"

    def __init__(self, model_path: Path, opt: ClassifierOpt, device_name: str = "") -> None:
        super().__init__(model_path, opt, device_name)

        if opt.data_format == ModelFormat.FULL_MODEL:
            model: torch.nn.Module = torch.load(model_path, weights_only=False)
        else:
            model = load_pth_tar(opt.num_classes, model_path)

        model = model.cuda()
        model.eval()  # 固定dropout/归一化层，否则每次推理结果不同

        self.model = model
        self.input_shape = opt.input_shape

        trans = [
            transforms.Resize(opt.input_shape),
            transforms.ToTensor(),
        ]
        if opt.normalized:
            trans.append(normalize)
        self.trans = transforms.Compose(trans)

    def show_detail(self) -> None:
        """显示细节信息"""
        summary(self.model, (3, 224, 224))
        logger.info("{}", self.model)

    def __str__(self) -> str:
        s = self.model.__str__()
        assert isinstance(s, str)
        return s

    def num_parameters(self) -> int:
        """TODO: 用途？"""
        return sum(torch.numel(parameter) for parameter in self.model.parameters())

    def __call__(self, img_bgr: ImageNda) -> ClassifierTchRes:
        """分类输入图像, 图像尺寸无限制"""
        # print('image shape0:', image.size)

        img: PilImage = bgr_to_pil(img_bgr.data())
        # print('img:', type(img))
        img_tensor: Tensor = self.trans(img)
        assert img_tensor.shape == torch.Size([3, 224, 224])
        img_tensor = img_tensor.view(
            1, 3, self.input_shape[0], self.input_shape[1]
        ).cuda()  # 多GPU可能接受CPU图片
        assert img_tensor.shape == torch.Size([1, 3, 224, 224])

        # print('image:', type(img))
        # print('image shape=', img.shape, 'dtype=', img.dtype)

        output = self.model(img_tensor)
        output = functional.softmax(output[0], dim=0)

        return ClassifierTchRes(output)

    def save(self, file: Path) -> None:
        """必须在至少一次推理后保存，格式.pth"""
        logger.info("model save to: {}", file)
        torch.save(self.model, file)
