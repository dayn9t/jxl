from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms  # type: ignore
from jiv.image.image_nda import ImageNda
from jml.cls.arch.torch_image import load_pth_tar
from jml.cls.classifier import ClassifierRes, IClassifier, ClassifierOpt, ModelFormat
from jml.label.info import ProbValue
from jiv.image.trans import bgr_to_pil, PilImage
from torch import Tensor
from torchsummary import summary  # type: ignore

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@dataclass
class ClassifierTchRes(ClassifierRes):
    """图片分类器结果"""
    output: Any

    def top(self) -> ProbValue:
        """最可能类别"""
        m = torch.max(self.output.data, 0)
        return ProbValue(m.indices.item(), m.values.item())

    def confidences(self) -> List[float]:
        """获取各个分类的置信度"""
        return self.output.tolist()

    def __len__(self) -> int:
        return len(self.output)


class ClassifierTch(IClassifier):
    """图片分类器"""

    model_class = 'image_net'

    def __init__(self, model_path: Path, opt: ClassifierOpt, device_name: str = ''):
        super().__init__(model_path, opt, device_name)

        if opt.data_format == ModelFormat.FULL_MODEL:
            model = torch.load(model_path)
        else:
            model = load_pth_tar(opt.num_classes, model_path)
        # print(opt)
        # show_keys(model.state_dict(), 'full:')

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
        print(self.model)

    def __str__(self) -> str:
        return self.model.__str__()

    def num_parameters(self) -> int:
        """ TODO: 用途？"""
        return sum(torch.numel(parameter) for parameter in self.model.parameters())

    def __call__(self, img_bgr: ImageNda) -> ClassifierTchRes:
        """分类输入图像, 图像尺寸无限制"""
        # print('image shape0:', image.size)

        img: PilImage = bgr_to_pil(img_bgr.data())
        # print('img:', type(img))
        img_tensor: Tensor = self.trans(img)
        assert img_tensor.shape == torch.Size([3, 224, 224])
        img_tensor = img_tensor.view(1, 3, self.input_shape[0], self.input_shape[1]).cuda()  # 多GPU可能接受CPU图片
        assert img_tensor.shape == torch.Size([1, 3, 224, 224])

        # print('image:', type(img))
        # print('image shape=', img.shape, 'dtype=', img.dtype)

        output = self.model(img_tensor)
        output = functional.softmax(output[0], dim=0)

        return ClassifierTchRes(output)

    def save(self, file: Path) -> None:
        """必须在至少一次推理后保存，格式.pth"""
        print('model save to:', file)
        torch.save(self.model, file)
