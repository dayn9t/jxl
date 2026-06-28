from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNetV2

StateDict = dict[str, torch.Tensor]


def create_resnet18(num_classes: int, state_dict: StateDict | None = None) -> nn.Module:
    model: nn.Module = torchvision.models.resnet18(num_classes=num_classes)
    if state_dict:
        state_dict["fc.weight"] = state_dict["fc.weight"][:num_classes, :]
        state_dict["fc.bias"] = state_dict["fc.bias"][:num_classes]
        model.load_state_dict(state_dict, strict=False)
    return model


def create_mobilenet_v2(
    num_classes: int, state_dict: StateDict | None = None
) -> nn.Module:
    model: nn.Module = MobileNetV2(num_classes=num_classes)
    if state_dict:
        state_dict["classifier.1.weight"] = state_dict["classifier.1.weight"][
            :num_classes, :
        ]
        state_dict["classifier.1.bias"] = state_dict["classifier.1.bias"][:num_classes]
        model.load_state_dict(state_dict, strict=False)
    return model


model_dir = "/opt/ias/model/pytorch/"

model_tab = {
    "mobilenet_v2": (create_mobilenet_v2, "mobilenet_v2-b0353104.pth"),
    "mobilenet_v3_large": (create_mobilenet_v2, "mobilenet_v3_large-8738ca79.pth"),
    "mobilenet_v3_small": (create_mobilenet_v2, "mobilenet_v3_small-047dcff4.pth"),
    "resnet18": (create_resnet18, "resnet18-5c106cde.pth"),
    "resnet34": (create_mobilenet_v2, "resnet34-333f7ec4.pth"),
    "resnet50": (create_mobilenet_v2, "resnet50-19c8e357.pth"),
    "resnet101": (create_mobilenet_v2, "resnet101-5d3b4d8f.pth"),
    "resnet152": (create_mobilenet_v2, "resnet152-b121ed2d.pth"),
    "resnext50_32x4d": (create_mobilenet_v2, "resnext50_32x4d-7cdf4587.pth"),
    "resnext101_32x8d": (create_mobilenet_v2, "resnext101_32x8d-8ba56ff5.pth"),
    "wide_resnet50_2": (create_mobilenet_v2, "wide_resnet50_2-95faca4d.pth"),
    "wide_resnet101_2": (create_mobilenet_v2, "wide_resnet101_2-32ee1156.pth"),
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

input_size = (244, 244)

train_trans = transforms.Compose(
    [
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # [0,255] -> [0.0, 1.0]
        normalize,
    ]
)

val_trans = transforms.Compose(
    [
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ]
)


def create(arch: str, num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
    """创建 Torch Image 分类模型.

    注意：
    - 尽量加载预训练参数，总比随机强
    - 每个模型必须要修改输出层。
    """
    ctor, file = model_tab[arch]

    file = model_dir + file
    state_dict = torch.load(file) if pretrained else None

    return ctor(num_classes, state_dict)



def show_keys(d: StateDict, name: str) -> None:
    from loguru import logger

    logger.info("name: {}", name)
    for k in d:
        logger.info("{}", k)


def load_pth_tar(num_classes: int, model_tar: Path) -> nn.Module:
    """加载 pth.tar 模型."""

    weights = torch.load(model_tar, weights_only=False)
    arch: str = weights["arch"]

    ctor, _file = model_tab[arch]
    model = ctor(num_classes)

    model = torch.nn.DataParallel(model)  # NOTE: 为state_dict.keys增加'module'前缀

    model.load_state_dict(weights["state_dict"])

    return model.module
