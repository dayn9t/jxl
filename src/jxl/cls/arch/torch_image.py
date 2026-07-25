from pathlib import Path
from typing import cast

import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger
from torchvision.models import MobileNetV2


def create_resnet18(
    num_classes: int,
    state_dict: dict[str, torch.Tensor] | None = None,
    **kwargs,
) -> torch.nn.Module:
    model = torchvision.models.resnet18(num_classes=num_classes, **kwargs)
    if state_dict:
        state_dict["fc.weight"] = state_dict["fc.weight"][:num_classes, :]
        state_dict["fc.bias"] = state_dict["fc.bias"][:num_classes]
        model.load_state_dict(state_dict, strict=False)
    return cast(torch.nn.Module, model)


def create_mobilenet_v2(
    num_classes: int,
    state_dict: dict[str, torch.Tensor] | None = None,
    **kwargs,
) -> torch.nn.Module:
    model = MobileNetV2(num_classes=num_classes, **kwargs)
    if state_dict:
        state_dict["classifier.1.weight"] = state_dict["classifier.1.weight"][
            :num_classes, :
        ]
        state_dict["classifier.1.bias"] = state_dict["classifier.1.bias"][:num_classes]
        model.load_state_dict(state_dict, strict=False)
    return cast(torch.nn.Module, model)


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


def create(
    arch: str,
    num_classes: int = 1000,
    pretrained: bool = True,
    **kwargs,
) -> torch.nn.Module:
    """创建 Torch Image 分类模型.

    注意：
    - 尽量加载预训练参数，总比随机强
    - 每个模型必须要修改输出层。
    """
    ctor, file = model_tab[arch]

    file = model_dir + file
    state_dict = torch.load(file) if pretrained else None

    return ctor(num_classes, state_dict, **kwargs)



def show_keys(d: dict[str, object], name: str) -> None:
    logger.info(f"\nname: {name}")
    for k in d:
        logger.info(f"{k}")
    logger.info("\n")


def load_pth_tar(num_classes: int, model_tar: Path) -> torch.nn.Module:
    """加载 pth.tar 模型."""

    weights = torch.load(model_tar)
    arch = weights["arch"]

    # print('type:', type(weights['state_dict']))
    # show_keys(weights['state_dict'], 'full:')

    ctor, _file = model_tab[arch]
    model = ctor(num_classes)
    # show_keys(model.state_dict(), 'empty:')
    # print('empty:', model)

    model = torch.nn.DataParallel(model)  # NOTE: 为state_dict.keys增加'module'前缀
    # show_keys(model.state_dict(), 'parallel:')
    # print('parallel:', model)

    model.load_state_dict(weights["state_dict"])
    # show_keys(model.state_dict(), 'full:')
    # print('full:', model)

    return model.module
