from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torchsummary import summary


@dataclass(frozen=True)
class ConfFile1:
    """分类器器选项"""

    conf: float
    """置信度"""
    file: Path
    """文件路径"""


def show_model(model_file: Path, _opt: object) -> None:
    """数据集测试"""

    shape = (3, 224, 224)

    with torch.no_grad():  # 不计算导数
        model: nn.Module = torch.load(model_file, weights_only=False)

        model = model.cuda()
        model.eval()  # 固定dropout/归一化层，否则每次推理结果不同

        summary(model, shape)
        logger.info("{}", model)
