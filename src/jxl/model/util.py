from dataclasses import dataclass
from pathlib import Path

import torch
from torchsummary import summary  # type: ignore


@dataclass(frozen=True)
class ConfFile1:
    """分类器器选项"""

    conf: float
    """置信度"""
    file: Path
    """文件路径"""


def show_model(model: Path, opt):
    """数据集测试"""

    shape = (3, 224, 224)

    with torch.no_grad():  # 不计算导数
        model = torch.load(model)

        model = model.cuda()
        model.eval()  # 固定dropout/归一化层，否则每次推理结果不同

        summary(model, shape)
        print(model)
