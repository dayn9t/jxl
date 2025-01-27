import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)

        return x


model = ResNet18()

'''
Q: 帮我编程修改resnet18的输入层尺寸为
A: 改了模型的第一层，将其卷积核大小设置为7x7，步长设置为2，填充设置为3，并将其输入通道数从3改为64。
最后，我们将模型的平均池化层替换为自适应平均池化层，并将输出大小设置为1x1。
'''
