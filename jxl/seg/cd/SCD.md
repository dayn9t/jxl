# 场景变化检测

- [Awesome Remote Sensing Change Detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection)
    - 数据集, 论文, 代码特别全!
- [austin-ml-change-detection-demo](https://github.com/makepath/austin-ml-change-detection-demo)
    - 1 Hao Chen, Z., & Zhenwei Shi (2021)。使用 Transformer 进行遥感图像变化检测
    - 技术较新, 单需要与处理tif图片, 好像麻烦
- 修改: '/opt/ias/env/lib/python3.10/site-packages/change_detection_pytorch/encoders/__init__.py'

## 库

- [examples/siamese_network](https://github.com/pytorch/examples/blob/main/siamese_network/main.py)
    - ***Torch官方支持!!!***
    - 网络改造自resnet18, 变化好像不大
    - ***从forward函数看, 输出结果好像是分类, 而不是分割***

- [ChangeFormer](https://github.com/wgcban/ChangeFormer)
    - 基于Transformer, 可能最先进
    - 看表格好像精度最高
    - 依赖单纯
    - 一年没更新了, OpenCD也整合了这个库

- [Change Detection Laboratory](https://github.com/Bobholamovic/CDLab)

## 数据集

- [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
    - 遥感建筑变化检测数据集, 1024x1024, 637 个超高分辨率
    - 5 至 14 年的双时相图像具有显着的土地利用变化，尤其是建筑增长

- [S2Looking](https://paperswithcode.com/dataset/s2looking)
    - 建筑物变化检测数据集，侧视卫星图像。包含 5,000 全球农村双时相图像对(1024*1024) 和超过 65,920 个带注释的变化实例

- [AICDDataset](https://computervisiononline.com/dataset/1105138664)
    - 1000 对 800x600 图像
    - 下载失败: http://www.computervisiononline.com/files/dataset/AICDDataset.zip

## BUG

- Q: 'Attempted to set the storage of a tensor on device “cpu“ to a storage on different device “cuda:0“.'
    - A: 加载权重时出现权重位置与网络位置不匹配导致报错，与系统、pytorch版本有关
    - 参考: https://blog.csdn.net/m0_61927224/article/details/125770533

## Tensor

tensor(-2.1179) tensor(-2.1179)
tensor(-0.9153) tensor(-0.9153)
tensor(0.4265) tensor(0.4265)

tensor(0.0741) tensor(0.0741)
tensor(1.3256) tensor(1.3256)
tensor(2.6400) tensor(2.6400)

看不出规律
