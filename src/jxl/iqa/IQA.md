# 视频诊断

- 视频诊断
    - 视频模糊, 局部/全屏
    - 场景变化/摄像机转动, 视频上部
    - 摄像机偏色, 全屏
    - 摄像机亮度/环境照度异常, 全屏

- TODO
    - 非监督学习好! 方向!
    - 摄像机移动改良
        - 要改为监督学习, 因为那些中部遮挡更异常, 但必须被排除, 通过训练改变权重
        - 通过非监督学习筛选样本, 参见: OD.md
        - 场景变化标定工具, 如何重叠两图, 原图, 叠加边缘图
    - 摄像机移动革命
        - match算法太耗CPU
        - 直接用一个大的孪生网络: SiameseNetwok

## 参考

- [python判断照片偏色](https://zhuanlan.zhihu.com/p/57266919)
- [100行python实现摄像机偏移、抖动告警](https://www.cnblogs.com/xiaozhi_5638/p/9993232.html)
- [图像质量评价领域前沿综述（2022）](https://blog.csdn.net/qq_36306288/article/details/124016593)
    - [CVRKD-IQA](https://github.com/guanghaoyin/CVRKD-IQA)
    - [IQT (Transformer)](https://github.com/anse3832/IQT)
    - 主流IQA数据库
- [KADID-10k IQA 数据库](http://database.mmsp-kn.de/kadid-10k-database.html)
    - 失真类型, 模糊, 颜色失真, 压缩, 噪音, 白噪声, 椒盐噪声, 亮度变化, 空间扭曲, 清晰度和对比度
- [NTIRE @ CVPR 2021 视频质量增强竞赛：数据库、方法及结果汇总（官方发布）](https://zhuanlan.zhihu.com/p/368256419)
- [视频质量评价：挑战与机遇](https://zhuanlan.zhihu.com/p/384603663)
    - 概念介绍比比较全面

- [视频质量诊断和图像质量诊断 视频质量分析算法](https://blog.csdn.net/zhulong1984/article/details/106041011)
    - 传统图像处理方法
