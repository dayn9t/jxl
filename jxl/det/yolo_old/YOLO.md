# YOLO

YOLO分裂成:

- YOLOv7, 训练时要注意于更新训练模型, 以便匹配的版本
    - 能裁减, 效果难料
- YOLOv8: https://github.com/ultralytics/ultralytics
    - 终于支持包了, 支持分类/分割, 接口提升很大
    - 支持

## 参考

- [数据目录](https://github.com/ultralytics/yolov5/releases/)
- [从零开始手把手教你利用yolov5训练自己的数据集(含coco128数据集/yolov5权重文件国内下载）](https://www.codenong.com/cs107099907/)
- [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [YOLO v5在医疗领域中消化内镜目标检测的应用](https://github.com/DataXujing/YOLO-v5)

## 疑难

### 尺寸控制

- 从datasets.py看，img_size是一维变量，所以检测器只能是方的。
  ```r = self.img_size / max(h0, w0)```

