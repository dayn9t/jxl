from mmdet.apis import init_detector, inference_detector


def main():
    config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
    checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
    inference_detector(model, '/home/jiang/ml/mm/mmdetection/demo/demo.jpg')


if __name__ == '__main__':
    main()
