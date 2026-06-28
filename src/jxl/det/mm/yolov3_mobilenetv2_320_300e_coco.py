checkpoint_config = {"interval": 1}
log_config = {"interval": 50, "hooks": [{"type": "TextLoggerHook"}]}
custom_hooks = [{"type": "NumClassCheckHook"}]
dist_params = {"backend": "nccl"}
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
auto_scale_lr = {"enable": False, "base_batch_size": 192}
model = {
    "type": "YOLOV3",
    "backbone": {
        "type": "MobileNetV2",
        "out_indices": (2, 4, 6),
        "act_cfg": {"type": "LeakyReLU", "negative_slope": 0.1},
        "init_cfg": {"type": "Pretrained", "checkpoint": "open-mmlab://mmdet/mobilenet_v2"},
    },
    "neck": {
        "type": "YOLOV3Neck",
        "num_scales": 3,
        "in_channels": [320, 96, 32],
        "out_channels": [96, 96, 96],
    },
    "bbox_head": {
        "type": "YOLOV3Head",
        "num_classes": 80,
        "in_channels": [96, 96, 96],
        "out_channels": [96, 96, 96],
        "anchor_generator": {
            "type": "YOLOAnchorGenerator",
            "base_sizes": [
                [(220, 125), (128, 222), (264, 266)],
                [(35, 87), (102, 96), (60, 170)],
                [(10, 15), (24, 36), (72, 42)],
            ],
            "strides": [32, 16, 8],
        },
        "bbox_coder": {"type": "YOLOBBoxCoder"},
        "featmap_strides": [32, 16, 8],
        "loss_cls": {
            "type": "CrossEntropyLoss", "use_sigmoid": True, "loss_weight": 1.0, "reduction": "sum"
        },
        "loss_conf": {
            "type": "CrossEntropyLoss", "use_sigmoid": True, "loss_weight": 1.0, "reduction": "sum"
        },
        "loss_xy": {
            "type": "CrossEntropyLoss", "use_sigmoid": True, "loss_weight": 2.0, "reduction": "sum"
        },
        "loss_wh": {"type": "MSELoss", "loss_weight": 2.0, "reduction": "sum"},
    },
    "train_cfg": {
        "assigner": {
            "type": "GridAssigner", "pos_iou_thr": 0.5, "neg_iou_thr": 0.5, "min_pos_iou": 0
        }
    },
    "test_cfg": {
        "nms_pre": 1000,
        "min_bbox_size": 0,
        "score_thr": 0.05,
        "conf_thr": 0.005,
        "nms": {"type": "nms", "iou_threshold": 0.45},
        "max_per_img": 100,
    },
}
dataset_type = "CocoDataset"
data_root = "data/coco/"
img_norm_cfg = {
    "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "to_rgb": True
}
train_pipeline = [
    {"type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": True},
    {
        "type": "Expand", "mean": [123.675, 116.28, 103.53], "to_rgb": True, "ratio_range": (1, 2)
    },
    {
        "type": "MinIoURandomCrop",
        "min_ious": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "min_crop_size": 0.3,
    },
    {"type": "Resize", "img_scale": (320, 320), "keep_ratio": True},
    {"type": "RandomFlip", "flip_ratio": 0.5},
    {"type": "PhotoMetricDistortion"},
    {
        "type": "Normalize",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "to_rgb": True,
    },
    {"type": "Pad", "size_divisor": 32},
    {"type": "DefaultFormatBundle"},
    {"type": "Collect", "keys": ["img", "gt_bboxes", "gt_labels"]},
]
test_pipeline = [
    {"type": "LoadImageFromFile"},
    {
        "type": "MultiScaleFlipAug",
        "img_scale": (320, 320),
        "flip": False,
        "transforms": [
            {"type": "Resize", "keep_ratio": True},
            {"type": "RandomFlip"},
            {
                "type": "Normalize",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": True,
            },
            {"type": "Pad", "size_divisor": 32},
            {"type": "DefaultFormatBundle"},
            {"type": "Collect", "keys": ["img"]},
        ],
    },
]
data = {
    "samples_per_gpu": 24,
    "workers_per_gpu": 4,
    "train": {
        "type": "RepeatDataset",
        "times": 10,
        "dataset": {
            "type": "CocoDataset",
            "ann_file": "data/coco/annotations/instances_train2017.json",
            "img_prefix": "data/coco/train2017/",
            "pipeline": [
                {"type": "LoadImageFromFile"},
                {"type": "LoadAnnotations", "with_bbox": True},
                {
                    "type": "Expand",
                    "mean": [123.675, 116.28, 103.53],
                    "to_rgb": True,
                    "ratio_range": (1, 2),
                },
                {
                    "type": "MinIoURandomCrop",
                    "min_ious": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    "min_crop_size": 0.3,
                },
                {"type": "Resize", "img_scale": (320, 320), "keep_ratio": True},
                {"type": "RandomFlip", "flip_ratio": 0.5},
                {"type": "PhotoMetricDistortion"},
                {
                    "type": "Normalize",
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                    "to_rgb": True,
                },
                {"type": "Pad", "size_divisor": 32},
                {"type": "DefaultFormatBundle"},
                {"type": "Collect", "keys": ["img", "gt_bboxes", "gt_labels"]},
            ],
        },
    },
    "val": {
        "type": "CocoDataset",
        "ann_file": "data/coco/annotations/instances_val2017.json",
        "img_prefix": "data/coco/val2017/",
        "pipeline": [
            {"type": "LoadImageFromFile"},
            {
                "type": "MultiScaleFlipAug",
                "img_scale": (320, 320),
                "flip": False,
                "transforms": [
                    {"type": "Resize", "keep_ratio": True},
                    {"type": "RandomFlip"},
                    {
                        "type": "Normalize",
                        "mean": [123.675, 116.28, 103.53],
                        "std": [58.395, 57.12, 57.375],
                        "to_rgb": True,
                    },
                    {"type": "Pad", "size_divisor": 32},
                    {"type": "DefaultFormatBundle"},
                    {"type": "Collect", "keys": ["img"]},
                ],
            },
        ],
    },
    "test": {
        "type": "CocoDataset",
        "ann_file": "data/coco/annotations/instances_val2017.json",
        "img_prefix": "data/coco/val2017/",
        "pipeline": [
            {"type": "LoadImageFromFile"},
            {
                "type": "MultiScaleFlipAug",
                "img_scale": (320, 320),
                "flip": False,
                "transforms": [
                    {"type": "Resize", "keep_ratio": True},
                    {"type": "RandomFlip"},
                    {
                        "type": "Normalize",
                        "mean": [123.675, 116.28, 103.53],
                        "std": [58.395, 57.12, 57.375],
                        "to_rgb": True,
                    },
                    {"type": "Pad", "size_divisor": 32},
                    {"type": "DefaultFormatBundle"},
                    {"type": "Collect", "keys": ["img"]},
                ],
            },
        ],
    },
}
optimizer = {"type": "SGD", "lr": 0.003, "momentum": 0.9, "weight_decay": 0.0005}
optimizer_config = {"grad_clip": {"max_norm": 35, "norm_type": 2}}
lr_config = {
    "policy": "step",
    "warmup": "linear",
    "warmup_iters": 4000,
    "warmup_ratio": 0.0001,
    "step": [24, 28],
}
runner = {"type": "EpochBasedRunner", "max_epochs": 30}
evaluation = {"interval": 1, "metric": ["bbox"]}
find_unused_parameters = True
