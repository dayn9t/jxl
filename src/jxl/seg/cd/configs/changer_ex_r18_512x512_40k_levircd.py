norm_cfg = {"type": "SyncBN", "requires_grad": True}
model = {
    "type": "DIEncoderDecoder",
    "pretrained": None,
    "backbone": {
        "type": "IA_ResNetV1c",
        "depth": 18,
        "num_stages": 4,
        "out_indices": (0, 1, 2, 3),
        "dilations": (1, 1, 1, 1),
        "strides": (1, 2, 2, 2),
        "norm_cfg": {"type": "SyncBN", "requires_grad": True},
        "norm_eval": False,
        "style": "pytorch",
        "contract_dilation": True,
        "interaction_cfg": (
            None,
            {"type": "SpatialExchange", "p": 0.5},
            {"type": "ChannelExchange", "p": 0.5},
            {"type": "ChannelExchange", "p": 0.5},
        ),
    },
    "decode_head": {
        "type": "Changer",
        "in_channels": [64, 128, 256, 512],
        "in_index": [0, 1, 2, 3],
        "channels": 128,
        "dropout_ratio": 0.1,
        "num_classes": 2,
        "norm_cfg": {"type": "SyncBN", "requires_grad": True},
        "align_corners": False,
        "loss_decode": {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 1.0},
        "sampler": {"type": "OHEMPixelSampler", "thresh": 0.7, "min_kept": 100000},
    },
    "train_cfg": {},
    "test_cfg": {"mode": "whole"},
}
dataset_type = "LEVIR_CD_Dataset"
data_root = "data/LEVIR-CD"
img_norm_cfg = {
    "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "to_rgb": True
}
crop_size = (512, 512)
train_pipeline = [
    {"type": "MultiImgLoadImageFromFile"},
    {"type": "MultiImgLoadAnnotations"},
    {"type": "MultiImgRandomRotate", "prob": 0.5, "degree": 180},
    {"type": "MultiImgRandomCrop", "crop_size": (512, 512)},
    {"type": "MultiImgRandomFlip", "prob": 0.5, "direction": "horizontal"},
    {"type": "MultiImgRandomFlip", "prob": 0.5, "direction": "vertical"},
    {"type": "MultiImgExchangeTime", "prob": 0.5},
    {
        "type": "MultiImgPhotoMetricDistortion",
        "brightness_delta": 10,
        "contrast_range": (0.8, 1.2),
        "saturation_range": (0.8, 1.2),
        "hue_delta": 10,
    },
    {
        "type": "MultiImgNormalize",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "to_rgb": True,
    },
    {"type": "MultiImgDefaultFormatBundle"},
    {"type": "Collect", "keys": ["img", "gt_semantic_seg"]},
]
test_pipeline = [
    {"type": "MultiImgLoadImageFromFile"},
    {
        "type": "MultiImgMultiScaleFlipAug",
        "img_scale": (1024, 1024),
        "flip": False,
        "transforms": [
            {"type": "MultiImgResize", "keep_ratio": True},
            {"type": "MultiImgRandomFlip"},
            {
                "type": "MultiImgNormalize",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": True,
            },
            {"type": "MultiImgImageToTensor", "keys": ["img"]},
            {"type": "Collect", "keys": ["img"]},
        ],
    },
]
data = {
    "samples_per_gpu": 8,
    "workers_per_gpu": 4,
    "train": {
        "type": "LEVIR_CD_Dataset",
        "data_root": "data/LEVIR-CD",
        "img_dir": "train",
        "ann_dir": "train/label",
        "pipeline": [
            {"type": "MultiImgLoadImageFromFile"},
            {"type": "MultiImgLoadAnnotations"},
            {"type": "MultiImgRandomRotate", "prob": 0.5, "degree": 180},
            {"type": "MultiImgRandomCrop", "crop_size": (512, 512)},
            {"type": "MultiImgRandomFlip", "prob": 0.5, "direction": "horizontal"},
            {"type": "MultiImgRandomFlip", "prob": 0.5, "direction": "vertical"},
            {"type": "MultiImgExchangeTime", "prob": 0.5},
            {
                "type": "MultiImgPhotoMetricDistortion",
                "brightness_delta": 10,
                "contrast_range": (0.8, 1.2),
                "saturation_range": (0.8, 1.2),
                "hue_delta": 10,
            },
            {
                "type": "MultiImgNormalize",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": True,
            },
            {"type": "MultiImgDefaultFormatBundle"},
            {"type": "Collect", "keys": ["img", "gt_semantic_seg"]},
        ],
    },
    "val": {
        "type": "LEVIR_CD_Dataset",
        "data_root": "data/LEVIR-CD",
        "img_dir": "val",
        "ann_dir": "val/label",
        "pipeline": [
            {"type": "MultiImgLoadImageFromFile"},
            {
                "type": "MultiImgMultiScaleFlipAug",
                "img_scale": (1024, 1024),
                "flip": False,
                "transforms": [
                    {"type": "MultiImgResize", "keep_ratio": True},
                    {"type": "MultiImgRandomFlip"},
                    {
                        "type": "MultiImgNormalize",
                        "mean": [123.675, 116.28, 103.53],
                        "std": [58.395, 57.12, 57.375],
                        "to_rgb": True,
                    },
                    {"type": "MultiImgImageToTensor", "keys": ["img"]},
                    {"type": "Collect", "keys": ["img"]},
                ],
            },
        ],
    },
    "test": {
        "type": "LEVIR_CD_Dataset",
        "data_root": "data/LEVIR-CD",
        "img_dir": "test",
        "ann_dir": "test/label",
        "pipeline": [
            {"type": "MultiImgLoadImageFromFile"},
            {
                "type": "MultiImgMultiScaleFlipAug",
                "img_scale": (1024, 1024),
                "flip": False,
                "transforms": [
                    {"type": "MultiImgResize", "keep_ratio": True},
                    {"type": "MultiImgRandomFlip"},
                    {
                        "type": "MultiImgNormalize",
                        "mean": [123.675, 116.28, 103.53],
                        "std": [58.395, 57.12, 57.375],
                        "to_rgb": True,
                    },
                    {"type": "MultiImgImageToTensor", "keys": ["img"]},
                    {"type": "Collect", "keys": ["img"]},
                ],
            },
        ],
    },
}
log_config = {"interval": 50, "hooks": [{"type": "TextLoggerHook", "by_epoch": False}]}
dist_params = {"backend": "nccl"}
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True
optimizer = {"type": "AdamW", "lr": 0.005, "betas": (0.9, 0.999), "weight_decay": 0.05}
optimizer_config = {}
lr_config = {
    "policy": "poly",
    "warmup": "linear",
    "warmup_iters": 1500,
    "warmup_ratio": 1e-06,
    "power": 1.0,
    "min_lr": 0.0,
    "by_epoch": False,
}
runner = {"type": "IterBasedRunner", "max_iters": 40000}
checkpoint_config = {"by_epoch": False, "interval": 4000}
evaluation = {
    "interval": 4000,
    "metric": ["mFscore", "mIoU"],
    "pre_eval": True,
    "save_best": "Fscore.changed",
    "greater_keys": ["Fscore"],
}
work_dir = "./changer_r18_levir_workdir"
gpu_ids = [0]
auto_resume = False
