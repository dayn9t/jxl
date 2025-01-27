norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DIEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='IA_ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        interaction_cfg=(None, {
            'type': 'SpatialExchange',
            'p': 0.5
        }, {
            'type': 'ChannelExchange',
            'p': 0.5
        }, {
            'type': 'ChannelExchange',
            'p': 0.5
        })),
    decode_head=dict(
        type='Changer',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'LEVIR_CD_Dataset'
data_root = 'data/LEVIR-CD'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=(512, 512)),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(
        type='MultiImgNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='MultiImgResize', keep_ratio=True),
            dict(type='MultiImgRandomFlip'),
            dict(
                type='MultiImgNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='MultiImgImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='LEVIR_CD_Dataset',
        data_root='data/LEVIR-CD',
        img_dir='train',
        ann_dir='train/label',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
            dict(type='MultiImgRandomCrop', crop_size=(512, 512)),
            dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
            dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
            dict(type='MultiImgExchangeTime', prob=0.5),
            dict(
                type='MultiImgPhotoMetricDistortion',
                brightness_delta=10,
                contrast_range=(0.8, 1.2),
                saturation_range=(0.8, 1.2),
                hue_delta=10),
            dict(
                type='MultiImgNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='MultiImgDefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='LEVIR_CD_Dataset',
        data_root='data/LEVIR-CD',
        img_dir='val',
        ann_dir='val/label',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(
                type='MultiImgMultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='MultiImgResize', keep_ratio=True),
                    dict(type='MultiImgRandomFlip'),
                    dict(
                        type='MultiImgNormalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='MultiImgImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LEVIR_CD_Dataset',
        data_root='data/LEVIR-CD',
        img_dir='test',
        ann_dir='test/label',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(
                type='MultiImgMultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='MultiImgResize', keep_ratio=True),
                    dict(type='MultiImgRandomFlip'),
                    dict(
                        type='MultiImgNormalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='MultiImgImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='AdamW', lr=0.005, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(
    interval=4000,
    metric=['mFscore', 'mIoU'],
    pre_eval=True,
    save_best='Fscore.changed',
    greater_keys=['Fscore'])
work_dir = './changer_r18_levir_workdir'
gpu_ids = [0]
auto_resume = False
