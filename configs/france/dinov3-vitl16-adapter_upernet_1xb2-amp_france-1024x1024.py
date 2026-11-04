# DINOv3 ViT-L/16 (LVD-1689M) + DINOv3 adapter + UPerNet on France (IGN CoSIA), 14 classes, City + Village + Other (356.6 M
# parameters). Own model of the experiment.
# Weights: vitl16_adapter_upernet_france.pth, release v1.0 of github.com/tester170/france-segformer-b0. Trained on the 12
# training fragments of this dataset (best epoch 19: val mIoU 78.67 %), then fine-tuned on 84 additional IGN CoSIA fragments of
# other regions of France (96 fragments in all; the additional fragments are not part of the published dataset).
# Result of the released weights (tools/test.py, fp32): val mIoU 80.35 %, kappa 0.909, OA 92.63 %.
# This config reproduces the first stage (training on the 12 fragments) and evaluates the released weights.
# Model: the official DINOv3 ViT-L/16 inside the official DINOv3 adapter (facebookresearch/dinov3, commit 6876159, fetched into
# vendor/dinov3 by env/fetch_vendor.sh), wrapped as DINOv3ViTAdapter (france_mmseg/dinov3_vit_adapter.py): 4 maps of 1024
# channels (strides 4-32) for the same UPerHead 512 + FCN auxiliary head (weight 0.4) as the ConvNeXt-B config. The first 12 ViT
# blocks are frozen, the last 12 blocks, the adapter and the heads are trained; AdamW 1e-4 / wd 0.05, ViT blocks at 0.2 x base LR
# with layer decay 0.9 (the ViT-Adapter-L UPerNet rule), fp16, warm-up 1500 iterations. BN instead of SyncBN (1 GPU); after the
# warm-up the LR is controlled by the validation mIoU (PlateauControlHook).
_base_ = ['./_base_/datasets/france_cosia.py', './_base_/default_runtime.py']
custom_imports = dict(imports=['france_mmseg', 'france_mmseg.dinov3_vit_adapter'], allow_failed_imports=False)
dims = [1024, 1024, 1024, 1024]
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                         bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=(1024, 1024))
model = dict(
    type='StreamSlideEncoderDecoder',
    slide_batch=1,     # one window per forward pass: prediction identical to mmseg's slide inference
    data_preprocessor=data_preprocessor,
    pretrained=None,
    # checkpoint: official DINOv3 ViT-L/16 LVD-1689M weights from Meta (used for training only, see configs/france/README_RU.md)
    backbone=dict(type='DINOv3ViTAdapter', train_last=12, with_cp=False,
                  checkpoint='checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'),
    decode_head=dict(type='UPerHead', in_channels=dims, in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=512,
                     dropout_ratio=0.1, num_classes=14, norm_cfg=norm_cfg, align_corners=False,
                     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(type='FCNHead', in_channels=dims[2], in_index=2, channels=256, num_convs=1, concat_input=False,
                        dropout_ratio=0.1, num_classes=14, norm_cfg=norm_cfg, align_corners=False,
                        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='float16', loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='ViTTopLayerDecayConstructor',
    paramwise_cfg=dict(vit_prefix='backbone.adapter.backbone.', vit_lr_mult=0.2, decay_rate=0.9, num_blocks=24))
warmup_iters = 1500
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters)]

# "epoch" of the experiment = 4098 crops = 2049 iterations at batch 2, validation every epoch, at most 200 epochs; LR x0.5 after
# 5 validations without a gain of the validation-mIoU EMA (> 0.1 pp), stop after 12 such validations (the experiment stopped
# after epoch 22, best epoch 19). The final evaluation (tools/test.py) is fp32.
iters_per_epoch = 2049
train_dataloader = dict(batch_size=2)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=200 * iters_per_epoch, val_interval=iters_per_epoch)
val_cfg = dict(type='ValLoop', fp16=True)
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iters_per_epoch, max_keep_ckpts=1,
                    save_last=True, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
custom_hooks = [
    dict(type='SpeedOptionsHook', channels_last=True, tf32=False),
    dict(type='PlateauControlHook', monitor='mIoU', ema_alpha=0.5, min_delta=0.1, lr_patience=5, factor=0.5,
         cooldown=2, max_reductions=4, stop_patience=12, start_iter=warmup_iters),
]
