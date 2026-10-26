# DINOv3 ConvNeXt-B (LVD-1689M) + UPerNet on France (IGN CoSIA), 14 classes, City + Village + Other (122 M parameters).
# Result (report, best epoch 43 of 2049 iterations = iteration 88107): val mIoU 75.30 %, kappa 0.880, OA 90.31 %.
# Weights: convnext_b_upernet_france_ep43.pth, release v1.0 of github.com/tester170/france-segformer-b0.
# Trained from the DINOv3 pretrained encoder, not from the model trained on DLR SkyScapes. Model and optimiser follow
# MMSeg's UPerNet-ConvNeXt recipe (configs/convnext/convnext-base_upernet_8xb2-amp-160k_ade20k-512x512.py): drop path
# 0.4, UPerHead 512 ch, FCN auxiliary head on stage 3 (weight 0.4), AdamW 1e-4 / wd 0.05, stage-wise LR decay 0.9 over 12
# layers, fp16, warm-up 1500 iterations. The backbone is the official DINOv3 ConvNeXt (facebookresearch/dinov3, commit
# 6876159, fetched into vendor/dinov3 by env/fetch_vendor.sh) wrapped as DINOv3ConvNeXt (france_mmseg/dinov3_backbones.py):
# the 4 stage maps, each through a LayerNorm (stage 4: DINOv3's own final norm). Training starts from DINOv3 weights
# converted to the official format (scripts/prepare_dinov3_weights.py); BN instead of SyncBN (1 GPU); after the warm-up
# the LR is controlled by the validation mIoU (PlateauControlHook).
_base_ = ['./_base_/datasets/france_cosia.py', './_base_/default_runtime.py']
custom_imports = dict(imports=['france_mmseg', 'france_mmseg.dinov3_backbones'], allow_failed_imports=False)
dims = [128, 256, 512, 1024]
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                         bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=(1024, 1024))
model = dict(
    type='StreamSlideEncoderDecoder',
    slide_batch=1,     # one window per forward pass: prediction identical to mmseg's slide inference
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(type='DINOv3ConvNeXt', arch='base', drop_path_rate=0.4,
                  checkpoint='checkpoints/dinov3_convnext_base_lvd1689m_official.pth'),   # used for training only
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
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12))
warmup_iters = 1500
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters)]

# "epoch" of the experiment = 4098 crops = 2049 iterations at batch 2 (the recipe's 2 images per GPU); validation every
# epoch (fp16 autocast: same mIoU as fp32 at the best epoch), at most 200 epochs; LR x0.5 after 5 validations without a
# gain of the validation-mIoU EMA (> 0.1 pp), stop after 12 such validations. The final evaluation (tools/test.py) is fp32.
iters_per_epoch = 2049
train_dataloader = dict(batch_size=2)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=200 * iters_per_epoch, val_interval=iters_per_epoch)
val_cfg = dict(type='ValLoop', fp16=True)
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2 * iters_per_epoch, max_keep_ckpts=1,
                    save_last=True, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
custom_hooks = [
    # channels_last memory format: 0.43 vs 0.52 s per iteration on an RTX 3090, same training losses
    dict(type='SpeedOptionsHook', channels_last=True, tf32=False),
    dict(type='PlateauControlHook', monitor='mIoU', ema_alpha=0.5, min_delta=0.1, lr_patience=5, factor=0.5,
         cooldown=2, max_reductions=4, stop_patience=12, start_iter=warmup_iters),
]
