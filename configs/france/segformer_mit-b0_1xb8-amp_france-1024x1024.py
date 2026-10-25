# SegFormer-B0 (MiT-B0 + all-MLP head, 3.7 M parameters) on France (IGN CoSIA), 14 classes, City + Village + Other.
# Result (report, best epoch 76 of 513 iterations = iteration 38988): val mIoU 72.90 %, kappa 0.860, OA 88.62 %.
# Weights: segformer_b0_france_ep76.pth, release v1.0 of github.com/tester170/france-segformer-b0.
# Recipe of the official MMSeg SegFormer Cityscapes 1024x1024 config (AdamW 6e-5, head LR x10, warm-up 1500 iterations),
# encoder initialised from LoveDA (mit_b0_loveda_encoder.pth, same release), fp16 mixed precision, batch 8. After the
# warm-up the LR is controlled by the validation mIoU instead of a fixed poly schedule (PlateauControlHook).
_base_ = ['./_base_/datasets/france_cosia.py', './_base_/default_runtime.py']
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                         bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=(1024, 1024))
model = dict(
    type='StreamSlideEncoderDecoder',
    slide_batch=1,     # one window per forward pass: prediction identical to mmseg's slide inference
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(type='MixVisionTransformer', in_channels=3, embed_dims=32, num_stages=4, num_layers=[2, 2, 2, 2],
                  num_heads=[1, 2, 5, 8], patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3),
                  mlp_ratio=4, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, init_cfg=None),
    decode_head=dict(type='SegformerHead', in_channels=[32, 64, 160, 256], in_index=[0, 1, 2, 3], channels=256,
                     dropout_ratio=0.1, num_classes=14, norm_cfg=norm_cfg, align_corners=False,
                     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
load_from = 'checkpoints/mit_b0_loveda_encoder.pth'    # LoveDA-pretrained encoder (training); test.py takes the trained weights

optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='float16', loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.0), 'norm': dict(decay_mult=0.0),
                                    'head': dict(lr_mult=10.0)}))
warmup_iters = 1500
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters)]

# "epoch" of the experiment = 4098 crops = 513 iterations at batch 8; validation every 2 epochs, at most 100 epochs;
# LR x0.5 after 5 validations without a gain of the validation-mIoU EMA (> 0.1 pp), stop after 12 such validations.
iters_per_epoch = 513
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100 * iters_per_epoch, val_interval=2 * iters_per_epoch)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4 * iters_per_epoch, max_keep_ckpts=1,
                    save_last=True, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
custom_hooks = [
    dict(type='PlateauControlHook', monitor='mIoU', ema_alpha=0.5, min_delta=0.1, lr_patience=5, factor=0.5,
         cooldown=2, max_reductions=4, stop_patience=12, start_iter=warmup_iters),
]
