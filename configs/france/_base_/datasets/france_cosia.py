# France (orthophotos ~0.2 m, IGN CoSIA labels), 14 classes, in the layout written by scripts/prepare_data.py:
#   data/france/img_dir/train/<fragment>_y<row>_x<col>.png  2048 x 2048 tiles of the 12 training fragments (BGR PNG)
#   data/france/ann_dir/train/<fragment>_y<row>_x<col>.png  train IDs 0..13 (CoSIA ids 0-5, 7-14 in order)
#   data/france/img_dir/val/<fragment>.png, ann_dir/val/<fragment>.png   the 4 validation fragments whole
# Evaluation as in the report: whole validation fragments at native resolution, sliding window 1024 / stride 768
# (model test_cfg, streamed by StreamSlideEncoderDecoder), mIoU over the 14 classes.
dataset_type = 'BaseSegDataset'
data_root = 'data/france'
metainfo = dict(
    classes=('building', 'permeable surface', 'impervious surface', 'swimming pool', 'bare soil', 'water',
             'coniferous', 'deciduous', 'brushwood', 'vineyard', 'herbaceous', 'agricultural', 'plowed land',
             'greenhouse'),
    palette=[[206, 112, 121], [152, 119, 82], [166, 170, 183], [98, 208, 255], [187, 176, 150], [51, 117, 161],
             [18, 100, 33], [76, 145, 41], [181, 195, 53], [176, 130, 144], [140, 215, 106], [222, 207, 85],
             [208, 163, 73], [185, 226, 212]])
crop_size = (1024, 1024)
# Training augmentation of the experiment with standard MMSeg transforms: random scale 0.75-1.5 of the native
# resolution, 1024 crop, mild photometric jitter (softer than the MMSeg defaults), the 8 dihedral views. The
# experiment sampled the crops from the whole fragments (every grid cell of the training fragments once per epoch
# + 25 % crops anchored on rare classes); here crops are uniform random within 2048 x 2048 tiles.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(2048, 2048), ratio_range=(0.75, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2),
         hue_delta=5),
    dict(type='RandomDihedral'),                     # france_mmseg: rot90^k and optional flip (aerial nadir views)
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo, reduce_zero_label=False,
                 img_suffix='.png', seg_map_suffix='.png',
                 data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo, reduce_zero_label=False,
                 img_suffix='.png', seg_map_suffix='.png',
                 data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                 pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator
