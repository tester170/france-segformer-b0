"""DINOv3 ConvNeXt backbone for MMSeg 1.2.2 built from the original facebookresearch/dinov3 code (vendor/dinov3, commit
6876159), exactly as in the DLR SkyScapes experiment (skyscapes_mmseg/dinov3_backbones.py, DINOv3ConvNeXt only).

DINOv3ConvNeXt = the official ConvNeXt (dinov3/models/convnext.py) used as in MMSeg's UPerNet-ConvNeXt recipe
(configs/convnext/convnext-base_upernet_*: mmpretrain ConvNeXt, out_indices=[0, 1, 2, 3], gap_before_final_norm=False):
the 4 stage maps (strides 4, 8, 16, 32), each through a LayerNorm. The stage-4 norm is DINOv3's pretrained final norm,
stages 1-3 get new ones. Parameter names stay downsample_layers.* / stages.*, which is what MMSeg's
LearningRateDecayOptimizerConstructor (decay_type='stage_wise') parses."""
import sys
from pathlib import Path

import torch
import torch._dynamo.config as _dynamo_cfg
from mmengine.logging import print_log
from mmseg.registry import MODELS
from torch import nn

VENDOR = Path(__file__).resolve().parents[1] / 'vendor' / 'dinov3'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
# dinov3/layers/block.py sets torch._dynamo.config.accumulated_cache_size_limit, a torch.compile cache knob that torch
# 2.1 does not have; torch.compile is not used, so the key is only allowed.
_dynamo_cfg._allowed_keys.add('accumulated_cache_size_limit')
from dinov3.models.convnext import ConvNeXt, LayerNorm, convnext_sizes  # noqa: E402


@MODELS.register_module()
class DINOv3ConvNeXt(ConvNeXt):

    def __init__(self, arch='base', drop_path_rate=0.4, checkpoint=None):
        size = convnext_sizes[arch]
        super().__init__(depths=size['depths'], dims=size['dims'], drop_path_rate=drop_path_rate)
        self.out_norms = nn.ModuleList([LayerNorm(d, eps=1e-6, data_format='channels_first') for d in size['dims'][:3]])
        for n in self.out_norms:
            n.init_weights()
        self.checkpoint = checkpoint

    def init_weights(self):
        # called by MMEngine when the segmentor is built for training; a resumed/loaded checkpoint overrides it anyway
        if not self.checkpoint:
            return super().init_weights()
        sd = torch.load(self.checkpoint, map_location='cpu')['state_dict']
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith('out_norms.') for k in missing), (missing, unexpected)
        print_log(f'DINOv3ConvNeXt: {len(sd)} pretrained tensors from {self.checkpoint}; new: {len(missing)} (out_norms)',
                  logger='current')

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
            outs.append(self.out_norms[i](x) if i < 3 else self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return tuple(outs)
