"""DINOv3 ViT-L/16 + the official DINOv3 adapter as an MMSeg 1.2.2 backbone, built from the original facebookresearch/dinov3
code (vendor/dinov3, commit 6876159, env/fetch_vendor.sh), exactly as in the experiment of the own France model.

DINOv3ViTAdapter = the official ViT-L/16 (dinov3.hub.backbones.dinov3_vitl16, LVD-1689M weights) inside the official
DINOv3_Adapter (dinov3/eval/segmentation/models/backbone/dinov3_adapter.py: ViT-Adapter, Z. Chen et al., ICLR 2023, in Meta's
version for DINOv3 segmentation: convolutional spatial prior module with strides 4/8/16/32, deformable-attention interaction with
ViT layers 4, 11, 17, 23, BatchNorm on the outputs). Output: 4 maps of 1024 channels (strides 4, 8, 16, 32) for UPerHead.

train_last = K: the last K ViT blocks and the final LayerNorm are trained, the first 24 - K blocks stay frozen (the model was trained
with K = 12). The adapter runs the ViT under torch.no_grad; for K > 0 its get_intermediate_layers is replaced by a copy of the
official method that runs the trained blocks and the norm under torch.enable_grad. The ViT is always in eval mode (no drop path)
and runs in bf16 (autocast inside the official adapter).

ViTTopLayerDecayConstructor: parameter groups of the training: heads and adapter at the base LR, ViT block i at
base LR x vit_lr_mult x decay_rate^(num_blocks - 1 - i) (the ViT-Adapter-L UPerNet rule), no weight decay for 1-D parameters,
biases and tokens.

Compatibility of the official code with torch 2.1 / mmcv 2.1 (installed on import, before the DINOv3 modules are loaded):
  1. torch.amp.custom_fwd / custom_bwd(device_type=...) appeared in torch 2.4; in 2.1 they are torch.cuda.amp.custom_fwd / bwd.
  2. The backward pass of the official MSDeformAttnFunction calls the compiled Deformable-DETR kernel
     MultiScaleDeformableAttention.ms_deform_attn_backward; the same kernel is built in mmcv 2.1 (mmcv.ops.multi_scale_deform_attn)
     and is exposed under that name with Deformable-DETR's signature.
  3. dinov3/eval/segmentation/models/__init__.py imports the evaluation builder (torchmetrics); the adapter needs only
     models/backbone and models/utils, so the package is registered as a plain namespace with the same path.
  4. On the GPU the forward pass of MSDeformAttnFunction uses mmcv's CUDA kernel (same algorithm as Meta's PyTorch version,
     which their code marks "for debug and test only"); on the CPU Meta's PyTorch version is kept."""
import sys
import types
from pathlib import Path

import torch
import torch._dynamo.config as _dynamo_cfg
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.optim import DefaultOptimWrapperConstructor
from mmseg.registry import MODELS, OPTIM_WRAPPER_CONSTRUCTORS

VENDOR = Path(__file__).resolve().parents[1] / 'vendor' / 'dinov3'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
# dinov3/layers/block.py sets a torch.compile cache key that torch 2.1 does not have; torch.compile is not used.
_dynamo_cfg._allowed_keys.add('accumulated_cache_size_limit')


def _install_compat():
    import torch.amp as _amp
    if not hasattr(_amp, 'custom_fwd'):
        def custom_fwd(fwd=None, *, device_type='cuda', cast_inputs=None):
            if fwd is not None:
                return torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs)
            return torch.cuda.amp.custom_fwd(cast_inputs=cast_inputs)

        def custom_bwd(bwd=None, *, device_type='cuda'):
            return torch.cuda.amp.custom_bwd(bwd) if bwd is not None else torch.cuda.amp.custom_bwd
        _amp.custom_fwd, _amp.custom_bwd = custom_fwd, custom_bwd
    if 'MultiScaleDeformableAttention' not in sys.modules:
        from mmcv.utils import ext_loader
        ext = ext_loader.load_ext('_ext', ['ms_deform_attn_backward'])
        m = types.ModuleType('MultiScaleDeformableAttention')

        def ms_deform_attn_backward(value, shapes, level_start, loc, attn, grad_output, im2col_step):
            gv, gl, ga = torch.zeros_like(value), torch.zeros_like(loc), torch.zeros_like(attn)
            ext.ms_deform_attn_backward(value, shapes, level_start, loc, attn, grad_output.contiguous(), gv, gl, ga,
                                        im2col_step=im2col_step)
            return gv, gl, ga
        m.ms_deform_attn_backward = ms_deform_attn_backward
        sys.modules['MultiScaleDeformableAttention'] = m
    name = 'dinov3.eval.segmentation.models'
    if name not in sys.modules:
        import dinov3.eval.segmentation as _seg
        m = types.ModuleType(name)
        m.__path__ = [str(Path(_seg.__file__).parent / 'models')]
        sys.modules[name] = m
    from dinov3.eval.segmentation.models.utils import ms_deform_attn as M
    if not getattr(M.MSDeformAttnFunction, '_france_cuda', False):
        from mmcv.utils import ext_loader
        ext_f = ext_loader.load_ext('_ext', ['ms_deform_attn_forward'])

        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, value, shapes, level_start, loc, attn, im2col_step):
            ctx.im2col_step = im2col_step
            if value.is_cuda:
                out = ext_f.ms_deform_attn_forward(value, shapes, level_start, loc, attn, im2col_step=im2col_step)
            else:
                out = M.ms_deform_attn_core_pytorch(value, shapes, loc, attn)
            ctx.save_for_backward(value, shapes, level_start, loc, attn)
            return out
        M.MSDeformAttnFunction.forward = forward
        M.MSDeformAttnFunction._france_cuda = True


_install_compat()


def _get_intermediate_layers_partial(self, x, *, n=1, reshape=False, return_class_token=False, return_extra_tokens=False,
                                     norm=True):
    """DinoVisionTransformer.get_intermediate_layers (vendor, commit 6876159) with gradients for blocks >= self._k0 and the norm."""
    t, (H, W) = self.prepare_tokens_with_masks(x)
    output, total = [], len(self.blocks)
    take = range(total - n, total) if isinstance(n, int) else n
    for i, blk in enumerate(self.blocks):
        rope = self.rope_embed(H=H, W=W) if self.rope_embed is not None else None
        if i >= self._k0:
            with torch.enable_grad():
                t = blk(t, rope)
        else:
            t = blk(t, rope)
        if i in take:
            output.append(t)
    assert len(output) == len(take)
    # norm and slicing under enable_grad: views made inside the adapter's no_grad carry no grad_fn
    with torch.enable_grad():
        if norm:
            normed = []
            for out in output:
                if self.untie_cls_and_patch_norms:
                    normed.append(torch.cat((self.cls_norm(out[:, : self.n_storage_tokens + 1]),
                                             self.norm(out[:, self.n_storage_tokens + 1:])), dim=1))
                else:
                    normed.append(self.norm(out))
            output = normed
        cls = [o[:, 0] for o in output]
        extra = [o[:, 1: self.n_storage_tokens + 1] for o in output]
        output = [o[:, self.n_storage_tokens + 1:] for o in output]
        if reshape:
            B, _, h, w = x.shape
            output = [o.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                      for o in output]
    if not return_class_token and not return_extra_tokens:
        return tuple(output)
    if return_class_token and not return_extra_tokens:
        return tuple(zip(output, cls))
    if not return_class_token and return_extra_tokens:
        return tuple(zip(output, extra))
    return tuple(zip(output, cls, extra))


@MODELS.register_module()
class DINOv3ViTAdapter(BaseModule):

    def __init__(self, checkpoint=None, train_last=12, interaction_indexes=(4, 11, 17, 23), drop_path_rate=0.3, with_cp=False):
        super().__init__()
        from dinov3.hub.backbones import dinov3_vitl16
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
        vit = dinov3_vitl16(pretrained=False)
        self.adapter = DINOv3_Adapter(vit, interaction_indexes=list(interaction_indexes), drop_path_rate=drop_path_rate,
                                      with_cp=with_cp)
        self.checkpoint, self.train_last = checkpoint, int(train_last)
        vit._k0 = len(vit.blocks) - self.train_last
        if self.train_last > 0:
            for b in vit.blocks[vit._k0:]:
                b.requires_grad_(True)
            vit.norm.requires_grad_(True)
            if vit.untie_cls_and_patch_norms:
                vit.cls_norm.requires_grad_(True)
            vit.get_intermediate_layers = types.MethodType(_get_intermediate_layers_partial, vit)

    @property
    def vit(self):
        return self.adapter.backbone

    def init_weights(self):
        # called by MMEngine when the segmentor is built for training: ViT-L/16 weights from the official DINOv3 file; the
        # adapter keeps its own official initialisation. A loaded checkpoint (testing, resuming) overrides everything anyway.
        if not self.checkpoint:
            return
        sd = torch.load(self.checkpoint, map_location='cpu')
        self.vit.load_state_dict(sd, strict=True)
        nt = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_log(f'DINOv3ViTAdapter: ViT-L/16 from {self.checkpoint}; trained ViT blocks {self.train_last}, '
                  f'trained backbone parameters {nt / 1e6:.1f} M', logger='current')

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, x):
        f = self.adapter(x)
        return tuple(f[k] for k in ('1', '2', '3', '4'))


NO_DECAY_SUFFIX = ('pos_embed', 'cls_token', 'mask_token', 'storage_tokens', 'level_embed')


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ViTTopLayerDecayConstructor(DefaultOptimWrapperConstructor):

    def add_params(self, params, module, **kwargs):
        cfg = self.paramwise_cfg
        pre, mult_top, decay, nb = cfg['vit_prefix'], cfg['vit_lr_mult'], cfg['decay_rate'], cfg['num_blocks']
        groups = {}
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            no_decay = p.ndim == 1 or name.endswith('.bias') or name.endswith(NO_DECAY_SUFFIX)
            if name.startswith(pre):
                rest = name[len(pre):]
                mult = mult_top * decay ** (nb - 1 - int(rest.split('.')[1])) if rest.startswith('blocks.') else mult_top
            else:
                mult = 1.0
            key = (round(mult, 10), no_decay)
            g = groups.setdefault(key, dict(params=[], lr=self.base_lr * mult, weight_decay=0.0 if no_decay else self.base_wd))
            g['params'].append(p)
        params.extend(groups[k] for k in sorted(groups))
