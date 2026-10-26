"""DINOv3 ConvNeXt-B initial weights (LVD-1689M pretraining) for training the ConvNeXt-B + UPerNet config from scratch
(configs/france/dinov3-convnext-b_upernet_1xb2-amp_france-1024x1024.py).

Downloads the timm release of DINOv3 ConvNeXt-B from Hugging Face (repository timm/convnext_base.dinov3_lvd1689m, no
access request needed; SHA-256 checked) and converts it with the mapping used in the experiment (verified there: outputs
equal to timm's to 6e-6) into the format of the official DINOv3 code:
  checkpoints/dinov3_convnext_base_lvd1689m_official.pth
The result is loaded strictly into the official ConvNeXt (vendor/dinov3, env/fetch_vendor.sh). The weights are DINO
Materials under the DINOv3 License (LICENSE_DINOv3.md).
Usage (repository root, environment of env/setup_env.sh): python scripts/prepare_dinov3_weights.py [--out checkpoints]"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ('timm/convnext_base.dinov3_lvd1689m', 'afa69ae6094b5b04a48810de78c52c92d4cbb6d9436462f07bf36defb09e26c1')


def fetch(repo, sha, out_dir):
    dst = out_dir / 'timm' / repo.split('/')[1] / 'model.safetensors'
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{repo}/resolve/main/model.safetensors'
    print('download', url, flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'france-segformer-b0'}), timeout=60) as r, \
            open(dst.with_suffix('.part'), 'wb') as f:
        for chunk in iter(lambda: r.read(1 << 22), b''):
            f.write(chunk)
    got = hashlib.sha256(dst.with_suffix('.part').read_bytes()).hexdigest()
    assert got == sha, f'{repo}: SHA-256 {got} != {sha}'
    dst.with_suffix('.part').replace(dst)
    return dst


def load_safetensors(path):
    """Minimal reader of the safetensors format (8-byte header length, JSON header, raw little-endian buffer)."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], 'little')
    header = json.loads(raw[8:8 + n])
    buf = memoryview(raw)[8 + n:]
    out = {}
    for k, v in header.items():
        if k == '__metadata__':
            continue
        a, b = v['data_offsets']
        if v['dtype'] == 'BF16':
            t = torch.from_numpy(np.frombuffer(buf[a:b], dtype=np.int16).copy()).view(torch.bfloat16)
        else:
            t = torch.from_numpy(np.frombuffer(buf[a:b], dtype={'F32': np.float32, 'F16': np.float16}[v['dtype']]).copy())
        out[k] = t.reshape(v['shape'])
    return out


def convnext_timm_to_official(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith('stem.'):                                   # stem: conv 4x4 + LayerNorm2d
            nk = 'downsample_layers.0.' + k[len('stem.'):]
        elif m := re.fullmatch(r'stages\.(\d+)\.downsample\.(\d)\.(weight|bias)', k):
            nk = f'downsample_layers.{m[1]}.{m[2]}.{m[3]}'          # LayerNorm2d + conv 2x2
        elif m := re.fullmatch(r'stages\.(\d+)\.blocks\.(\d+)\.(.+)', k):
            tail = {'conv_dw.weight': 'dwconv.weight', 'conv_dw.bias': 'dwconv.bias', 'norm.weight': 'norm.weight',
                    'norm.bias': 'norm.bias', 'mlp.fc1.weight': 'pwconv1.weight', 'mlp.fc1.bias': 'pwconv1.bias',
                    'mlp.fc2.weight': 'pwconv2.weight', 'mlp.fc2.bias': 'pwconv2.bias', 'gamma': 'gamma'}[m[3]]
            nk = f'stages.{m[1]}.{m[2]}.{tail}'
        elif k in ('head.norm.weight', 'head.norm.bias'):          # final norm (official: norm == norms[3])
            out['norms.3.' + k.split('.')[-1]] = v
            nk = 'norm.' + k.split('.')[-1]
        else:
            raise KeyError(f'unmapped timm key {k}')
        out[nk] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'checkpoints'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from france_mmseg.dinov3_backbones import ConvNeXt, convnext_sizes  # official dinov3 code (vendor/dinov3)
    conv = convnext_timm_to_official(load_safetensors(fetch(*SOURCE, out)))
    ConvNeXt(**{k: convnext_sizes['base'][k] for k in ('depths', 'dims')}).load_state_dict(conv, strict=True)
    dst = out / 'dinov3_convnext_base_lvd1689m_official.pth'
    torch.save(dict(state_dict=conv, meta=dict(source=SOURCE[0], source_sha256=SOURCE[1],
                                               arch='dinov3 ConvNeXt-base (facebookresearch/dinov3)', pretrain='LVD-1689M')),
               dst)
    print('written', dst, flush=True)


if __name__ == '__main__':
    main()
