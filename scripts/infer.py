"""Segmentation of your own orthophotos with the trained model (streaming sliding window 1024 / stride 768, as in the
evaluation). Input: one raster or a folder of rasters readable by GDAL (GeoTIFF/TIFF, JP2, IMG, VRT, JPG, PNG, ...),
RGB 8 bit, ~0.2 m per pixel for best results. For every input it writes
  <name>_cosia.tif|png   class mask with CoSIA codes (0-5, 7-14) and the CoSIA colour table; GeoTIFF with the source
                         georeferencing when the input has one, else PNG
  <name>_color.png       colour mask (CoSIA palette), and with --overlay a half-transparent overlay on the image
  <name>_preview.jpg     image and colour mask side by side, reduced to 1600 px width (for a quick look)
Usage: python scripts/infer.py --input <file or folder> --out <folder> [--checkpoint ...] [--bands 1 2 3] [--fp16]
       [--overlay]"""
import argparse
import json
import os
import sys

os.environ['MPLBACKEND'] = 'Agg'   # notebook kernels export an inline backend unknown to this environment
import warnings  # noqa: E402

warnings.filterwarnings('ignore', category=UserWarning)   # mmseg deprecation notices at model building
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
SPLIT = json.loads((BUNDLE / 'splits' / 'split_v1.json').read_text(encoding='utf-8'))
EXT = {'.tif', '.tiff', '.jp2', '.img', '.vrt', '.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default=str(BUNDLE / 'configs/france/segformer_mit-b0_1xb8-amp_france-1024x1024.py'))
    ap.add_argument('--checkpoint', default=str(BUNDLE / 'checkpoints/segformer_b0_france_ep76.pth'))
    ap.add_argument('--fp16', action='store_true', help='half-precision inference (GPU only)')
    ap.add_argument('--overlay', action='store_true')
    ap.add_argument('--bands', nargs=3, type=int, default=[1, 2, 3], help='band numbers of R G B (default 1 2 3)')
    a = ap.parse_args()

    import cv2
    import france_mmseg  # noqa: F401
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmseg.registry import MODELS
    init_default_scope('mmseg')
    cfg = Config.fromfile(a.config)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MODELS.build(cfg.model)
    load_checkpoint(model, a.checkpoint, map_location='cpu', logger='silent')
    model = model.to(dev).eval()
    mean = torch.tensor(cfg.model.data_preprocessor.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg.model.data_preprocessor.std, device=dev).view(1, 3, 1, 1)
    to_cosia = np.array([c['cosia_id'] for c in SPLIT['classes']], np.uint8)
    palette = np.array([c['color'] for c in SPLIT['classes']], np.uint8)
    cmap = {c['cosia_id']: tuple(c['color']) + (255,) for c in SPLIT['classes']}

    src = Path(a.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXT) if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        t0 = time.time()
        with rasterio.open(f) as ds:
            assert ds.count >= max(a.bands), f'{f}: {ds.count} band(s), bands {a.bands} requested'
            rgb = ds.read(a.bands)                         # 3 x H x W
            profile, georef = ds.profile, ds.crs is not None
        assert rgb.dtype == np.uint8, f'{f}: 8-bit RGB is expected, got {rgb.dtype}'
        x = (torch.from_numpy(rgb).to(dev)[None].float() - mean) / std
        meta = dict(img_shape=x.shape[2:], ori_shape=x.shape[2:], pad_shape=x.shape[2:])
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=a.fp16 and dev == 'cuda'):
            pred = model.stream_slide(x, meta).cpu().numpy()
        del x
        codes = to_cosia[pred]
        if georef:
            p = out / f'{f.stem}_cosia.tif'
            prof = dict(driver='GTiff', height=codes.shape[0], width=codes.shape[1], count=1, dtype='uint8',
                        crs=profile['crs'], transform=profile['transform'], compress='deflate', tiled=True,
                        nodata=255)
            with rasterio.open(p, 'w', **prof) as dst:
                dst.write(codes, 1)
                dst.write_colormap(1, cmap)
        else:
            p = out / f'{f.stem}_cosia.png'
            cv2.imwrite(str(p), codes)
        color = palette[pred]
        cv2.imwrite(str(out / f'{f.stem}_color.png'), color[..., ::-1])
        k = min(1.0, 800 / pred.shape[1])                   # preview: image | mask, 1600 px wide
        size = (max(1, round(pred.shape[1] * k)), max(1, round(pred.shape[0] * k)))
        img_s = cv2.resize(np.transpose(rgb, (1, 2, 0)), size, interpolation=cv2.INTER_AREA)
        col_s = cv2.resize(color, size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / f'{f.stem}_preview.jpg'), np.concatenate([img_s, col_s], 1)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if a.overlay:
            img = np.transpose(rgb, (1, 2, 0))
            blend = (0.55 * img + 0.45 * color).astype(np.uint8)
            cv2.imwrite(str(out / f'{f.stem}_overlay.jpg'), blend[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        share = np.bincount(pred.ravel(), minlength=len(palette)) / pred.size
        top = ', '.join(f'{SPLIT["classes"][i]["name_ru"]} {100 * share[i]:.1f} %' for i in np.argsort(-share)[:4])
        print(f'{f.name}: {pred.shape[1]} x {pred.shape[0]} px, {time.time() - t0:.1f} s -> {p.name} ({top})', flush=True)


if __name__ == '__main__':
    main()
