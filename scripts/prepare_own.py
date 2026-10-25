"""Your own labelled data -> the MMSegmentation layout of configs/france (for fine-tuning and evaluation):

  my_data/train/images/*  images: GeoTIFF/TIFF, JP2, IMG, VRT, JPG, PNG, BMP, WEBP (8-bit RGB, ~0.2 m per pixel)
  my_data/train/masks/*   masks with the same names (any extension): PNG or TIFF, the same size as the image, either
                          one band with CoSIA codes 0-5, 7-14 (255 or any other value = not labelled) or RGB colours
                          of the CoSIA palette (any other colour = not labelled)
  my_data/val/...         optional; without it every 5th training image (alphabetical) is held out for validation

Output (--out): img_dir|ann_dir/train = 2048 x 2048 tiles (smaller images and edges are padded: image 0, label 255),
img_dir|ann_dir/val = validation images whole, labels as train IDs 0..13 (255 = not labelled). Prints per image the
size and the labelled share, and the class distribution of the training part.
Usage: python scripts/prepare_own.py --data my_data --out data/my_data [--bands 1 2 3]"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio

BUNDLE = Path(__file__).resolve().parents[1]
CLASSES = json.loads((BUNDLE / 'splits' / 'split_v1.json').read_text(encoding='utf-8'))['classes']
LUT = np.full(256, 255, np.uint8)
for c in CLASSES:
    LUT[c['cosia_id']] = c['train_id']
COLOR_KEY = {(r << 16) | (g << 8) | b: c['train_id'] for c in CLASSES for r, g, b in [c['color']]}
EXT = {'.tif', '.tiff', '.jp2', '.img', '.vrt', '.jpg', '.jpeg', '.png', '.bmp', '.webp'}
TILE = 2048
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def read_image(path, bands):
    with rasterio.open(path) as ds:
        idx = bands or [1, 2, 3]
        assert ds.count >= max(idx), f'{path.name}: {ds.count} band(s), bands {idx} requested'
        rgb = ds.read(idx)
    assert rgb.dtype == np.uint8, f'{path.name}: 8-bit RGB is expected, got {rgb.dtype}'
    return np.ascontiguousarray(np.transpose(rgb, (1, 2, 0))[..., ::-1])      # BGR, as LoadImageFromFile


def read_mask(path):
    with rasterio.open(path) as ds:
        m = ds.read()
    if m.shape[0] == 1:
        assert m.dtype == np.uint8, f'{path.name}: one-band mask must be 8-bit, got {m.dtype}'
        return LUT[m[0]]
    m = m[:3].astype(np.int64)
    key = (m[0] << 16) | (m[1] << 8) | m[2]
    out = np.full(key.shape, 255, np.uint8)
    for k, tid in COLOR_KEY.items():
        out[key == k] = tid
    return out


def pairs(folder):
    imgs = sorted(p for p in (folder / 'images').iterdir() if p.suffix.lower() in EXT)
    masks = {p.stem: p for p in (folder / 'masks').iterdir() if p.suffix.lower() in EXT}
    missing = [p.name for p in imgs if p.stem not in masks]
    assert not missing, f'no masks for: {missing[:5]}'
    return [(p, masks[p.stem]) for p in imgs]


def starts(n):
    """Tile origins along one side: every TILE px, the last one aligned to the edge; one tile if n <= TILE."""
    if n <= TILE:
        return [0]
    s = list(range(0, n - TILE + 1, TILE))
    return s + [n - TILE] if s[-1] != n - TILE else s


def write(path, arr):
    assert cv2.imwrite(str(path), arr, PNG), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='my_data')
    ap.add_argument('--out', default='data/my_data')
    ap.add_argument('--bands', nargs=3, type=int, help='band numbers of R G B (default 1 2 3)')
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    train = pairs(data / 'train')
    if (data / 'val' / 'images').exists():
        val = pairs(data / 'val')
    else:
        val, train = train[::5], [p for i, p in enumerate(train) if i % 5]
        assert train and val, 'at least 2 images are needed when there is no val/ folder'
        print(f'no val/ folder: {len(val)} of {len(train) + len(val)} images held out for validation (every 5th)')
    shutil.rmtree(out, ignore_errors=True)
    for d in ('img_dir', 'ann_dir'):
        for s in ('train', 'val'):
            (out / d / s).mkdir(parents=True)
    counts = np.zeros(256, np.int64)
    for split, items in (('train', train), ('val', val)):
        for ip, mp in items:
            img, lab = read_image(ip, a.bands), read_mask(mp)
            assert img.shape[:2] == lab.shape, f'{ip.name}: image {img.shape[:2]} and mask {lab.shape} differ'
            labelled = float((lab != 255).mean())
            print(f'{split:5s} {ip.name}: {img.shape[1]} x {img.shape[0]} px, labelled {100 * labelled:.1f} %', flush=True)
            if split == 'val':
                write(out / 'img_dir' / 'val' / f'{ip.stem}.png', img)
                write(out / 'ann_dir' / 'val' / f'{ip.stem}.png', lab)
                continue
            counts += np.bincount(lab.ravel(), minlength=256)
            H, W = lab.shape
            for y in starts(H):
                for x in starts(W):
                    ti = np.zeros((TILE, TILE, 3), np.uint8)             # padding: image 0, label 255
                    tl = np.full((TILE, TILE), 255, np.uint8)
                    h, w = min(TILE, H - y), min(TILE, W - x)
                    ti[:h, :w], tl[:h, :w] = img[y:y + h, x:x + w], lab[y:y + h, x:x + w]
                    write(out / 'img_dir' / 'train' / f'{ip.stem}_y{y:05d}_x{x:05d}.png', ti)
                    write(out / 'ann_dir' / 'train' / f'{ip.stem}_y{y:05d}_x{x:05d}.png', tl)
    n_tiles = len(list((out / 'img_dir' / 'train').glob('*.png')))
    total = counts[:len(CLASSES)].sum()
    assert total > 0, 'the training masks contain no labelled pixels of the 14 classes'
    print(f'ready: {out} (train tiles {n_tiles}, val images {len(val)})')
    print('classes in the training masks: ' + ', '.join(
        f'{c["name_ru"]} {100 * counts[c["train_id"]] / total:.1f} %' for c in CLASSES if counts[c['train_id']]))


if __name__ == '__main__':
    main()
