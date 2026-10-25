"""France dataset (as downloaded by scripts/fetch.py dataset) -> MMSegmentation layout used by configs/france:

  <out>/img_dir/train/<fragment>_y<row>_x<col>.png   2048 x 2048 tiles of the 12 training fragments (BGR PNG)
  <out>/ann_dir/train/<fragment>_y<row>_x<col>.png   train IDs 0..13 (CoSIA ids 0-5, 7-14 in order), 255 = ignore
  <out>/img_dir/val/<fragment>.png, <out>/ann_dir/val/<fragment>.png   the 4 validation fragments whole

<fragment> = <Region>__<lat>_<lon>, e.g. City__48.95591_2.90781. Train / val fragments come from splits/split_v1.json.
Tiles start every 2048 px; the last row / column of tiles is aligned to the fragment edge (it overlaps its
neighbour), so every tile has the full size and every pixel is covered. One process per fragment; a fragment whose
outputs exist is skipped. Usage: python scripts/prepare_data.py --src data/France --out data/france [--workers 3]"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

BUNDLE = Path(__file__).resolve().parents[1]
SPLIT = json.loads((BUNDLE / 'splits' / 'split_v1.json').read_text(encoding='utf-8'))
LUT = np.full(256, 255, np.uint8)
for c in SPLIT['classes']:
    LUT[c['cosia_id']] = c['train_id']
TILE = 2048
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def stem_of(rel):
    region, name = rel.split('/')
    parts = Path(name).stem.split('_')            # fragment_WGS84_<w>x<h>_z19_<lat>_<lon>
    return f'{region}__{parts[-2]}_{parts[-1]}'


def starts(n):
    s = list(range(0, n - TILE + 1, TILE))
    return s + [n - TILE] if s[-1] != n - TILE else s


def write(path, arr):
    tmp = path.with_name(path.stem + '.tmp.png')
    assert cv2.imwrite(str(tmp), arr, PNG), path
    os.replace(tmp, path)


def prepare(job):
    src, out, rel, split = job
    stem = stem_of(rel)
    img_dir, ann_dir = out / 'img_dir' / split, out / 'ann_dir' / split
    done = out / '.done' / stem
    if done.exists():
        return stem, 'present'
    rgb = tifffile.imread(src / rel)
    lab = tifffile.imread(src / rel.replace('.tif', '_indexed.tif'))
    assert rgb.ndim == 3 and rgb.dtype == np.uint8 and lab.shape == rgb.shape[:2], (rel, rgb.shape, lab.shape)
    bgr = np.ascontiguousarray(rgb[..., 2::-1])
    ids = LUT[lab]
    if split == 'val':
        write(img_dir / f'{stem}.png', bgr)
        write(ann_dir / f'{stem}.png', ids)
        n = 1
    else:
        n = 0
        for y in starts(ids.shape[0]):
            for x in starts(ids.shape[1]):
                name = f'{stem}_y{y:05d}_x{x:05d}.png'
                write(img_dir / name, bgr[y:y + TILE, x:x + TILE])
                write(ann_dir / name, ids[y:y + TILE, x:x + TILE])
                n += 1
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(str(n))
    return stem, f'{n} {"fragment" if split == "val" else "tiles"}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/France', help='folder with City/, Village/, Other/ (scripts/fetch.py dataset)')
    ap.add_argument('--out', default='data/france')
    ap.add_argument('--workers', type=int, default=3, help='parallel fragments (~1.5 GB of RAM each)')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    for d in ('img_dir', 'ann_dir'):
        for s in ('train', 'val'):
            (out / d / s).mkdir(parents=True, exist_ok=True)
    jobs = [(src, out, r, s) for s in ('val', 'train') for r in SPLIT[s]]
    missing = [r for _, _, r, _ in jobs if not (src / r).exists() or not (src / r.replace('.tif', '_indexed.tif')).exists()]
    if missing:
        raise SystemExit(f'missing source files in {src}: {missing[:3]} ... (python scripts/fetch.py dataset --out {src})')
    with ProcessPoolExecutor(a.workers) as ex:
        for stem, what in ex.map(prepare, jobs):
            print(f'{stem}: {what}', flush=True)
    n_tr = len(list((out / 'img_dir' / 'train').glob('*.png')))
    n_va = len(list((out / 'img_dir' / 'val').glob('*.png')))
    print(f'ready: {out} (train tiles {n_tr}, val fragments {n_va})', flush=True)


if __name__ == '__main__':
    main()
