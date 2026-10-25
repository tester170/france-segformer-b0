"""RandomDihedral: uniform element of the dihedral group D4 (rot90^k, optional flip) on the image and the label.
Nadir aerial views have no preferred orientation, so all 8 symmetries of the square are equally valid samples."""
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomDihedral(BaseTransform):
    """Returns strided views without copying; place it right before PackSegInputs, whose CHW transpose makes the
    single copy. Pixel-wise transforms (photometric) commute with it and go before it."""

    def transform(self, results):
        k, f = np.random.randint(4), np.random.rand() < 0.5
        for key in ['img'] + results.get('seg_fields', []):
            a = np.rot90(results[key], k)
            results[key] = a[:, ::-1] if f else a
        results['img_shape'] = results['img'].shape[:2]
        return results
