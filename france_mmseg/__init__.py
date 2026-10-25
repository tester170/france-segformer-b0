"""Modules the France configs use beyond MMSegmentation 1.2.2 (registered in the mmseg registries on import)."""
from PIL import Image

from . import patches  # noqa: F401  (must precede model building)
from . import hooks, segmentor, transforms  # noqa: F401

# validation fragments are ~120 megapixels: above Pillow's decompression-bomb guard (89 Mpx warning, 179 Mpx error)
Image.MAX_IMAGE_PIXELS = None
