"""EncoderDecoder with a streaming sliding-window inference for 120-megapixel fragments.

mmseg 1.2.2 adds every window through F.pad to a full-frame tensor and keeps C x H x W float logits of the whole
frame (14 x 12832 x 9581 x 4 B = 6.9 GB for one France fragment, plus the count map). Here windows are run grid row
by grid row (`slide_batch` windows per forward pass); their logits are added into a band buffer that covers only the
rows still reachable by later windows, and every row that no later window touches is finalised at once: divided by
its window count and argmax'ed into a uint8 prediction. The additions per pixel happen in the same order as in mmseg,
so with slide_batch=1 the prediction equals mmseg's slide inference pixel for pixel; peak memory is one band
(~C x 1.8k x W) instead of the frame (about 4.5 GB in total for a France fragment). Window placement is copied from
mmseg. Other test modes ('whole') and padded / flipped inputs fall back to mmseg."""
import torch
from mmengine.structures import PixelData
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class StreamSlideEncoderDecoder(EncoderDecoder):

    def __init__(self, *args, slide_batch=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.slide_batch = slide_batch

    def _windows(self, h_img, w_img):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        rows = []
        for h_idx in range(h_grids):
            y2 = min(h_idx * h_stride + h_crop, h_img)
            xs = [(max(min(w_idx * w_stride + w_crop, w_img) - w_crop, 0), min(w_idx * w_stride + w_crop, w_img))
                  for w_idx in range(w_grids)]
            rows.append((max(y2 - h_crop, 0), y2, xs))
        return rows

    @torch.no_grad()
    def stream_slide(self, inputs, meta):
        """inputs 1 x 3 x H x W (normalised) -> prediction H x W uint8 on the input device."""
        _, _, H, W = inputs.shape
        C = self.out_channels
        pred = torch.empty((H, W), dtype=torch.uint8, device=inputs.device)
        rows = self._windows(H, W)
        r0 = 0                                             # first row held in the band buffer
        acc = inputs.new_zeros((C, 0, W))
        cnt = inputs.new_zeros((0, W))
        for i, (y1, y2, xs) in enumerate(rows):
            if y2 - r0 > acc.shape[1]:                     # extend the band down to y2
                grow = y2 - r0 - acc.shape[1]
                acc = torch.cat([acc, inputs.new_zeros((C, grow, W))], 1)
                cnt = torch.cat([cnt, inputs.new_zeros((grow, W))], 0)
            for j in range(0, len(xs), self.slide_batch):
                chunk = xs[j:j + self.slide_batch]
                crops = torch.cat([inputs[:, :, y1:y2, x1:x2] for x1, x2 in chunk], 0)
                logits = self.encode_decode(crops, [dict(meta, img_shape=crops.shape[2:])] * len(chunk)).float()
                for k, (x1, x2) in enumerate(chunk):
                    acc[:, y1 - r0:y2 - r0, x1:x2] += logits[k]
                    cnt[y1 - r0:y2 - r0, x1:x2] += 1
            f = rows[i + 1][0] if i + 1 < len(rows) else H  # rows < f are reached by no later window
            if f > r0:
                n = f - r0
                pred[r0:f] = (acc[:, :n] / cnt[:n]).argmax(0).to(torch.uint8)
                acc, cnt = acc[:, n:].contiguous(), cnt[n:].contiguous()
                r0 = f
        return pred

    def predict(self, inputs, data_samples=None):
        metas = [ds.metainfo for ds in data_samples] if data_samples is not None else []
        if self.test_cfg.get('mode') != 'slide' or not metas or any(
                tuple(m.get('ori_shape', ())) != tuple(inputs.shape[2:]) or m.get('flip', False) for m in metas):
            return super().predict(inputs, data_samples)
        for i, ds in enumerate(data_samples):
            ds.set_data(dict(pred_sem_seg=PixelData(data=self.stream_slide(inputs[i:i + 1], metas[i])[None])))
        return data_samples
