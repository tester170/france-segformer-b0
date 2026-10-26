"""Training-control hooks.

VramLogHook          logs peak reserved memory (the allocator cache WDDM must hold, not only allocated tensors).
PlateauControlHook   for iteration-based training (mmengine's ReduceOnPlateau supports only by_epoch=True):
                     after every validation (i) saves the per-frame confusion matrices to
                     work_dir/val_confusion/iter_<n>.json, (ii) multiplies all LRs by `factor` when `monitor` has not
                     improved by `min_delta` for `lr_patience` validations (then waits `cooldown` validations),
                     (iii) stops training after `stop_patience` validations without improvement.
                     Validations before `start_iter` (warm-up) are not counted. Its state is stored in checkpoints,
                     so a resumed run continues the same counters; the LRs themselves are restored with the optimizer.
SpeedOptionsHook     channels_last memory format / TF32 for the whole run (ConvNeXt-B + UPerNet config).
Best checkpoints are saved by mmengine CheckpointHook(save_best=...).
"""
import json
import os
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class VramLogHook(Hook):
    priority = 'LOW'

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        runner.message_hub.update_scalar('train/reserved_mb', torch.cuda.max_memory_reserved() / 2 ** 20)


@HOOKS.register_module()
class PlateauControlHook(Hook):
    """LR reduction on plateau + early stopping for any train loop, decided on an EMA of the monitored metric
    (ema = alpha * value + (1 - alpha) * ema), so single noisy validations on 2 frames do not trigger actions.
    Improvement = EMA above its best by more than min_delta. After lr_patience validations without improvement:
    LR x factor, then `cooldown` validations are not counted towards the next reduction; at most max_reductions.
    After stop_patience validations without improvement: stop. Validations before start_iter (warm-up) only feed the
    EMA. Also saves val (per frame) and train-crop confusion matrices of every validation to work_dir/val_confusion.
    State is stored in checkpoints for resume."""
    priority = 'LOW'  # after CheckpointHook (best checkpoints on the raw metric), before ConsoleProgressHook

    def __init__(self, monitor='mIoU', ema_alpha=0.5, min_delta=0.1, lr_patience=5, factor=0.5, cooldown=2,
                 max_reductions=4, stop_patience=12, start_iter=0):
        self.monitor, self.ema_alpha, self.min_delta = monitor, ema_alpha, min_delta
        self.lr_patience, self.factor, self.cooldown, self.max_reductions = lr_patience, factor, cooldown, max_reductions
        self.stop_patience, self.start_iter = stop_patience, start_iter
        self.state = dict(ema=None, best=None, best_iter=None, best_raw=None, best_raw_iter=None, wait=0,
                          since_reduce=0, cooldown_left=0, reductions=0, history=[])

    def _save_confusion(self, runner, metrics):
        from .patches import TRAIN_CONFUSION as tc
        if not hasattr(runner.val_evaluator.metrics[0], 'last_frames'):    # standard IoUMetric: nothing to save
            return
        frames = runner.val_evaluator.metrics[0].last_frames
        out = Path(runner.work_dir) / 'val_confusion'
        out.mkdir(exist_ok=True)
        path = out / f'iter_{runner.iter}.json'
        tmp = path.with_suffix('.tmp')
        train_cm = None if tc.cm is None else tc.cm.reshape(tc.k, tc.k).cpu().numpy().tolist()
        metric = runner.val_evaluator.metrics[0]
        levels_cm = None
        if metric.last_level_cms:
            from .seg_metrics import level_key
            levels_cm = {f'{c}{n}': metric.last_level_cms[level_key(G)].tolist()
                         for c, steps in metric.chains.items() for n, G, _ in steps[1:]}
        tmp.write_text(json.dumps(dict(iter=runner.iter, epoch=runner.epoch,
                                       metrics={k: float(v) for k, v in metrics.items()},
                                       lr=[g['lr'] for g in runner.optim_wrapper.optimizer.param_groups][:1],
                                       frames={s: c.tolist() for s, c in frames.items()},
                                       train_crops_cm=train_cm, levels_cm=levels_cm)))
        os.replace(tmp, path)

    def after_val_epoch(self, runner, metrics=None):
        if not metrics or self.monitor not in metrics:
            return
        self._save_confusion(runner, metrics)
        s, raw, it = self.state, float(metrics[self.monitor]), runner.iter
        s['ema'] = raw if s['ema'] is None else self.ema_alpha * raw + (1 - self.ema_alpha) * s['ema']
        s['history'].append((it, raw, s['ema']))
        if s['best_raw'] is None or raw > s['best_raw']:
            s.update(best_raw=raw, best_raw_iter=it)
        if it < self.start_iter:
            return
        log = runner.logger.info
        if s['best'] is None or s['ema'] > s['best'] + self.min_delta:
            s.update(best=s['ema'], best_iter=it, wait=0, since_reduce=0)
            log(f'[plateau] {self.monitor} EMA improved to {s["ema"]:.3f} (raw {raw:.3f}) at iter {it}')
            return
        s['wait'] += 1
        if s['cooldown_left'] > 0:
            s['cooldown_left'] -= 1
        else:
            s['since_reduce'] += 1
        log(f'[plateau] no improvement {s["wait"]}/{self.stop_patience}: EMA {s["ema"]:.3f} vs best {s["best"]:.3f} '
            f'(+{self.min_delta} needed), raw {raw:.3f}')
        if s['since_reduce'] >= self.lr_patience and s['reductions'] < self.max_reductions:
            for g in runner.optim_wrapper.optimizer.param_groups:
                g['lr'] *= self.factor
            s.update(since_reduce=0, cooldown_left=self.cooldown, reductions=s['reductions'] + 1)
            log(f'[plateau] LR x{self.factor} (reduction {s["reductions"]}/{self.max_reductions}), '
                f'base lr now {runner.optim_wrapper.optimizer.param_groups[0]["lr"]:.3e}')
        if s['wait'] >= self.stop_patience:
            runner.train_loop.stop_training = True
            log(f'[plateau] early stop at iter {it}: best EMA {s["best"]:.3f} at {s["best_iter"]}, '
                f'best raw {s["best_raw"]:.3f} at {s["best_raw_iter"]}')

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {})['plateau_control'] = dict(self.state)

    def after_load_checkpoint(self, runner, checkpoint):
        state = checkpoint.get('meta', {}).get('plateau_control')
        if state:
            self.state.update(state)


@HOOKS.register_module()
class SpeedOptionsHook(Hook):
    """Speed options measured by tools/profile_train_step.py, set before anything runs:
    channels_last: convolution weights in NHWC (cuDNN tensor-core kernels; the maths is unchanged);
    tf32: TF32 tensor cores for the fp32 matmuls/convolutions that remain outside autocast."""
    priority = 'VERY_HIGH'

    def __init__(self, channels_last=False, tf32=False):
        self.channels_last, self.tf32 = channels_last, tf32

    def before_run(self, runner):
        if self.tf32:                              # otherwise PyTorch defaults (matmul fp32, cuDNN convolutions TF32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        if self.channels_last:
            runner.model.to(memory_format=torch.channels_last)
        runner.logger.info(f'[speed] channels_last={self.channels_last} tf32={self.tf32}')
