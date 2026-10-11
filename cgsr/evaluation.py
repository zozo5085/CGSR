"""Shared evaluator, metrics and paired image-level bootstrap.

These functions are the ones used to produce the paper-facing numbers,
moved here without changes to their arithmetic:

* ``shared_postprocess``: the shared evaluator's post-processing (class-presence
  filter ``PD``, bilinear resize to the model input and then to the label
  size, softmax, argmax). Applied identically to Base, Candidate and CGSR.
  It re-implements the evaluation post-processing of the upstream ReCLIP++
  code (Wang & Kang); see THIRD_PARTY_NOTICES.md.
* ``raw_label`` / ``intersection_union`` / ``miou``: per-image intersections and
  unions; dataset mIoU = sum over classes of I/U (0 when the union is empty)
  divided by the number of classes.
* ``bootstrap``: paired image-level bootstrap (one resample shared by all
  methods), percentile and BCa intervals.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def shared_postprocess(logits, input_shape, original_shape, pd_threshold):
    """Token logits ``(K,Hp,Wp)`` -> label map at ``original_shape``.

    Operates on a clone, so the input tensor is never modified.
    """
    output = logits.unsqueeze(0).clone()
    N, C, H, W = output.shape
    scores = F.softmax(output * 10, dim=1)
    max_cls_conf = scores.view(N, C, -1).max(dim=-1)[0]
    selected = (max_cls_conf < pd_threshold)[:, :, None, None].expand(N, C, H, W)
    output[selected] = -100
    output = F.interpolate(output, input_shape, None, "bilinear", False).reshape(
        1, C, input_shape[0], input_shape[1]
    )
    output = F.interpolate(output, original_shape, None, "bilinear", False).reshape(
        1, C, original_shape[0], original_shape[1]
    )
    output = F.softmax(output, dim=1)
    return torch.argmax(output, dim=1).squeeze(0)


def raw_label(label_path: Path, reduce_zero: bool):
    """Load a label PNG; with ``reduce_zero`` map 0 -> ignore and k -> k-1."""
    label = np.asarray(Image.open(label_path)).copy().astype(np.int64)
    if reduce_zero:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    return label


def intersection_union(prediction: torch.Tensor, label: np.ndarray, class_count: int):
    """Per-class intersection and union over valid pixels (label != 255)."""
    pred = prediction.detach().cpu().numpy().astype(np.int64)
    valid = label != 255
    target = label[valid]
    pred = pred[valid]
    intersect = pred[pred == target]
    area_intersect = np.bincount(intersect, minlength=class_count + 1)[:class_count + 1]
    area_pred = np.bincount(pred, minlength=class_count + 1)[:class_count + 1]
    area_target = np.bincount(target, minlength=class_count + 1)[:class_count + 1]
    return area_intersect, area_pred + area_target - area_intersect


def miou(I, U, K):
    """Dataset mIoU in [0, 1] from summed intersections/unions."""
    return float(np.where(U > 0, I / np.maximum(U, 1), 0.0).sum() / K)


def _ndtri(p):
    return NormalDist().inv_cdf(float(p))


def _ndtr(z):
    return NormalDist().cdf(float(z))


def bootstrap(stat_fn, n, B, seed):
    """Paired image-level bootstrap.

    ``stat_fn(idx) -> float`` for an index array. One resample per replicate
    is shared by every contrast evaluated with the same seed. Returns the point
    estimate, the 95 % percentile CI (paper-facing), the 95 % BCa CI, and the
    fraction of positive replicates.
    """
    rng = np.random.RandomState(seed)
    theta = stat_fn(np.arange(n))
    reps = np.array([stat_fn(rng.randint(0, n, n)) for _ in range(B)])
    jack = np.array([stat_fn(np.delete(np.arange(n), i)) for i in range(n)])
    z0 = _ndtri((reps < theta).mean().clip(1e-6, 1 - 1e-6))
    jm = jack.mean()
    a = ((jm - jack) ** 3).sum() / (6.0 * (((jm - jack) ** 2).sum()) ** 1.5 + 1e-12)
    zl, zh = _ndtri(0.025), _ndtri(0.975)
    lo = _ndtr(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    hi = _ndtr(z0 + (z0 + zh) / (1 - a * (z0 + zh)))
    return dict(
        point=float(theta),
        percentile=[float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
        bca=[float(np.percentile(reps, 100 * lo)), float(np.percentile(reps, 100 * hi))],
        frac_positive=float((reps > 0).mean()),
        boot_mean=float(reps.mean()),
    )
