"""PASCAL Context-59 class order used by the paper-facing protocol ("H1").

The base predictor's Context checkpoint and its text embedding
(``context_ViT16_clip_text_alpha.pth``) use the alphabetical class order of
``pascal_context_classes`` in the base code. The ground-truth PNGs in
``SegmentationClassContext`` follow the official PASCAL-Context
``59_labels.txt`` order (after ``reduce_zero_label``: id = line number - 1).

Predicted class ids are therefore remapped by class name after the argmax.
GT, logits, softmax, PD filter and argmax are unchanged. This equals the
two-step remap used for the frozen table (alphabetical -> project-internal
order -> official order). The equality is checked in docs/RELEASE_CHECKS.md.
"""

from __future__ import annotations

import numpy as np

# Source: https://cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt
OFFICIAL_59_LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "table", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor", "bag", "bed", "bench", "book", "building", "cabinet", "ceiling", "cloth",
    "computer", "cup", "door", "fence", "floor", "flower", "food", "grass", "ground",
    "keyboard", "light", "mountain", "mouse", "curtain", "platform", "sign", "plate", "road",
    "rock", "shelves", "sidewalk", "sky", "snow", "bedclothes", "track", "tree", "truck",
    "wall", "water", "window", "wood",
]

# Upstream class-name spelling -> official spelling (only names that differ).
ALIASES = {"airplane": "aeroplane", "potted plant": "pottedplant", "tv monitor": "tvmonitor"}


def alphabetical_to_official_lut(model_class_names) -> np.ndarray:
    """``lut[model_channel] = official_id``, matched by class name; asserts a bijection."""
    names = list(model_class_names)
    assert len(names) == 59, len(names)
    official = {name: i for i, name in enumerate(OFFICIAL_59_LABELS)}
    assert len(official) == 59
    lut = np.full(59, -1, np.int64)
    for channel, name in enumerate(names):
        key = ALIASES.get(name, name)
        assert key in official, f"Context class {name!r} has no official match"
        lut[channel] = official[key]
    assert sorted(lut.tolist()) == list(range(59)), "Context class map is not a bijection"
    return lut
