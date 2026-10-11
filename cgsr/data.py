"""Validation file lists for the five paper-facing datasets.

The image/label pairing follows the file-list conventions of the base code
(``read_file_list`` in the upstream ReCLIP++ repository), with three
release-side differences that do not change any prediction or the point mIoU:

* Only the validation split is listed (the upstream helper also opens the
  training lists and pseudo-label JSON files, which evaluation never uses).
* Directory listings are sorted. On the machine that produced the paper
  numbers the unsorted listing was already in sorted order, so the image
  order (and therefore the bootstrap) is unchanged.
* Cityscapes labels are always ``gtFine`` ``*_gtFine_labelTrainIds.png``,
  derived from the image id, as in the paper-facing protocol.
"""

from __future__ import annotations

import os
from pathlib import Path


class DatasetError(SystemExit):
    """Raised when the dataset layout does not match the expected structure."""


def _require(path: Path, what: str, dataset: str):
    if not path.exists():
        raise DatasetError(
            f"[{dataset}] {what} not found: {path}\n"
            "Check --data-root and the expected layout in docs/DATASETS.md."
        )


def _read_ids(list_file: Path, dataset: str):
    _require(list_file, "validation list", dataset)
    with open(list_file, "r") as stream:
        return stream.read().split()


def _listdir_ids(image_dir: Path, dataset: str):
    _require(image_dir, "image directory", dataset)
    return sorted(name[:-4] for name in os.listdir(image_dir) if name[-3:] in ["jpg"])


def build_val_list(dataset: str, data_root: str | os.PathLike):
    """Return ``(ids, image_paths, label_paths)`` for the validation split."""
    root = Path(data_root)
    _require(root, "--data-root", dataset)
    if dataset == "voc":
        ids = _read_ids(root / "ImageSets" / "Segmentation" / "val.txt", dataset)
        images = [root / "JPEGImages" / f"{i}.jpg" for i in ids]
        labels = [root / "SegmentationClass" / f"{i}.png" for i in ids]
    elif dataset == "context59":
        ids = _read_ids(root / "ImageSets" / "SegmentationContext" / "val.txt", dataset)
        images = [root / "JPEGImages" / f"{i}.jpg" for i in ids]
        labels = [root / "SegmentationClassContext" / f"{i}.png" for i in ids]
    elif dataset == "ade20k":
        ids = _listdir_ids(root / "images" / "validation", dataset)
        images = [root / "images" / "validation" / f"{i}.jpg" for i in ids]
        labels = [root / "annotations" / "validation" / f"{i}.png" for i in ids]
    elif dataset == "coco_stuff27":
        ids = _listdir_ids(root / "images" / "val2017", dataset)
        images = [root / "images" / "val2017" / f"{i}.jpg" for i in ids]
        labels = [root / "annotations" / "val2017" / f"{i}_27labelTrainIds.png" for i in ids]
    elif dataset == "cityscapes":
        image_dir = root / "leftImg8bit" / "val"
        _require(image_dir, "image directory", dataset)
        files = sorted(image_dir.glob("*/*_leftImg8bit.png"), key=lambda p: p.name)
        ids = [p.name[: -len("_leftImg8bit.png")] for p in files]
        images = files
        labels = [
            root / "gtFine" / "val" / i.split("_")[0] / f"{i}_gtFine_labelTrainIds.png"
            for i in ids
        ]
    else:
        raise DatasetError(f"unknown dataset: {dataset}")

    if not ids:
        raise DatasetError(f"[{dataset}] no validation images found under {root}")
    missing = [p for p in list(images) + list(labels) if not Path(p).is_file()]
    if missing:
        raise DatasetError(
            f"[{dataset}] {len(missing)} image/label files are missing, e.g. {missing[:3]}\n"
            "See docs/DATASETS.md for the expected layout (COCO-Stuff-27 labels must be "
            "converted first; Cityscapes needs gtFine *_labelTrainIds.png)."
        )
    return ids, [Path(p) for p in images], [Path(p) for p in labels]
