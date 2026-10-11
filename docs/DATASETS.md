# Datasets

No dataset files are included. Obtain each dataset from its official provider under that provider's terms, then point `--data-root` at the directory described below.

## Summary

| Dataset | Split | Images | Label protocol | Config | Expected root (`--data-root`) |
|---|---|---:|---|---|---|
| PASCAL VOC 2012 | val | 1,449 | 20 classes; background and boundary (255) ignored (`reduce_zero_label`) | `configs/voc/voc_cgsr.yaml` | `.../VOCdevkit/VOC2012` |
| PASCAL Context-59 | val | 5,105 | 59 classes in official `59_labels.txt` order ("H1"); background ignored (`reduce_zero_label`) | `configs/context59/context59_cgsr.yaml` | `.../VOCdevkit/VOC2010` |
| ADE20K (SceneParsing, ADEChallengeData2016) | validation | 2,000 | 150 classes; label 0 ("other") ignored (`reduce_zero_label`) | `configs/ade20k/ade20k_cgsr.yaml` | `.../ADEChallengeData2016` |
| COCO-Stuff-27 | val2017 | 5,000 | 27 coarse classes, label values 0–26, 255 ignored | `configs/coco_stuff27/coco_stuff27_cgsr.yaml` | `.../COCOStuff27` |
| Cityscapes | val | 500 | 19 classes, gtFine `labelTrainIds`, 255 ignored | `configs/cityscapes/cityscapes_cgsr.yaml` | `.../cityscapes` |

The image counts were checked on the copies used for the paper; `tools/eval_cgsr.py` warns if a full run finds a different number.

Ignore handling is the same for every dataset. Pixels labelled 255 are excluded from the confusion counts. With `reduce_zero_label`, label 0 becomes 255 and labels `k ≥ 1` become `k − 1` before scoring. No other remapping is applied to ground truth.

## Expected directory layouts

Only the files listed here are read. Training splits and pseudo-label files are not needed for evaluation.

### PASCAL VOC 2012

```text
VOC2012/
├── ImageSets/Segmentation/val.txt        # 1,449 ids, one per line (order = evaluation order)
├── JPEGImages/<id>.jpg
└── SegmentationClass/<id>.png            # palette PNG: 0 = background, 1..20 classes, 255 = boundary
```

### PASCAL Context-59

```text
VOC2010/
├── ImageSets/SegmentationContext/val.txt   # 5,105 ids
├── JPEGImages/<id>.jpg
└── SegmentationClassContext/<id>.png       # 0 = background, 1..59 = classes in 59_labels.txt order
```

`SegmentationClassContext` holds the standard 59-class Context label maps: value = line number in the official `59_labels.txt`, 0 = background. The script that produced the copies used for the paper is not part of this release.

**Class order (H1).** The base predictor's Context checkpoint and its text embedding use the alphabetical class order of the base code (upstream ReCLIP++). After the argmax, `tools/eval_cgsr.py` remaps predicted ids to the official order by class name (`cgsr/context_h1.py`). Ground truth is not remapped.

### ADE20K

```text
ADEChallengeData2016/
├── images/validation/<id>.jpg              # 2,000 images; all *.jpg files are evaluated, sorted by name
└── annotations/validation/<id>.png         # 0 = other (ignored), 1..150 = classes
```

### COCO-Stuff-27

```text
COCOStuff27/
├── images/val2017/<id>.jpg                       # 5,000 COCO 2017 val images, sorted by name
└── annotations/val2017/<id>_27labelTrainIds.png  # values 0..26, 255 = ignore
```

The `_27labelTrainIds.png` files must be created from the COCO-Stuff 164k fine annotations (values 0–181, 255 = unlabeled):

```bash
python tools/convert_coco_stuff27_labels.py --dataroot /path/to/cocostuff --outroot /path/to/COCOStuff27 --splits val2017
```

The fine-to-coarse grouping is the standard COCO-Stuff-27 table of the IIC / PiCIE / STEGO line of work (taken from STEGO, MIT license). The coarse class order matches the upstream `coco_stuff_classes_27` list. Copy or link `images/val2017` into the same root.

### Cityscapes

```text
cityscapes/
├── leftImg8bit/val/<city>/<id>_leftImg8bit.png               # 500 images, sorted by file name
└── gtFine/val/<city>/<id>_gtFine_labelTrainIds.png           # 19 trainIds, 255 = ignore
```

`*_labelTrainIds.png` files are produced by the official `cityscapesScripts` (`createTrainIdLabelImgs`). Only **gtFine** labels are used. Scores against gtCoarse are not part of the paper protocol.

## Evaluation commands

After `scripts/setup_upstream.*` and placing the weights (`docs/WEIGHTS.md`):

```bash
bash scripts/eval.sh voc          /path/to/VOCdevkit/VOC2012
bash scripts/eval.sh context59    /path/to/VOCdevkit/VOC2010
bash scripts/eval.sh ade20k       /path/to/ADEChallengeData2016
bash scripts/eval.sh coco_stuff27 /path/to/COCOStuff27
bash scripts/eval.sh cityscapes   /path/to/cityscapes
```

On Windows, use `powershell -File scripts\eval.ps1 <dataset> <data_root>`. See `README.md` for the underlying `tools/eval_cgsr.py` command.

## Sources

Obtain each dataset from its official provider, and follow that provider's license and terms of use. The images and annotations of these datasets are not redistributable through this repository.
* PASCAL VOC 2012
* PASCAL Context (the label list is at https://cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt)
* ADE20K
* COCO 2017 and COCO-Stuff 164k
* Cityscapes

Download URLs other than the Context label list are not listed here because they were not verified for this release.
