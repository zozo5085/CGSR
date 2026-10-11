# Context-Guided Selective Rollback (CGSR)

Code and evaluation protocol for **"Context-Guided Selective Rollback for Post-Refinement Verification in CLIP-Based Semantic Segmentation"**.

## Overview

CGSR is a training-free post-refinement verifier evaluated on a fixed ReCLIP++ refinement pipeline. It selects between Base and Candidate logit states using local pre-refinement class support.

* **Base** (`FrozenBasePredictor`): ReCLIP++ (Wang & Kang; ViT-B/16, author-released checkpoints), used frozen and unmodified.
* **Candidate** (`FixedCandidateGenerator`): an existing, fixed refinement stage (SFP + DTLR). It is not part of the CGSR contribution.
* **CGSR** (`SelectiveRollbackVerifier`): at every token, CGSR keeps the Candidate or rolls back to the Base logits. The decision depends on which of the two predicted classes has more support among the weighted Base class probabilities in a 5×5 neighbourhood. CGSR has no learnable parameters and needs no training or ground truth.

The scope of the evidence is deliberately narrow:
* One Base/refiner family.
* Five datasets.
* Full validation splits.
* Paired-bootstrap CIs.

Read `docs/LIMITATIONS.md` before relying on the method elsewhere.

## Method

```text
image ─► FrozenBasePredictor ─► base_logits L0, projected_features F, penultimate_attention
      ─► FixedCandidateGenerator ─► candidate_logits Lr
      ─► SelectiveRollbackVerifier:
           P0 = softmax(L0);  Q = 5×5 local class support of P0;  c0 = argmax L0,  cr = argmax Lr
           g  = Q(cr) − Q(c0);   final_logits = Lr where g > 0, else L0   (whole logit vector, token grid)
      ─► shared_postprocess ─► segmentation
```

`CGSRPipeline` (`cgsr/model_cgsr.py`) runs these stages. Full definitions, the paper-notation-to-code mapping, the Candidate generator's token selection and the shared evaluator are in `docs/METHOD.md`.

**Code ownership.**

| Ownership | Components | Where |
|---|---|---|
| CGSR-owned (this work) | `SelectiveRollbackVerifier`, `local_class_support` | `cgsr/verifier.py`; depends only on PyTorch |
| Experimental integration | `CGSRPipeline` | `cgsr/model_cgsr.py` |
| Third-party / upstream | `FrozenBasePredictor` (the upstream ReCLIP++ class `RECLIPPP`); `FixedCandidateGenerator` (class `RECLIPPP` of the separately distributed `cgsr/candidate/model_sfp_dtlr.py`) | not redistributed (`THIRD_PARTY_NOTICES.md`) |

The CGSR core can be used on its own, without the base-predictor code or the Candidate generator:

```python
from cgsr import SelectiveRollbackVerifier, local_class_support
final_logits, states = SelectiveRollbackVerifier()(base_logits, candidate_logits, projected_features)
```

`tests/test_standalone_import.py` checks this in a clean copy of the package.

## Repository structure

```text
cgsr/                 SelectiveRollbackVerifier / local_class_support (verifier.py, CGSR core, PyTorch only),
                      CGSRPipeline integration (model_cgsr.py), shared_postprocess and metrics (evaluation.py),
                      data lists, frozen settings, base-code loader (upstream.py)
cgsr/candidate/       FixedCandidateGenerator location (distributed separately, see README there)
configs/<dataset>/    paper-facing evaluation configs (5 datasets)
tools/eval_cgsr.py    evaluate Base / Candidate / CGSR on one dataset
tools/verify_assets.py, tools/make_text_embeddings.py, tools/convert_coco_stuff27_labels.py
scripts/              setup_upstream.{sh,ps1}, eval.{sh,ps1}
tests/                test_standalone_import.py (CGSR core without base code / Candidate)
docs/                 METHOD, DATASETS, WEIGHTS, RESULTS, LIMITATIONS, release checks
results/tables/       paper-facing result tables (CSV)
weights/              place checkpoints and text embeddings here (not in git)
```

## Installation

You need Python ≥ 3.9, a CUDA GPU, and PyTorch with CUDA (https://pytorch.org). Then:

```bash
pip install -r requirements.txt
```

The paper numbers were produced with Python 3.10.19, PyTorch 2.11.0.dev20260214+cu128 (nightly), CUDA 12.8 and an NVIDIA RTX 5090. Any CUDA GPU that runs the base predictor should work. Bit-exact agreement is established only for that environment (`REPRODUCIBILITY.md`).

**Base-predictor code (upstream ReCLIP++).** It is not included in this repository. Clone it at the pinned commit:

```bash
bash scripts/setup_upstream.sh third_party/base_predictor    # Windows: powershell -File scripts\setup_upstream.ps1
```

**Candidate generator.** Place `model_sfp_dtlr.py` in `cgsr/candidate/`. It is distributed separately; see `cgsr/candidate/README.md`.

## Datasets

PASCAL VOC 2012, PASCAL Context-59 (official `59_labels.txt` order), ADE20K, COCO-Stuff-27 and Cityscapes (gtFine). The expected layouts and label conventions are in `docs/DATASETS.md`. No dataset files are included.

## Pretrained weights

Base predictor weights (the author-released ReCLIP++ checkpoints) and CLIP text embeddings. Sizes, SHA-256 values, the official download links and the generation command are in `docs/WEIGHTS.md`. Verify them with:

```bash
python tools/verify_assets.py --base-code-root third_party/base_predictor --weights-dir weights
```

## Quick start

A smoke run on 20 VOC images (this checks the setup only; the output is **not** a result):

```bash
python tools/eval_cgsr.py --dataset voc --base-code-root third_party/base_predictor \
    --data-root /path/to/VOCdevkit/VOC2012 --checkpoint weights/voc.pth \
    --text-embedding weights/text/voc_ViT16_clip_text.pth \
    --output-dir outputs/voc_smoke --limit 20 --bootstrap 0
```

## Evaluation

```bash
python tools/eval_cgsr.py --dataset <voc|context59|ade20k|coco_stuff27|cityscapes> \
    --base-code-root third_party/base_predictor --data-root /path/to/dataset \
    --checkpoint weights/<ckpt>.pth --text-embedding weights/text/<embedding>.pth \
    --output-dir outputs/<dataset>
```

A single forward call per image yields all three predictions (Base, Candidate, CGSR). The output directory receives two files:
* `result.json`: mIoU, the three paired differences with 95 % bootstrap CIs, input hashes and environment.
* `per_image_iu.npz`: per-image intersections and unions.

`python tools/eval_cgsr.py --help` lists all options.

## Reproducing paper results

```bash
bash scripts/eval.sh voc          /path/to/VOCdevkit/VOC2012
bash scripts/eval.sh context59    /path/to/VOCdevkit/VOC2010
bash scripts/eval.sh ade20k       /path/to/ADEChallengeData2016
bash scripts/eval.sh coco_stuff27 /path/to/COCOStuff27
bash scripts/eval.sh cityscapes   /path/to/cityscapes
```

On Windows use `scripts\eval.ps1`. A full run compares its mIoU with the frozen paper values and prints `paper parity ... PASS` when they agree to 1e-9. Protocol, seeds and determinism: `REPRODUCIBILITY.md`.

**Caveats:**
* **Text embeddings.** Exact parity needs the exact text-embedding files listed in `docs/WEIGHTS.md`. Regenerated embeddings match bit-for-bit only for VOC, Context-59, ADE20K and COCO-Stuff-27, and only on the paper's GPU environment.
* **Cityscapes.** A regenerated Cityscapes embedding differs slightly from the paper file. `tools/eval_cgsr.py` then stops with a hash mismatch. With `--skip-hash-check` it runs, but the last digits of the Cityscapes numbers can differ.
* **Hardware.** Other GPUs and library versions can also change the last digits.
* **This release's code path.** End-to-end GPU parity on the unmodified upstream checkout is not yet verified (`REPRODUCIBILITY.md`, provenance note).

## Expected results

mIoU (%) on the full validation sets, author checkpoints. The values in brackets are 95 % paired-bootstrap CIs, in mIoU points.

| Dataset | Base | Candidate | CGSR | CGSR − Base | CGSR − Candidate |
|---|---:|---:|---:|---|---|
| VOC 2012 | 85.3617 | 85.8969 | 85.7780 | +0.4163 [+0.2910, +0.5456] | −0.1189 [−0.3565, +0.0790] |
| Context-59 (H1) | 35.9988 | 30.8701 | 36.7901 | +0.7913 [+0.6797, +0.9004] | +5.9200 [+5.6127, +6.2314] |
| ADE20K | 16.4326 | 14.8911 | 16.4626 | +0.0300 [−0.0668, +0.1660] | +1.5715 [+1.2696, +1.9322] |
| COCO-Stuff-27 | 23.8842 | 23.9680 | 24.6723 | +0.7880 [+0.7359, +0.8429] | +0.7043 [+0.6184, +0.7881] |
| Cityscapes (gtFine) | 26.4569 | 24.9073 | 27.7328 | +1.2760 [+1.1344, +1.4180] | +2.8255 [+2.5478, +3.0859] |

* On ADE20K, the gain over Base is not significant.
* On VOC, CGSR is slightly below the Candidate; that difference is not significant either.
* All CIs here use one bootstrap seed (20260923), which `tools/eval_cgsr.py` reproduces. The paper prints VOC and ADE20K CIs from an earlier resampler (seed 20260918). They differ by at most 0.0083 mIoU points (`docs/RESULTS.md` §1).

`docs/RESULTS.md` has the following:
* transition (Retention / Recovery) statistics;
* mechanism controls (uniform weights, label plurality, random rollback);
* window-size and affinity ablations;
* the bootstrap protocol.

## Efficiency

Cityscapes, author checkpoint, RTX 5090, batch 1, FP32, 448×896 input, 20 warm-up iterations, 5 repeats:

| | Base | Candidate | CGSR (end-to-end) | CGSR operator only |
|---|---:|---:|---:|---:|
| ms / image | 26.641 | 34.317 | 61.651 | 1.090 |

The CGSR operator itself takes about 1 ms. The current implementation runs the backbone twice, so end-to-end CGSR costs 2.31× Base. CGSR adds no parameters.

## Limitations

These are summarised from `docs/LIMITATIONS.md`:
* Evaluated with one Base/refiner family.
* Two full backbone passes per image.
* Whole-logit rollback and pre-refinement context were not compared with alternatives.
* CLIP-feature affinity is not shown to matter: uniform or RGB weights and a label-plurality rule perform about as well.
* 5×5 is a fixed choice, not an optimum.
* The Candidate generator applies VOC-specific structure-class indices on every dataset.
* Gains are not uniform (VOC vs Candidate, ADE20K vs Base).

## Citation

See `CITATION.cff`. The paper metadata (authors, venue) is not final. Please also cite ReCLIP++:

```bibtex
@article{wang2025reclippp,
  title   = {ReCLIP++: Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation},
  author  = {Wang, Jingyun and Kang, Guoliang},
  journal = {International Journal of Computer Vision},
  volume  = {133},
  number  = {11},
  pages   = {7946--7968},
  year    = {2025}
}
```

## Acknowledgements

This work builds on ReCLIP++ by Jingyun Wang and Guoliang Kang, and on OpenAI CLIP. The COCO-Stuff-27 class grouping follows the IIC / PiCIE / STEGO line of work.

## Third-party code

This repository contains no ReCLIP++ source code. The base-predictor code is cloned from its own repository (upstream ReCLIP++) at a pinned commit, and the base predictor weights (ReCLIP++ checkpoints) are downloaded from the authors' links. The role names used in the code (`FrozenBasePredictor`, `FixedCandidateGenerator`, `shared_postprocess`) are names for these third-party components, not new implementations of them. Third-party components and their licenses: `THIRD_PARTY_NOTICES.md`.

## License

**Not yet chosen; see `LICENSE_PENDING.md`.** Until a `LICENSE` file is added, no license is granted for the code in this repository.
