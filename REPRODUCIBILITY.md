# Reproducibility

This document states exactly which protocol produced the paper-facing numbers and how to repeat it. The evaluation is deterministic given fixed inputs. The paired bootstrap that produces the confidence intervals is a seeded random procedure; section 6 keeps the two apart.

## 1. Paper protocol

| Item | Value |
|---|---|
| Base predictor (`FrozenBasePredictor`) | ReCLIP++ ViT-B/16 (Wang & Kang), author-released checkpoint per dataset (`docs/WEIGHTS.md`) |
| Base code | the upstream ReCLIP++ repository at commit `46298007fcfa75b0ca41c3a9a8420b2dfa800475` (`scripts/setup_upstream.*`), used unmodified by this release. The paper numbers were produced with a research fork of this commit; see the provenance note below. |
| Candidate (`FixedCandidateGenerator`) | existing SFP + DTLR refinement with fixed settings (`docs/METHOD.md` §3; `cgsr/frozen.py::CANDIDATE_ATTRIBUTES`) |
| CGSR (`SelectiveRollbackVerifier`) | 5×5 local class support, τ = 1, strict `g > 0`, whole-logit selection, native token grid (`cgsr/verifier.py`; integrated by `CGSRPipeline` in `cgsr/model_cgsr.py`) |
| Inference | single scale, batch size 1, FP32, no test-time augmentation, one `CGSRPipeline` forward call per image returning `base_logits` (`L0`), `candidate_logits` (`Lr`) and `final_logits` (`L_final`) |
| Evaluator | `shared_postprocess`: class-presence filter `PD` → bilinear to model input → bilinear to label size → softmax → argmax (`cgsr/evaluation.py`) |
| Metric | mIoU from per-image intersections/unions summed over the full validation split; empty-union classes count 0; ignore label 255 |
| Datasets | full validation splits: VOC 2012 (1,449), Context-59 H1 (5,105), ADE20K (2,000), COCO-Stuff-27 (5,000), Cityscapes gtFine (500) (`docs/DATASETS.md`) |

**Provenance note (fork vs unmodified upstream).** The paper-facing predictions were generated with a research fork of the pinned upstream commit. On the evaluation path, the fork changes three things:
1. How `TextEncoder` loads the prompt token at construction. The full checkpoint load afterwards overwrites that token, and `tools/eval_cgsr.py` asserts every checkpoint tensor.
2. Added config defaults for the Candidate generator. They equal the generator's built-in defaults, which this release asserts at run time.
3. The format of the Cityscapes list file. This release builds its own lists.

The Candidate and CGSR code, configs, inputs and evaluation are identical, and the evaluation re-scores the paper predictions exactly (`docs/RELEASE_CHECKS.md`). However, **an end-to-end GPU run of this release on the unmodified upstream checkout has not yet been compared with the paper predictions** (`docs/RELEASE_CHECKS.md`, "Not run"). Until it has, bit-exact equality of the predictions is argued from the code, not measured.

## 2. Checkpoints and text embeddings

Every required file, with its SHA-256, is listed in `docs/WEIGHTS.md` and `weights/CHECKSUMS.sha256`.
* `tools/eval_cgsr.py` refuses a checkpoint or text embedding whose SHA-256 differs from the paper-facing file, unless `--skip-hash-check` is given.
* It also asserts that every checkpoint tensor equals the loaded model state.
* The base predictor's learned reference-prompt token is stored inside each checkpoint (`text_encoder.prompt_token`).

## 3. Configs

`configs/<dataset>/<dataset>_cgsr.yaml` equal the frozen run configs except for four path keys: `DATASET.DATAROOT`, `DATASET.TEXT_WEIGHT`, `LOAD_PATH` and `SAVE_DIR`. The check was a key-by-key YAML comparison (`docs/RELEASE_CHECKS.md`). The path keys are set from the command line.

Inference settings per dataset:

| Dataset | Upstream name | Classes | Short side (`SCALE`) | `PD` | `reduce_zero_label` |
|---|---|---:|---:|---:|---|
| VOC | `voc` | 20 | 336 | 1.0 | yes |
| Context-59 | `context` | 59 | 448 | 0.85 | yes |
| ADE20K | `ade` | 150 | 512 | 0.5 | yes |
| COCO-Stuff-27 | `stuff` | 27 | 512 | 0.95 | no |
| Cityscapes | `cityscapes` | 19 | 448 | 0.5 | no |

`SCALE = [2048, s]` resizes each image by `min(2048 / long side, s / short side)`, keeping the aspect ratio. In practice the short side becomes `s`.

The Candidate generator and CGSR have no YAML keys. Their settings are constants, asserted at run time.

## 4. Commands

```bash
# 0. environment (see section 8)
pip install -r requirements.txt

# 1. base-predictor code (upstream ReCLIP++) at the pinned commit
bash scripts/setup_upstream.sh third_party/base_predictor

# 2. weights (docs/WEIGHTS.md), then verify
python tools/verify_assets.py --base-code-root third_party/base_predictor --weights-dir weights

# 3. one dataset (GPU required); repeat for each dataset
python tools/eval_cgsr.py --dataset voc --base-code-root third_party/base_predictor \
    --data-root /path/to/VOCdevkit/VOC2012 --checkpoint weights/voc.pth \
    --text-embedding weights/text/voc_ViT16_clip_text.pth --output-dir outputs/voc
```

`--limit N` evaluates the first N images only. Its output is labelled `SMOKE ONLY -- NOT A SCIENTIFIC RESULT` and must not be compared with the paper.

## 5. Expected results

The expected mIoU values are stored in `cgsr/frozen.py::EXPECTED_MIOU`. A full run prints `paper parity (|diff| <= 1e-9): PASS` when the result matches them. The values in percent are:

| Dataset | Base | Candidate | CGSR |
|---|---:|---:|---:|
| VOC 2012 | 85.3617 | 85.8969 | 85.7780 |
| Context-59 (H1) | 35.9988 | 30.8701 | 36.7901 |
| ADE20K | 16.4326 | 14.8911 | 16.4626 |
| COCO-Stuff-27 | 23.8842 | 23.9680 | 24.6723 |
| Cityscapes (gtFine) | 26.4569 | 24.9073 | 27.7328 |

Confidence intervals, transitions, controls and efficiency: `docs/RESULTS.md`.

## 6. Determinism vs bootstrap randomness

* **Inference** has no sampling. The same inputs, code, library versions and GPU give identical logits and predictions. The paper-facing predictions were re-generated twice on the full validation sets and were bit-identical each time (14,054 images × 3 methods).
* **Bootstrap.**
  * Each replicate draws `n` image indices with replacement (`numpy.random.RandomState(seed).randint(0, n, n)`).
  * Base, Candidate and CGSR share the same indices (paired).
  * The statistic is the dataset mIoU recomputed from the resampled per-image intersections and unions.
  * B = 10,000 replicates, seed **20260923**; the 95 % percentile interval is reported, and a BCa interval is stored as well (`cgsr/evaluation.py::bootstrap`).
  * The interval depends on the seed and on the image order. The image order is fixed: list order for VOC and Context, sorted file names for ADE20K, COCO-Stuff-27 and Cityscapes.
* The paper's VOC and ADE20K intervals came from an earlier resampler with seed 20260918. They differ by at most 0.0083 mIoU points from the seed-20260923 intervals `tools/eval_cgsr.py` produces (`docs/RESULTS.md` §1).
* No checkpoint was trained for this work. Each dataset has one author checkpoint, so training-seed variance is not part of any interval.

## 7. Known limitations of reproduction

* **Hardware and library versions.** Bit-exact parity is established only for the environment in section 8. Another GPU, driver, CUDA, cuDNN, PyTorch or OpenCV version may change a few argmax decisions and the last digits of mIoU.
* **CPU-only execution is not supported.** The upstream code loads CUDA-saved tensors, and the paper numbers are CUDA results.
* **Candidate generator.** It must be obtained separately (`cgsr/candidate/README.md`). Without it, nothing runs.
* **Ground-truth conversions.** COCO-Stuff-27 labels are converted locally (`tools/convert_coco_stuff27_labels.py`). Cityscapes `labelTrainIds` come from `cityscapesScripts`. Context-59 label PNGs come from the standard 59-class conversion.
* **Text embeddings.** Regenerating them with CLIP can differ in the last bits. Use the files with the listed SHA-256 (`docs/WEIGHTS.md`).

## 8. Environment used for the paper-facing numbers

| Component | Version |
|---|---|
| OS | Windows 10 (10.0.19045) |
| GPU / driver | NVIDIA GeForce RTX 5090 (32 GB), driver 616.92 (recorded for the efficiency run) |
| Python | 3.10.19 |
| PyTorch / torchvision | 2.11.0.dev20260214+cu128 / 0.26.0.dev20260214+cu128 (nightly builds) |
| CUDA / cuDNN | 12.8 / 9.10.2 |
| numpy, opencv-python, Pillow | 2.2.6, 5.0.0.93, 12.0.0 |
| PyYAML, easydict, ftfy, regex, tqdm | 6.0.3, 1.13, 6.3.1, 2026.1.15, 4.67.3 |
| OpenAI CLIP | git commit `dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1` |

No RTX 5090 is needed to run the code. Any CUDA GPU supported by your PyTorch build should work, but bit-exact agreement with the table is only established for this environment. The upstream README lists Python 3.9, PyTorch 2.1.0 and CUDA 11.8 for ReCLIP++ itself.
