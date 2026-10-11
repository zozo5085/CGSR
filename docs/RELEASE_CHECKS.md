# Release checks

These checks ran on 2026-10-09 and were repeated on 2026-10-10 after the role-name refactor (see "Role-name refactor" below). They ran before the first commit, on a **clean-clone simulation**:
* a copy of this repository;
* the upstream ReCLIP++ code extracted at commit `46298007fcfa75b0ca41c3a9a8420b2dfa800475` (`git archive`, unmodified);
* the separately distributed Candidate generator.

All checks ran on the CPU, with the GPU hidden. The only model inference is the CPU A/B source-parity test (C4b); its CPU outputs are compared with each other, never with paper numbers. The GPU end-to-end check is listed as not run at the end.

## Code and configuration

| ID | Check | Result |
|---|---|---|
| C1 | `python -m compileall cgsr tools` | PASS |
| C2 | Base-code files imported at run time (`model/model.py`, `utils/preprocess.py`, `config/configs.py`, `utils/cls_token.pt`) equal the pinned commit (`cgsr/upstream.py::verify_base_code`) | PASS (0 mismatches) |
| C3 | All five `configs/*/*.yaml` parse with the **unmodified** upstream `cfg_from_file`. `NAME`, `NUM_CLASSES`, `PD`, `REDUCE_ZERO_LABEL` and `SCALE` match the protocol, and no `SFP_DTLR` section is defined. | PASS (5/5) |
| C3b | Each config equals its frozen run config except `DATASET.DATAROOT`, `DATASET.TEXT_WEIGHT`, `LOAD_PATH` and `SAVE_DIR` (key-by-key YAML comparison at build time) | PASS (5/5) |
| C4 | Code equivalence with the audited sources (Python AST, docstrings and function names ignored): `local_class_support` (`cgsr/verifier.py`) equals the audited `local_Q`. `shared_postprocess`, `raw_label`, `intersection_union`, `miou` and `bootstrap` equal the functions that produced the frozen numbers. The Candidate file is byte-identical to the audited file (SHA-256 `7877892b…`). The refactored `cgsr/model_cgsr.py` is no longer AST-identical to the audited file (role names, verifier extracted); its numerical equality is tested by C4b. | PASS (7/7) |
| C4b | **CPU A/B source parity.** The audited pre-refactor `model_cgsr.py` and the refactored `CGSRPipeline` are both built on the same unmodified base code, Candidate module and author checkpoint. 5 validation images of each of the 5 datasets are run through both, on CPU. All 18 checks per image are bit-identical (`torch.equal`): 14 state tensors (base/candidate/final logits, projected features, penultimate attention, probabilities, local support, classes, supports, gain, accept mask, final class), the output, and the 3 post-processed label maps (`shared_postprocess` vs the audited `final_prediction`). The state dicts are identical, and the reference is deterministic across repeated runs. The images cover token grids of 588–1,568 tokens (Top-K active above 800) and 26–248 accepted tokens per image. | PASS (25/25 images, 450/450 checks) |
| C5 | The Candidate generator's built-in defaults (25 attributes) equal `cgsr/frozen.py::CANDIDATE_ATTRIBUTES`. Upstream `configs.py` defines no `SFP_DTLR`, so these defaults are what runs. | PASS (0 mismatches) |
| C6 | The Context-59 name-based LUT (`cgsr/context_h1.py`) equals the two-step LUT used for the frozen table (alphabetical → internal order → official order) | PASS (bijection; 54 of 59 ids differ from identity) |
| C7 | `cgsr.upstream.activate_base_code()` loads `CGSRPipeline` on the unmodified base code with the class chain `CGSRPipeline` → `FixedCandidateGenerator` (`cgsr.candidate.model_sfp_dtlr`) → `FrozenBasePredictor` (upstream `model.model`) | PASS |
| C8 | Validation lists built by `cgsr/data.py` have the protocol image counts (1,449 / 5,105 / 2,000 / 5,000 / 500), and their order equals the image order of the frozen runs | PASS (5/5) |

## Numerical parity (no GPU)

**C9.** The frozen paper-facing **prediction PNGs** (Base / Candidate / CGSR, 14,054 images × 3) were re-scored with this repository's validation lists, label loading, IoU and bootstrap code. Context-59 predictions were mapped to the official order. The scores were then compared with the paper-facing results.

| Dataset | Base / Candidate / CGSR mIoU (raw) | max abs diff vs paper | max abs diff of the 3 CIs and points vs paper (seed 20260923) |
|---|---|---:|---:|
| VOC 2012 | 0.8536171478072758 / 0.8589691435052641 / 0.8577803683964914 | 0.0 | 0.0 |
| Context-59 (H1) | 0.3599881136274712 / 0.30870109844639604 / 0.36790070114319684 | 0.0 | 0.0 |
| ADE20K | 0.16432640866588497 / 0.14891131688957823 / 0.16462632184272766 | 0.0 | 0.0 |
| COCO-Stuff-27 | 0.23884218192073192 / 0.2396795170515858 / 0.24672267184700136 | 0.0 | 0.0 |
| Cityscapes (gtFine) | 0.26456859013662504 / 0.2490733769613716 / 0.2773282291242289 | 0.0 | 0.0 |

Result: **PASS**. The evaluation half of the pipeline (lists, labels, metric, bootstrap, Context remap) reproduces the paper numbers exactly.

## Command-line behaviour (clean clone)

| ID | Situation | Result |
|---|---|---|
| N1 | `tools/eval_cgsr.py --help` | exit 0 |
| N2 | missing required arguments | exit 2, argparse lists them |
| N3 | missing dataset root | exit 1: "--data-root not found … see docs/DATASETS.md" |
| N4 | missing checkpoint | exit 1: "--checkpoint not found … See docs/WEIGHTS.md" |
| N5 | wrong file as checkpoint | exit 1: SHA-256 mismatch message, with a pointer to `--skip-hash-check` |
| N6 | missing base-code checkout | exit 1: setup instructions with the pinned commit |
| N7 | Candidate generator absent | exit 1: "Candidate generator not found … see cgsr/candidate/README.md" |
| N8 | `tools/verify_assets.py --base-code-root …` without the Candidate | exit 1, SOME CHECKS FAILED |
| N9 | all inputs valid, no CUDA device | exit 1: "A CUDA GPU is required …" |
| N10 | `tools/verify_assets.py --base-code-root …` with the Candidate | exit 0, ALL CHECKS PASSED |
| N10b | `tools/verify_assets.py --base-code-root … --skip-candidate` (what the setup scripts run) without the Candidate | exit 0, ALL CHECKS PASSED |
| N13 | the pre-refactor flag `--reclip-root` | rejected by argparse (exit 2) |
| N11, N12 | `tools/make_text_embeddings.py --help`, `tools/convert_coco_stuff27_labels.py --help` | exit 0 |
| T1 | Re-saved text embeddings (identical tensors, different file hash) | accepted as `match (tensor content)`; a perturbed tensor is rejected as `MISMATCH` |

No failing command created an output directory.

## Weights

* Author checkpoints: the SHA-256 of all five files equals the values in `docs/WEIGHTS.md`. All five are plain state dicts (614 tensors, `module.` prefix) that load with `torch.load(..., weights_only=True)`. Each contains `module.text_encoder.prompt_token`, which the unmodified upstream `TextEncoder` reads.
* Text embeddings: the file SHA-256 and tensor-content SHA-256 of the five paper files are recorded in `cgsr/frozen.py`. The upstream module provides 85 prompt templates.

## Repository hygiene

The scans cover every file in the repository: 43 files plus `docs/RELEASE_CHECKS.md`, `RELEASE_MANIFEST.md` and `RELEASE_REPORT.md`. Totals are in `RELEASE_REPORT.md` §14–§16.

| Scan | Result |
|---|---|
| Secrets (GitHub / Hugging Face / OpenAI-style / AWS key patterns, private-key blocks, password / token / API-key assignments, wandb keys) | 0 hits |
| E-mail addresses, user names, machine names | 0 hits |
| Absolute local paths (drive letters, `/home`, `/Users`, `/mnt`) and private folder names | 0 hits |
| Files > 10 MB / > 50 MB / > 100 MB | none / none / none |
| Binary files (`.pth`, `.pt`, `.ckpt`, `.npz`, `.npy`, images, PDFs, archives) | none |

## Role-name refactor (2026-10-10)

Public-facing code now uses architecture-neutral role names: `FrozenBasePredictor`, `FixedCandidateGenerator`, `SelectiveRollbackVerifier`, `CGSRPipeline` and `shared_postprocess`. The command-line flags changed as well: `--base-code-root`, `--allow-base-code-mismatch` and `BASE_CODE_ROOT`, with default checkout `third_party/base_predictor`. Old → new names are listed in `RELEASE_REPORT.md` §5. The refactor kept every attribution (`THIRD_PARTY_NOTICES.md`, "Role names used in the code").

| Re-run after the refactor | Result |
|---|---|
| C1–C3, C5–C8 | PASS (same results as before) |
| C4 (AST, adapted as described) | PASS (7/7) |
| C4b (CPU A/B source parity, new) | PASS (450/450 bit-identical) |
| N1–N13 command-line tests with the new flags | PASS |
| C9 (re-score of the frozen predictions) | still valid without re-running: `raw_label`, `intersection_union`, `miou`, `bootstrap`, `cgsr/data.py` logic and the configs are unchanged (C4, C8, C3b) |
| Hygiene scans | 0 hits (see below) |

## Verifier split (2026-10-11)

`local_class_support` and `SelectiveRollbackVerifier` were moved verbatim from `cgsr/model_cgsr.py` into `cgsr/verifier.py`, which imports only PyTorch. Both definitions are AST-identical before and after the move. `cgsr/model_cgsr.py` keeps the integration: the `FrozenBasePredictor` and `FixedCandidateGenerator` bindings and `CGSRPipeline`. `cgsr/__init__.py` exports the two names lazily.

| Check | Result |
|---|---|
| `python -c "from cgsr import SelectiveRollbackVerifier, local_class_support; print('PASS')"` in this repository (no `third_party/base_predictor`, no `cgsr/candidate/model_sfp_dtlr.py`) | PASS |
| `tests/test_standalone_import.py`: a clean copy of `cgsr/` without the Candidate file, fresh interpreter. It checks that the base code is not importable, that no integration module is loaded, and that the verifier runs on random tensors. | PASS (also from the clean-clone copy, where the base code and Candidate exist on disk) |
| C1–C8 re-run | PASS |
| Three-way CPU A/B source parity, 5 images × 5 datasets: current `CGSRPipeline` vs the v2 `model_cgsr.py` vs the audited original. The current pipeline uses `cgsr.verifier`. | PASS: 900/900 bit-identical (450 vs v2, 450 vs audited) |

## Fresh-context verification

An independent reviewer that had not seen how the release was built checked the tree adversarially:
* **Numbers:** 355 numbers in the docs and tables were compared with the frozen result files, with 0 mismatches. The bootstrap reproduces the frozen CIs bit-exactly from the frozen per-image I/U.
* **Method description:** `docs/METHOD.md` was checked against the code.
* **Claims:** no forbidden claims were found.
* **Configs:** they differ from the frozen configs only in the four path keys.
* **Legal / privacy:** no copied upstream module, no private paths, e-mails or secrets. The STEGO notice matches the upstream license text.

Its verdict was **PARTIAL**. It raised three major points, all resolved:
* The end-to-end GPU parity on the unmodified upstream checkout is unmeasured. Disclosed in `REPRODUCIBILITY.md` (provenance note) and `README.md`; the check is still not run.
* Three cited files were missing. They now exist.
* Cityscapes parity needs the exact text embedding. Now stated in `README.md`.

All ten minor points (references, wording, attribution detail, setup-script exit code) were fixed.

## Not run

| Check | Why | How to run |
|---|---|---|
| GPU end-to-end parity: run `tools/eval_cgsr.py --save-predictions --limit 20` on all five datasets and compare the PNGs with the frozen predictions pixel by pixel | needs the GPU (about 1–2 minutes) and approval | a ready-made script is kept with the release staging materials (outside the repository) |
| Full-validation re-run | hours of GPU time, not needed for the release | `scripts/eval.sh <dataset> <root>` |
| `tools/make_text_embeddings.py` on GPU | needs the GPU | see `docs/WEIGHTS.md` |

Without the GPU check, end-to-end equality rests on three facts established above:
* identical code: C2, C4, C5 and C7;
* identical inputs: checkpoints, embeddings, configs (C3b) and lists (C8);
* identical evaluation: C9.

The one difference in the code path is a fork-only change to how the prompt token is loaded at model construction. The full checkpoint state dict overwrites that token afterwards, and `tools/eval_cgsr.py` asserts that every checkpoint tensor equals the loaded model state.
