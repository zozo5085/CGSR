# CGSR GitHub Release Report

Prepared 2026-10-09. The repository is initialised (`git init`) but nothing is staged or committed, no remote is configured, and nothing was pushed or uploaded.

Path aliases used in this report:
* `[fork]`: the research fork of ReCLIP++.
* `[workspace]`: the research workspace.
* `[cgsr-workspace]`: the CGSR sub-folder of the research workspace.
* `[checkpoints]`: the local copies of the author checkpoints.
* `[held-back]`: the staging folder kept outside this repository.

The alias-to-local-path mapping is in the release inventory, which is also kept outside the repository.

## 1. Status

**READY WITH USER DECISIONS.**

The tree is complete and CPU-verified. It contains no third-party source file, no weights, no data, and no private paths or secrets.

* **Private GitHub push:** safe now.
* **Public release:** still **blocked** by two unresolved decisions:
  * the Candidate generator's redistribution status (D1). Until it is cleared, others cannot run the full pipeline; the CGSR core in `cgsr/verifier.py` can be used on its own.
  * the repository license (D2). Without a `LICENSE`, no reuse rights are granted.

## 2. Source Roots Audited

| Alias | Content used |
|---|---|
| `[fork]` | `model/model_cgsr.py`, `model/model_sfp_dtlr.py`, `model/model.py`, `utils/*`, `config/*`, `tools/*`, `text/*`, git history (remote `origin` = upstream ReCLIP++) |
| `[workspace]` | CGSR sub-folder `[cgsr-workspace]` (results lock, claim firewall, final audits, architecture/code audit, transition analysis, mechanism challenge, figures), `experiments/*` (frozen runners, scores, predictions, efficiency) |
| `[checkpoints]` | five author-released ReCLIP++ checkpoints + download record |
| upstream | https://github.com/dogehhh/ReCLIP at `46298007fcfa75b0ca41c3a9a8420b2dfa800475` (local git objects; GitHub API: `license: null`) |
| dataset roots | the five validation sets (counted, not copied) |

## 3. Files Included

There are 49 repository files. `RELEASE_MANIFEST.md` lists each of them with its source and release SHA-256, except `RELEASE_MANIFEST.md` itself (a file cannot record its own hash). The 49 files are:
* 14 Python files (9 in `cgsr/`, 4 in `tools/`, 1 in `tests/`);
* 5 configs;
* 4 scripts;
* 14 Markdown documents;
* 5 result tables;
* 3 files under `weights/` (checksums and placeholders);
* 4 metadata files (`CITATION.cff`, `requirements.txt`, `.gitignore`, `.gitattributes`).

Copied or modified sources:
* `cgsr/model_cgsr.py`: docstring path removed.
* `tools/convert_coco_stuff27_labels.py`: local default paths removed.
* `configs/*.yaml`: path keys replaced.

Everything else was written for the release. Where code was moved, the function bodies are AST-identical to their sources (`docs/RELEASE_CHECKS.md` C4).

## 4. Files Excluded

| Excluded | Class | Where it is accounted for |
|---|---|---|
| Upstream ReCLIP++ code (`model.py`, `preprocess.py`, `configs.py`, `test.py`, `test_mIoU.py`, `prompt_engineering.py`, `cls_token.pt`) | EXCLUDE_THIRD_PARTY_LICENSE (no license) | cloned by the user at the pinned commit |
| Candidate generator `model_sfp_dtlr.py` | NEEDS_USER_DECISION (provenance) | staged in `[held-back]`; gitignored here |
| Author checkpoints, text embeddings, CLIP weights | EXCLUDE_WEIGHT | `docs/WEIGHTS.md` |
| Locally trained checkpoints | EXCLUDE_PRIVATE | never presented as author weights |
| Datasets | EXCLUDE_DATASET | `docs/DATASETS.md` |
| Prediction trees, per-image caches, ISCAS per-image tensors | EXCLUDE_LARGE_RESULT | — |
| Mixed / superseded result files (results lock, pre-relock CSV, multi-scale JSON, LOCAL efficiency) | EXCLUDE_PRIVATE | `docs/RESULTS.md` §6 |
| Abandoned model branches, other configs, third-party baselines, other projects | EXCLUDE_TEMP / EXCLUDE_THIRD_PARTY_LICENSE | inventory |
| Decision packets, handoffs, reviews, audits, GPU logs, prompts, backups, scratch, manuscript | EXCLUDE_PRIVATE / EXCLUDE_TEMP | inventory |
| Paper figures (architecture, Candidate, qualitative, failure cases) | NEEDS_USER_DECISION (they embed VOC / ADE20K / Cityscapes photographs) | D5 |
| Frozen runner scripts (`export_formal.py`, `run_p1.py`, `eval_author_checkpoint.py`, parity helper) | EXCLUDE_DUPLICATE / re-implemented | `tools/eval_cgsr.py`, `cgsr/evaluation.py` |

## 5. Core Code Trace

```text
tools/eval_cgsr.py
 ├─ cgsr/data.py           validation lists (order = frozen runs; C8)
 ├─ cgsr/upstream.py       verify_base_code (pinned base-code files); activate_base_code → CGSRPipeline
 ├─ base config: upstream config.configs.cfg_from_file(configs/<ds>/<ds>_cgsr.yaml); cfg.LOAD_PATH = --checkpoint
 ├─ clip.load("ViT-B/16") → CGSRPipeline(cfg, clip, rank=0) → load_state_dict(base predictor weights)
 │    guards: no missing/unexpected keys; all tensors equal; 25 Candidate settings == frozen
 ├─ per image: base-code val_preprocess → pipeline(..., return_debug=True)
 │    CGSRPipeline: FrozenBasePredictor pass (return_feat) → base_logits, projected_features
 │                  FixedCandidateGenerator pass → candidate_logits (SFP + DTLR)
 │                  SelectiveRollbackVerifier (cgsr/verifier.py) → final_logits
 ├─ cgsr/evaluation.shared_postprocess (PD filter, resize ×2, softmax, argmax) for each of the three
 ├─ Context-59: alphabetical → official 59_labels order (cgsr/context_h1.py)
 └─ cgsr/evaluation: per-image I/U → mIoU; paired bootstrap B = 10,000, seed 20260923
```

Old → new names (role-name refactor, 2026-10-10). Numerically unchanged: the CPU A/B source-parity test (C4b) gave 450/450 bit-identical checks.

| Old | New | Kind |
|---|---|---|
| `RECLIPPP` (CGSR class in `model_cgsr.py`) | `CGSRPipeline` | class (this work) |
| inline gate in `forward` | `SelectiveRollbackVerifier` | class (this work; same operations) |
| `local_Q` | `local_class_support` | function (AST-identical body) |
| verifier and `local_class_support` defined in `cgsr/model_cgsr.py` | moved verbatim to `cgsr/verifier.py` (PyTorch only); `from cgsr import SelectiveRollbackVerifier, local_class_support` (2026-10-11) | module boundary; CPU A/B 900/900 bit-identical vs v2 and audited |
| `_SFPDTLRRECLIPPP` / `model.model_sfp_dtlr.RECLIPPP` | `FixedCandidateGenerator` (`cgsr/candidate/__init__.py`) | alias of the third-party class; file unchanged |
| upstream `model.model.RECLIPPP` | `FrozenBasePredictor` | alias of the upstream class; not redistributed |
| `ReCLIP` re-export | removed | was only needed by the fork's `tools/test.py` loader |
| `final_prediction` | `shared_postprocess` | function (AST-identical body) |
| debug keys `L0, F_patch, A_cache, Lr, L_final, P0, Q, base_cls, proposal_cls, q_old, q_new, CSG, gate, final_cls` | `base_logits, projected_features, penultimate_attention, candidate_logits, final_logits, base_probs, local_support, base_class, candidate_class, support_base, support_candidate, support_gain, accept, final_class` | `states` dict keys |
| `upstream.verify_upstream` / `upstream.activate` | `verify_base_code` / `activate_base_code` | functions |
| `frozen.UPSTREAM_*`, `DATASETS[...]["upstream_name"]` | `frozen.BASE_CODE_*`, `["config_name"]` | constants (values identical) |
| `--reclip-root`, `--allow-upstream-mismatch`, `RECLIP_ROOT`, `third_party/ReCLIP` | `--base-code-root`, `--allow-base-code-mismatch`, `BASE_CODE_ROOT`, `third_party/base_predictor` | CLI / environment / default path |
| `build_model`, `model`, `debug` (in `tools/eval_cgsr.py`) | `build_pipeline`, `pipeline`, `states` | local names |
| result.json key `upstream` | `base_code` | output field |

Frozen settings, verified against the code (`docs/METHOD.md`):
* **Candidate:** reliability score `A_cls,i − A_i,i` from layer 11 of 12; low-confidence eligibility `max softmax(10·L0) < 0.97`; token Top-K = 800; `LOGIT_BETA` 0.55 with the 5×5 proxy (λ = 2.0); `DTLR_BETA` 1.20 (σ_s = 70, σ_r = 1.5); structure classes [4, 8, 10].
* **CGSR:** 5×5 window; τ = 1; strict `g > 0`; whole-logit selection at token resolution.

Core source SHA-256 (current = audited):
* `model.py`: `f0fc1e51…` (fork); the upstream file used is `4a66e902…`.
* `model_sfp_dtlr.py`: `7877892b…`.
* `model_cgsr.py`: `972d00bf…` (audited research file). The release copy is now the role-name refactor; it is not byte-identical, and equality is shown by C4b.

## 6. Configs Included

| Config | Dataset (upstream name) | Classes | Short side | PD | reduce_zero_label | Derived from |
|---|---|---:|---:|---:|---|---|
| `configs/voc/voc_cgsr.yaml` | `voc` | 20 | 336 | 1.0 | yes | frozen `voc_official_ori.yaml` |
| `configs/context59/context59_cgsr.yaml` | `context` | 59 | 448 | 0.85 | yes | frozen `context_test_ori_cfg.yaml` |
| `configs/ade20k/ade20k_cgsr.yaml` | `ade` | 150 | 512 | 0.5 | yes | frozen `ade_official_ori.yaml` |
| `configs/coco_stuff27/coco_stuff27_cgsr.yaml` | `stuff` | 27 | 512 | 0.95 | no | frozen `coco_official_ori.yaml` |
| `configs/cityscapes/cityscapes_cgsr.yaml` | `cityscapes` | 19 | 448 | 0.5 | no | frozen `cityscapes_test_verified.yaml` |

Only `DATASET.DATAROOT`, `DATASET.TEXT_WEIGHT`, `LOAD_PATH` and `SAVE_DIR` differ from the frozen files (C3b).

## 7. Dataset Instructions

`docs/DATASETS.md` covers layouts, label protocols, ignore and `reduce_zero_label` handling, the Context-59 H1 order, the COCO-Stuff-27 conversion (`tools/convert_coco_stuff27_labels.py`), and Cityscapes gtFine `labelTrainIds`.

Validation counts were checked on the paper copies: VOC 1,449, Context 5,105, ADE20K 2,000, COCO-Stuff-27 5,000 and Cityscapes 500. No dataset download URL is given except the Context `59_labels.txt` page, because no other URL was verified.

## 8. Weight Inventory

| Asset | Size (B) | SHA-256 (prefix) | Required |
|---|---:|---|:---:|
| `voc.pth` | 658,427,542 | `f5e68096` | yes |
| `context.pth` | 660,733,078 | `21d78dc1` | yes |
| `ade.pth` | 667,296,086 | `8b05b8c5` | yes |
| `coco.pth` | 658,818,838 | `2c5b55de` | yes |
| `city.pth` | 658,372,438 | `a7daa5da` | yes |
| 5 CLIP text embeddings | 40,580–308,861 | `561365cf`, `ae450fd9`, `5410010b`, `34044aa7`, `28b83c54` | yes |
| CLIP ViT-B/16 | 350,837,078 | `5806e77c` (checked by the `clip` package) | yes (auto-download) |
| `utils/cls_token.pt` | — | `153b2bdc` | yes (in the upstream clone) |

The full hashes are in `docs/WEIGHTS.md` and `weights/CHECKSUMS.sha256`.
* Each checkpoint contains the learned prompt token, the decoder and a full CLIP copy (614 tensors).
* Locally trained checkpoints differ from the author files in 10 tensors (prompt, `pe_proj`, decoder). They are excluded.

## 9. Weight Redistribution Status

* **Checkpoints: NOT REDISTRIBUTED.** The upstream source is required. The verified links are the five Google Drive links in the upstream README Rectification table (lines 139–143 at the pinned commit). Files downloaded from them match the hashes.
* **No download script.** The upstream repository states no terms for redistribution or automated download.
* **Text embeddings:** not redistributed (yet). `tools/make_text_embeddings.py` regenerates them. On the paper environment, the tensors are bit-identical for VOC, Context-59, ADE20K and COCO-Stuff-27; Cityscapes is only numerically equal (max abs diff 4.9e-4). That GPU result comes from an earlier audit and was not re-run here. On CPU, regeneration is not bit-exact. Attaching the exact files is decision D3.

## 10. Third-Party License Status

| Component | License | Status |
|---|---|---|
| ReCLIP++ code (dogehhh/ReCLIP) | none | not included; imported from the user's clone. `cgsr/evaluation.py::shared_postprocess` re-implements its evaluator post-processing (15 lines; AST-identical to the research helper). The configs follow its schema, including the `PALETTE` lists. → low-risk items for D2 review. |
| ReCLIP++ checkpoints | none stated | not included |
| Candidate generator (ported from a third-party script, author not identified) | unknown | **not included**; D1 |
| OpenAI CLIP | MIT | pip dependency; the text-embedding recipe follows the CLIP zero-shot recipe |
| STEGO fine-to-coarse table | MIT (verified 2026-10-09) | included with its notice in `THIRD_PARTY_NOTICES.md` |
| PASCAL-Context class list | dataset site | 59 class names listed with their source URL |

No repository `LICENSE` was invented; `LICENSE_PENDING.md` explains the gap.

## 11. Environment

`requirements.txt` lists the dependencies, with the paper versions as comments; no exact pins are enforced, because the paper used a nightly PyTorch build. The verified historical environment is listed below. `environment.yml` was not created, since it could not be reconstructed reliably.

| Component | Version |
|---|---|
| Python | 3.10.19 |
| PyTorch / torchvision | 2.11.0.dev20260214+cu128 / 0.26.0.dev20260214+cu128 |
| CUDA / cuDNN | 12.8 / 9.10.2 |
| numpy, opencv-python, Pillow | 2.2.6, 5.0.0.93, 12.0.0 |
| OpenAI CLIP | commit `dcba3cb2` |
| GPU (paper) | RTX 5090 |
| OS | Windows 10 |

The README states that an RTX 5090 is not required.

## 12. Reproduction Commands

```bash
pip install -r requirements.txt
bash scripts/setup_upstream.sh third_party/base_predictor
python tools/verify_assets.py --base-code-root third_party/base_predictor --weights-dir weights
bash scripts/eval.sh voc          /path/to/VOCdevkit/VOC2012
bash scripts/eval.sh context59    /path/to/VOCdevkit/VOC2010
bash scripts/eval.sh ade20k       /path/to/ADEChallengeData2016
bash scripts/eval.sh coco_stuff27 /path/to/COCOStuff27
bash scripts/eval.sh cityscapes   /path/to/cityscapes
```

Windows: `scripts\setup_upstream.ps1`, `scripts\eval.ps1`. Underlying CLI: `python tools/eval_cgsr.py --help`.

## 13. Expected Results

mIoU (%), author checkpoints, full validation:

| Dataset | Base | Candidate | CGSR |
|---|---:|---:|---:|
| VOC 2012 | 85.3617 | 85.8969 | 85.7780 |
| Context-59 (H1) | 35.9988 | 30.8701 | 36.7901 |
| ADE20K | 16.4326 | 14.8911 | 16.4626 |
| COCO-Stuff-27 | 23.8842 | 23.9680 | 24.6723 |
| Cityscapes (gtFine) | 26.4569 | 24.9073 | 27.7328 |

These were cross-checked against the frozen author-checkpoint table, and reproduced exactly by re-scoring the frozen predictions with the release code (C9). Efficiency on Cityscapes, author checkpoint: Base 26.641 ms, Candidate 34.317 ms, CGSR 61.651 ms end-to-end, operator 1.090 ms. Full tables: `docs/RESULTS.md`.

Locally trained Context / Cityscapes numbers, the superseded H0 / gtCoarse numbers and the LOCAL efficiency run are excluded.

## 14. Large-File Scan

* Files > 10 MB: 0. Files > 50 MB: 0. Files > 100 MB: 0.
* Largest file: `RELEASE_MANIFEST.md` (about 16 KB). The largest code file is `tools/eval_cgsr.py` (13,892 B); the whole tree is about 177 KB (176,984 B).
* No `.pth`, `.pt`, `.ckpt`, `.npz`, `.npy`, `.png`, `.pdf`, `.zip`, `.tar` or `.log` files.
* Git LFS is not needed.

## 15. Secret/Privacy Scan

* 0 hits for tokens and keys (GitHub, Hugging Face, OpenAI-style and AWS key patterns; private-key blocks; password, token and API-key assignments; wandb keys).
* 0 e-mail addresses.
* 0 user or machine names.

## 16. Absolute-Path Scan

0 drive-letter, `/home`, `/Users` or `/mnt` paths, and 0 private folder names, in any file. Paths in docs are generic placeholders (`/path/to/...`).

## 17. Syntax/Import Checks

On the CPU, in the clean-clone simulation (`docs/RELEASE_CHECKS.md`):
* C1 compileall: PASS.
* C2 upstream pinned files: PASS.
* C3/C3b configs: PASS.
* C4 code equivalence: PASS (7/7).
* C5 Candidate defaults: PASS (25/25).
* C6 Context LUT: PASS.
* C7 module registration: PASS.
* C8 lists and order: PASS (5/5).
* C9 re-score of the frozen predictions: PASS, with exact mIoU and CIs on 5/5 datasets.
* After the 2026-10-10 role-name refactor: C1–C8 re-run PASS. C4b CPU A/B source parity (audited vs refactored pipeline, 5 images × 5 datasets, 18 tensor/label checks each) PASS, 450/450 bit-identical. N1–N13 CLI tests with the new flags PASS.
* N1–N12 CLI and error-path tests: PASS.

A fresh-context adversarial verifier gave **PARTIAL**:
* 355 numbers checked, 0 mismatches.
* No forbidden claims.
* Configs, method description and legal/privacy are clean.
* All 3 major and 10 minor findings were fixed or disclosed (`docs/RELEASE_CHECKS.md`).

The remaining gap is that bit-exact end-to-end equality on the **unmodified** upstream checkout is argued from the code, not measured. The paper predictions came from a research fork whose evaluation-path changes the release bypasses (`REPRODUCIBILITY.md`, provenance note).

**Not run:** the GPU end-to-end parity smoke (needs approval; D6).

## 18. Git Status

`git init` was run in this folder. The state is: branch with no commits, nothing staged, all files untracked, no remote. The ignore rules exclude `third_party/`, `outputs/`, weights, and `cgsr/candidate/model_sfp_dtlr.py`. Git is 2.50.1. The global identity is the user's own account; no AI co-author trailers are to be added.

## 19. Remaining User Decisions

| ID | Decision | Recommendation |
|---|---|---|
| D1 | **Candidate generator rights.** `model_sfp_dtlr.py` ports core logic from a third-party research script (`othermodel_guide/.../model_lrab_v1_voc_final_862.py`) whose author and license are unknown. Confirm the author's permission and add attribution, or keep it unpublished. | Ask the original author. Without the file nobody can run the release. |
| D2 | **License** for the original CGSR code. Also confirm comfort with the small re-implemented evaluator snippet and the configs that follow the ReCLIP++ schema. | Choose a permissive license (e.g. MIT or Apache-2.0) if your institution allows. Optionally ask the ReCLIP++ authors to add a license upstream. |
| D3 | **Text embeddings.** Attach the five exact `.pth` files (about 570 KB in total) to a GitHub Release, or rely on regeneration. Cityscapes regeneration is not bit-exact. | Attach them as release assets (not in git). |
| D4 | **CITATION authors and venue** (currently TODO). | Fill them in after submission or acceptance. |
| D5 | **Figures.** Any paper figure embeds dataset photographs; Cityscapes terms restrict redistribution. | Ship none, or a photo-free schematic. |
| D6 | **GPU parity smoke** (about 1–2 minutes): 20 images per dataset through the release runner on unmodified upstream, compared pixel by pixel with the frozen predictions. | Approve. It is the only remaining end-to-end check. |
| D7 | **Repository name and visibility.** | Create it private first, then switch to public. |

## 20. Exact Next Commands for GitHub Push

Run these only after D1, D2 and D4 are settled (`docs/GITHUB_PUSH_CHECKLIST.md`):

```bash
cd <path-to-this-repository>
git status
git add .
git status
git commit -m "Initial CGSR release"
git branch -M main
git remote add origin <USER_REPO_URL>
git push -u origin main
```
