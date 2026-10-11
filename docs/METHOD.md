# Method

CGSR (Context-Guided Selective Rollback) is a training-free verifier that runs after an existing refinement stage. It has no learnable parameters. At every token it keeps the refined prediction or rolls back to the pre-refinement prediction, depending on which class is better supported by the pre-refinement predictions of its 5×5 neighbourhood.

The roles of the components are kept separate throughout this repository. The code uses architecture-neutral role names; the third column records what each role is in this study:

| Role (code name) | What it does | Implementation and provenance | Contribution of this work? |
|---|---|---|---|
| `FrozenBasePredictor` | produces the base logits, the projected patch features and the penultimate attention | ReCLIP++ ViT-B/16 with author-released checkpoints (Wang & Kang, IJCV 2025). It is the upstream base-predictor class, imported unmodified from the user's checkout and not redistributed (`THIRD_PARTY_NOTICES.md`). | No |
| `FixedCandidateGenerator` | existing refinement stage with fixed settings; produces the candidate logits | SFP + DTLR refinement (`cgsr/candidate/`, distributed separately; provenance in `cgsr/candidate/README.md`) | No; it is the object of verification |
| `SelectiveRollbackVerifier` (with `local_class_support`) | post-refinement verifier and state selector; produces the final logits | `cgsr/verifier.py`. It depends only on PyTorch and can be imported without the base code or the Candidate generator. | **Yes**: the CGSR-owned core |
| `CGSRPipeline` | experimental integration of the three roles above in one module (two backbone passes per image) | `cgsr/model_cgsr.py`. It subclasses the candidate generator and the base predictor, which keep their origin. | Only its `forward()` integration |
| `shared_postprocess` | turns any of the three logit states into a segmentation | `cgsr/evaluation.py`; re-implements the evaluation post-processing of the base code | No (shared evaluator) |

## 1. Pipeline

```text
image ─► FrozenBasePredictor ─┬─► base_logits            L0  (K × Hp × Wp token logits)
                              ├─► projected_features     F   (unit-norm projected CLIP patch features, C × Hp × Wp)
                              └─► penultimate_attention       (input to the candidate generator only)
image ─► FixedCandidateGenerator ──► candidate_logits     Lr  (K × Hp × Wp)
(L0, Lr, F) ─► SelectiveRollbackVerifier ─► final_logits  L_final  (token grid)
{L0, Lr, L_final} ─► shared_postprocess ─► segmentation (label maps at label resolution)
```

* `Hp × Wp` is the ViT-B/16 patch-token grid at the evaluation scale. The verifier selects at this **native token resolution**; it never sees pixels.
* The **final evaluation is pixel-level**. Base, candidate and CGSR logits each go through the same `shared_postprocess`, so all three rows of every table share identical post-processing (Section 4).
* `CGSRPipeline` calls the backbone **twice per image**. The first pass returns `base_logits` and `projected_features`; the second, plain pass returns `candidate_logits`. The inherited model API exposes either the base outputs or the refined logits, never both. This doubles backbone compute; see `docs/LIMITATIONS.md`. Batch size is 1.
* `CGSRPipeline(..., return_debug=True)` returns `(final_logits, states)`. `states` holds every tensor named in this document under its code name.

## 2. SelectiveRollbackVerifier (the contribution)

| Symbol (paper) | Code name (`states` key) | Definition | Code (`cgsr/verifier.py`; first three rows: `cgsr/model_cgsr.py`) |
|---|---|---|---|
| `L0` | `base_logits` | base logits, `K × Hp × Wp` | first pass, `return_feat=True` |
| `F` | `projected_features` | unit-norm projected patch features | first pass |
| `Lr` | `candidate_logits` | candidate logits, `K × Hp × Wp` | second pass |
| `P0` | `base_probs` | `softmax(L0)` over classes, temperature τ = 1 | `torch.softmax(base_logits, dim=0)` |
| `Q` | `local_support` | 5×5 local class support (below) | `local_class_support(base_probs, projected_features)` |
| `c0` | `base_class` | `argmax_k L0(k)` per token | `base_probs.argmax(0)` |
| `cr` | `candidate_class` | `argmax_k Lr(k)` per token | `softmax(candidate_logits).argmax(0)` |
| `Q(c0)`, `Q(cr)` | `support_base`, `support_candidate` | support of each token's two classes, from the same `Q` | `local_support.gather(...)` |
| `g` | `support_gain` | `Q(cr) − Q(c0)` | `support_candidate − support_base` |
| gate | `accept` | `g > 0`, strict (ties and `cr = c0` keep `L0`) | `support_candidate > support_base` |
| `L_final` | `final_logits` | `Lr` if accepted, otherwise `L0`; the **whole logit vector** is swapped | `torch.where(accept, candidate_logits, base_logits)` |

The local class support at token `i` is

```text
Q_i(k) = Σ_{j ∈ N5(i)} w_ij · P0_j(k),     w_ij = exp(<f_i, f_j>) / Σ_{j' ∈ N5(i)} exp(<f_i, f_j'>)
```

* `N5(i)` is the 5×5 window centred at `i`, centre included.
* Positions outside the token grid get affinity `−∞`, i.e. weight 0.
* `f_i` are the unit-norm projected CLIP patch features of the base pass. The softmax over the 25 dot products uses no temperature scaling (τ = 1).

The `argmax` of a softmax equals the `argmax` of the logits; the two code forms can differ only on exact floating-point ties.

The full-validation controls show that this weighting is close to uniform: the mean effective number of neighbours is about 22–23 of 25 (`docs/RESULTS.md` §4). The measured mechanism is **selective rollback driven by local class support**. The paper's controls do not show that feature affinity is necessary: uniform weights and RGB affinity perform about as well, and a simple label-count (plurality) rule is never significantly worse (`docs/RESULTS.md` §3–§4).

The window size (5×5), τ = 1 and the strict gate were fixed before the full-validation evaluation. They are not tuned optima.

## 3. FixedCandidateGenerator (existing refinement stage, fixed)

The candidate is produced by the SFP + DTLR refinement used throughout the study. Every setting below is fixed and identical on all five datasets. It is **not** part of the CGSR contribution and is documented here only so the pipeline can be reproduced (code: `cgsr/candidate/model_sfp_dtlr.py`; see `cgsr/candidate/README.md` for its distribution status).

**Token selection.** Selection is per token over the spatial grid, not a class-wise Top-K:

```text
s_i   = A_cls,i − A_i,i                         reliability score from the penultimate ViT layer
                                                (layer 11 of 12), attention averaged over heads
Ω     = { i : max_k softmax(10 · L0_i)_k < 0.97 }       low-confidence eligibility
U     = Top-K_s of s_i over i ∈ Ω,   K_s = 800          (if |Ω| ≤ 800, U = Ω)
```

**Local logit purification** (selected tokens `U` only; mask `m`):

```text
t_i   = n_i + λ · a_i · (p_i − n_i)             λ = 2.0
          n_i: mean logits of the non-selected 8-neighbours (0 if none of the 8 neighbours is non-selected)
          p_i: mean logits of the 5×5 high-confidence tokens (max softmax(10·L0) > 0.95, not in U)
          a_i: 1 if any such proxy token exists, else 0
L1    = (1 − β) · L0 + β · [ L0 ⊙ (1 − m) + t ⊙ m ]     β = LOGIT_BETA = 0.55
```

**RGB-guided local refinement (DTLR)**:
* A domain-transform recursive filter is guided by the model-input RGB image, resized to the token grid and min-max normalised. Its settings are σ_s = 70 (absolute, in token units), σ_r = 1.50 and one iteration.
* The filter is applied to `L1`, and only the selected tokens are updated:

```text
Lr    = L1 + β_d · m' ⊙ (DT(L1) − L1)                  β_d = DTLR_BETA = 1.20
```

**Structure protection.** `m'` is `m` with some tokens rejected. A token is rejected when its `L1` class is in `DTLR_STRUCTURE_CLASSES = [4, 8, 10]`, the filtered class differs, and the filtered confidence is lower (gain threshold 0.00).

These three ids are **VOC channel indices** (bottle, chair, dining table). They are applied unchanged to every dataset, where they denote other classes:
* Context-59 (alphabetical channels): bench, book, building
* ADE20K: tree, windowpane, cabinet
* COCO-Stuff-27: indoor, outdoor, sports
* Cityscapes: fence, vegetation, sky

This is a property of the existing generator; it is disclosed in `docs/LIMITATIONS.md`.

## 4. Shared evaluator (`shared_postprocess`)

The same post-processing is applied to `base_logits`, `candidate_logits` and `final_logits` (`cgsr/evaluation.py::shared_postprocess`). It re-implements the evaluation post-processing of the base code (upstream ReCLIP++):

1. **Class-presence filter `PD`.** Classes whose maximum `softmax(10·L)` over all tokens is below `PD` get logit −100 everywhere. `PD` per dataset: VOC 1.0, Context-59 0.85, ADE20K 0.5, COCO-Stuff-27 0.95, Cityscapes 0.5. These are the base predictor's evaluation settings, stored in the base configs (`configs/`).
2. Bilinear resize to the model-input size, then to the label size.
3. Softmax and argmax.

mIoU is computed over valid pixels (label ≠ 255) from per-image intersections and unions summed over the dataset. Classes with an empty union count as 0, and the sum is divided by the number of classes.

## 5. What the method does not include

* No training, no learnable parameters, no ground truth, and no per-dataset tuning in CGSR.
* No multi-scale or test-time augmentation.
* No change to the base predictor weights, features or attention.
