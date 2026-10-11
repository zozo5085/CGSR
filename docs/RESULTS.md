# Results

All numbers are **paper-facing full-validation results** produced with the **author-released ReCLIP++ checkpoints** and the protocol in `docs/METHOD.md`, unless a section says otherwise.
* mIoU is reported in percent (raw fraction × 100), and differences in **mIoU points**.
* CIs are 95 % percentile intervals from a paired image-level bootstrap with 10,000 replicates (`REPRODUCIBILITY.md` §6).
* Machine-readable copies are in `results/tables/`. They were generated from the frozen result files, whose SHA-256 are listed in `RELEASE_MANIFEST.md`.

Status legend used below: **paper-facing** (full validation, frozen) · **diagnostic** (not a main-table result) · **not released** · **not run**.

## 1. Main results (paper-facing)

Candidate = the existing SFP + DTLR refinement; CGSR = the Candidate verified by CGSR.

| Dataset | Images | Base | Candidate | CGSR | Candidate − Base | CGSR − Base | CGSR − Candidate |
|---|---:|---:|---:|---:|---|---|---|
| VOC 2012 | 1,449 | 85.3617 | 85.8969 | 85.7780 | +0.5352 [+0.2473, +0.8527] | +0.4163 [+0.2910, +0.5456] | −0.1189 [−0.3565, +0.0790] |
| Context-59 (H1) | 5,105 | 35.9988 | 30.8701 | 36.7901 | −5.1287 [−5.4844, −4.7719] | +0.7913 [+0.6797, +0.9004] | +5.9200 [+5.6127, +6.2314] |
| ADE20K | 2,000 | 16.4326 | 14.8911 | 16.4626 | −1.5415 [−1.9308, −1.1851] | +0.0300 [−0.0668, +0.1660] | +1.5715 [+1.2696, +1.9322] |
| COCO-Stuff-27 | 5,000 | 23.8842 | 23.9680 | 24.6723 | +0.0837 [−0.0375, +0.2106] | +0.7880 [+0.7359, +0.8429] | +0.7043 [+0.6184, +0.7881] |
| Cityscapes (gtFine) | 500 | 26.4569 | 24.9073 | 27.7328 | −1.5495 [−1.8967, −1.1923] | +1.2760 [+1.1344, +1.4180] | +2.8255 [+2.5478, +3.0859] |

Reading the table without over-claiming:

* CGSR is above Base on all five datasets. On ADE20K the gain is +0.03 mIoU points and its CI contains 0.
* CGSR is above the Candidate on four datasets. On VOC it is −0.12 mIoU points below the Candidate, with a CI containing 0.
* The largest gains are where the Candidate is below Base (Context-59, ADE20K, Cityscapes): CGSR rolls back many harmful revisions there.

**Bootstrap note.** The CIs above use one resampler and one seed for all datasets (seed 20260923, `cgsr/evaluation.py::bootstrap`). `tools/eval_cgsr.py` reproduces them exactly. The paper's table prints VOC and ADE20K CIs from an earlier resampler (seed 20260918). Its intervals differ by at most 0.0083 mIoU points, with no change of sign or of zero-containment:

| Dataset | Contrast | Paper (seed 20260918) | This table (seed 20260923) |
|---|---|---|---|
| VOC | Candidate − Base | [+0.2556, +0.8557] | [+0.2473, +0.8527] |
| VOC | CGSR − Base | [+0.2929, +0.5472] | [+0.2910, +0.5456] |
| VOC | CGSR − Candidate | [−0.3579, +0.0746] | [−0.3565, +0.0790] |
| ADE20K | Candidate − Base | [−1.9306, −1.1927] | [−1.9308, −1.1851] |
| ADE20K | CGSR − Base | [−0.0670, +0.1657] | [−0.0668, +0.1660] |
| ADE20K | CGSR − Candidate | [+1.2738, +1.9378] | [+1.2696, +1.9322] |

Base parity with the published ReCLIP++ numbers is within about 0.10 mIoU points on every dataset (VOC −0.04, Context −0.10, ADE20K +0.03, COCO-Stuff-27 +0.08, Cityscapes −0.04).

**Protocol notes that travel with this table:**
* **Context-59 is scored in the official `59_labels.txt` class order ("H1").** This order was adopted after an earlier project-internal class order failed Base parity. The earlier order is superseded and not reported.
* **COCO-Stuff-27 labels were converted locally** from COCO-Stuff with the standard 27-class grouping (`tools/convert_coco_stuff27_labels.py`).
* **Cityscapes uses gtFine** `labelTrainIds` (19 classes) on the 500 validation images. gtCoarse scores are superseded and not reported.

## 2. Transitions: Retention and Recovery (paper-facing)

These statistics are pixel level, computed at the original resolution over valid GT pixels.
* B, R and G mean that Base, Candidate and CGSR, respectively, are correct at a pixel.
* **Retention** (BPR) = #(B✗, R✓, G✓) / #(B✗, R✓): the fraction of beneficial revisions kept.
* **Recovery** (HRR) = #(B✓, R✗, G✓) / #(B✓, R✗): the fraction of harmful revisions undone.
* J = Retention + Recovery − 1.

| Dataset | B✗→R✓ (% valid px) | B✓→R✗ (% valid px) | Retention | Recovery | J |
|---|---:|---:|---:|---:|---:|
| VOC 2012 | 0.84 | 0.56 | 0.6247 | 0.4992 | +0.1239 |
| Context-59 (H1) | 7.18 | 11.07 | 0.4949 | 0.8508 | +0.3457 |
| ADE20K | 5.42 | 8.12 | 0.4530 | 0.7938 | +0.2469 |
| COCO-Stuff-27 | 5.56 | 4.50 | 0.5470 | 0.6837 | +0.2307 |
| Cityscapes (gtFine) | 7.97 | 9.31 | 0.6298 | 0.8514 | +0.4812 |

J > 0 on every dataset, so CGSR is selective rather than a blanket rollback. It still discards 37–55 % of the beneficial revisions.

## 3. Mechanism controls: local-support variants (paper-facing, pre-registered)

All variants share the same `base_logits` (`L0`), `candidate_logits` (`Lr`), `projected_features` (`F`), `base_probs` (`P0`), base/candidate classes, strict gate and whole-logit selection; only the support rule changes:
* **Uniform-Q**: uniform weights over the in-grid 5×5 window.
* **Label-Plurality**: accept iff, in the 5×5 window of Base labels, the candidate class occurs more often than the base class.
* **Random**: the same number of accepted tokens per image as CGSR, chosen at random among disagreeing tokens (10 seeds).

| Dataset | CGSR | Uniform-Q | Label-Plurality | Random (mean ± std, 10 seeds) |
|---|---:|---:|---:|---:|
| VOC 2012 | 85.7780 | 85.7709 | 85.7604 | 85.7168 ± 0.0212 |
| Context-59 (H1) | 36.7901 | 36.6673 | 36.9797 | 35.8106 ± 0.0253 |
| ADE20K | 16.4626 | 16.4330 | 16.6084 | 16.2026 ± 0.0100 |
| COCO-Stuff-27 | 24.6723 | 24.6816 | 24.7568 | 24.3152 ± 0.0067 |
| Cityscapes (gtFine) | 27.7328 | 27.7487 | 27.7895 | 26.5489 ± 0.0168 |

Paired differences: mIoU points, 95 % CI. Markers give significance at 95 % / 99 % Bonferroni over 5 datasets (pos, neg or ns).

| Dataset | CGSR − Uniform-Q | CGSR − Label-Plurality | Seeds with CGSR > Random (95 % / 99 %) |
|---|---|---|---|
| VOC 2012 | +0.0071 [−0.0108, +0.0252] ns/ns | +0.0177 [−0.0183, +0.0538] ns/ns | 2/10 / 1/10 |
| Context-59 (H1) | +0.1228 [+0.1012, +0.1468] pos/pos | −0.1896 [−0.2227, −0.1566] neg/neg | 10/10 / 10/10 |
| ADE20K | +0.0296 [+0.0092, +0.0447] pos/ns | −0.1458 [−0.1784, −0.1092] neg/neg | 10/10 / 10/10 |
| COCO-Stuff-27 | −0.0093 [−0.0160, −0.0025] neg/neg | −0.0845 [−0.1013, −0.0686] neg/neg | 10/10 / 10/10 |
| Cityscapes (gtFine) | −0.0159 [−0.0510, +0.0189] ns/ns | −0.0567 [−0.1073, −0.0017] neg/ns | 10/10 / 10/10 |

The pre-registered conclusion is class **D, "simple plurality is sufficient"**:
* CGSR is never significantly better than Label-Plurality.
* Plurality is significantly better on 4/5 datasets at 95 % and 3/5 at 99 %.
* Against random rollback, the local-support decision is significantly better on 4/5 datasets (all 10 seeds), but **not on VOC**.

The paper keeps the frozen CGSR rule and reports these controls; it does not claim that feature affinity or probability aggregation is necessary.

## 4. Context window and affinity (paper-facing, pre-registered)

Window size, with CLIP affinity:

| Dataset | 3×3 | 5×5 (CGSR) | 7×7 | CGSR − 3×3 | CGSR − 7×7 |
|---|---:|---:|---:|---|---|
| VOC 2012 | 85.6778 | 85.7780 | 85.8514 | +0.1003 [+0.0475, +0.1557] pos/pos | −0.0734 [−0.1359, −0.0197] neg/neg |
| Context-59 (H1) | 36.7218 | 36.7901 | 36.5131 | +0.0682 [+0.0140, +0.1217] pos/ns | +0.2769 [+0.2267, +0.3282] pos/pos |
| ADE20K | 16.5244 | 16.4626 | 16.3167 | −0.0618 [−0.1142, +0.0098] ns/ns | +0.1459 [+0.0865, +0.2016] pos/pos |
| COCO-Stuff-27 | 24.5085 | 24.6723 | 24.6789 | +0.1638 [+0.1405, +0.1870] pos/pos | −0.0067 [−0.0292, +0.0156] ns/ns |
| Cityscapes (gtFine) | 27.4130 | 27.7328 | 27.8406 | +0.3198 [+0.2761, +0.3647] pos/pos | −0.1078 [−0.1725, −0.0494] neg/neg |

5×5 beats 3×3 on 4/5 datasets. 7×7 is significantly better on VOC and Cityscapes, and significantly worse on Context-59 and ADE20K. On COCO-Stuff-27, 7×7 is higher by 0.0067 mIoU points, which is not significant. **5×5 was fixed a priori; no optimality claim is made.**

Affinity, with a 5×5 window:

| Dataset | Uniform | RGB affinity | CLIP affinity (CGSR) | CGSR − RGB | RGB − Uniform |
|---|---:|---:|---:|---|---|
| VOC 2012 | 85.7709 | 85.7726 | 85.7780 | +0.0055 [−0.0092, +0.0208] ns/ns | +0.0017 [−0.0090, +0.0119] ns/ns |
| Context-59 (H1) | 36.6673 | 36.7814 | 36.7901 | +0.0087 [−0.0133, +0.0321] ns/ns | +0.1141 [+0.1006, +0.1272] pos/pos |
| ADE20K | 16.4330 | 16.4647 | 16.4626 | −0.0021 [−0.0165, +0.0100] ns/ns | +0.0317 [+0.0171, +0.0426] pos/pos |
| COCO-Stuff-27 | 24.6816 | 24.7220 | 24.6723 | −0.0497 [−0.0564, −0.0430] neg/neg | +0.0404 [+0.0363, +0.0444] pos/pos |
| Cityscapes (gtFine) | 27.7487 | 27.7780 | 27.7328 | −0.0451 [−0.0819, −0.0112] neg/neg | +0.0293 [+0.0187, +0.0394] pos/pos |

* **CLIP affinity adds no measurable value over RGB affinity**: CGSR − RGB is significantly positive on 0/5 datasets and negative on COCO-Stuff-27 and Cityscapes.
* The 5×5 kernel is close to uniform. The mean effective number of neighbours is 22.42 / 23.15 / 23.18 / 23.24 / 23.14 of 25 for CLIP affinity (VOC / Context / ADE20K / COCO / Cityscapes).
* RGB affinity is defined as α_ij = softmax_j(−(2/3)·‖I_i − I_j‖²), using the de-normalised model input resized to the token grid.
* The window and affinity variants are evaluation-only controls. This repository does not include code for them.

## 5. Efficiency (paper-facing; Cityscapes author checkpoint only)

| Method | ms / image | × Base |
|---|---:|---:|
| Base | 26.641 | 1.000 |
| Candidate | 34.317 | 1.288 |
| CGSR (end-to-end) | 61.651 | 2.314 |
| CGSR operator only (verifier) | 1.090 | n/a |

**Setting:**
* NVIDIA GeForce RTX 5090, batch size 1, FP32 (no AMP, no `torch.compile`).
* Input 1×3×448×896; the first 32 Cityscapes validation images.
* 20 warm-up inferences and 5 repeats; `time.perf_counter` with `torch.cuda.synchronize()`.
* The timing boundary runs from the prepared CUDA input tensor to the final argmax (disk I/O, decoding and CPU pre-processing excluded).
* Parameters: 150,698,107 in total; **CGSR adds 0 learnable parameters**.
* The timed predictions were bit-identical to the frozen author predictions on 32/32 images.

**Interpretation:**
* The 1.090 ms figure is the CGSR operator alone, **not** the end-to-end cost of CGSR.
* The current implementation runs two full backbone passes, so end-to-end CGSR costs 2.31× Base and 1.80× Candidate.
* FLOPs, other datasets and batch sizes > 1 were not measured.

## 6. Status of other numbers

| Item | Status |
|---|---|
| Sections 1–5 above | paper-facing, full validation, author checkpoints |
| Results with locally trained Context / Cityscapes checkpoints | diagnostic (robustness supplement); not released |
| Context-59 under the superseded internal class order (H0); Cityscapes gtCoarse | superseded; not released |
| Multi-scale majority variant (MS_MAJ) | diagnostic; not the evaluated method; not released |
| Parameter study and window one-factor-at-a-time sweeps on 256-image training-split cohorts | diagnostic / dev-only; not released |
| Candidate component ablation (SFP only / DTLR only), rollback granularity, pre- vs post-refinement context, other refiners, gate-threshold sweep, FLOPs, latency on other datasets | not run |
| Comparison with published methods | positioning only. The protocols are not matched, so no ranking is claimed. |
