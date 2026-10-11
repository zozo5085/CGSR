# Limitations

These limitations follow from the frozen evidence (`docs/RESULTS.md`) and the code. They are part of the release, not a footnote.

1. **One Base / refinement family.** CGSR was evaluated with a single Base predictor (ReCLIP++, ViT-B/16, author checkpoints) and a single refinement stage (the fixed SFP + DTLR candidate generator). No other refiner was tested, so CGSR is **not** shown to be refiner-agnostic or plug-and-play for arbitrary refiners.

2. **Two full backbone passes.**
   * `CGSRPipeline` computes `base_logits`/`projected_features` and the `candidate_logits` in two separate forward passes (`cgsr/model_cgsr.py`).
   * On Cityscapes (RTX 5090, batch 1, FP32), end-to-end CGSR takes 61.651 ms/image. That is 2.31× Base (26.641 ms) and 1.80× the Candidate (34.317 ms). The CGSR operator alone takes 1.090 ms.
   * The end-to-end overhead is therefore not negligible. A fused single-pass implementation was not built or tested.

3. **Whole-logit rollback was not compared with alternatives.** CGSR swaps the complete logit vector of a token. Partial or soft rollback granularities were not evaluated, so whole-logit rollback is not shown to be superior.

4. **Pre- vs post-refinement context was not compared.** The support `Q` is built from the pre-refinement (Base) probabilities. Building it from the refined probabilities was not evaluated.

5. **CLIP affinity is not shown to matter.**
   * In full-validation controls, uniform weights and RGB affinity perform about as well as CLIP-feature affinity. CLIP − RGB is significantly positive on 0/5 datasets and negative on 2/5.
   * A label-count plurality rule is never significantly worse than CGSR, and is significantly better on 4/5 datasets at 95 %.
   * The measured mechanism is selective rollback driven by local class support.

6. **5×5 is a fixed, pre-specified window, not an optimum.** 3×3 is worse on 4/5 datasets. 7×7 is significantly better on VOC and Cityscapes, slightly (not significantly) higher on COCO-Stuff-27, and worse on Context-59 and ADE20K.

7. **The Candidate generator uses the same numeric "structure" class indices on every dataset.**
   * `DTLR_STRUCTURE_CLASSES = [4, 8, 10]` are VOC channel indices (bottle, chair, dining table). They are applied unchanged to all label spaces, where they denote other classes (for example Cityscapes fence, vegetation and sky).
   * On development cohorts, removing the list changed CGSR by at most 0.025 mIoU points per dataset and the Candidate by at most 0.06, except Cityscapes, where the Candidate dropped 0.44.
   * Its effect on the full validation sets was not measured.

8. **Gains are not uniform.**
   * On VOC, CGSR is 0.12 mIoU points below the Candidate, and the CI contains 0.
   * On ADE20K, CGSR is 0.03 above Base, and the CI contains 0.
   * CGSR discards 37–55 % of the Candidate's beneficial pixel revisions.
   * On VOC, a random rollback with the same acceptance count is not significantly worse (2/10 seeds).

9. **Evaluation protocol choices.**
   * Context-59 is scored in the official `59_labels.txt` order. This was adopted after the project's earlier internal order failed Base parity, so it is a post-hoc protocol correction and is disclosed as such.
   * COCO-Stuff-27 labels were converted locally.
   * The class-presence filter `PD` of the base predictor's evaluation is part of the shared evaluator (`shared_postprocess`) and uses per-dataset values.

10. **Hardware determinism.** The paper-facing predictions were produced on CUDA (RTX 5090, PyTorch 2.11 nightly, CUDA 12.8). Bit-exact agreement on other GPUs, drivers or library versions is not established. Small floating-point differences can change a few argmax decisions.

11. **No training-seed variance.** Each dataset has one author-released checkpoint, so variability across training seeds is not measured. The bootstrap CIs cover image sampling only.

12. **Statistical scope.** CIs are 95 % paired-bootstrap percentile intervals. No p-values or multiple-comparison corrections are reported for the main table; the controls use a 99 % Bonferroni check over 5 datasets.
