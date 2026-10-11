# Release manifest

Every file in this repository, with its provenance. Source-path aliases:
* `[fork]` = the research fork of ReCLIP++ used for the experiments.
* `[workspace]` = the research workspace; `[cgsr-workspace]` = its CGSR sub-folder.

The local paths behind these aliases are recorded outside the repository. Scientific logic was not changed in any copied or moved code; see `docs/RELEASE_CHECKS.md` (C4) for the AST equality checks.

## 1. Files

| Release path | Source | SHA-256 of source | SHA-256 of release copy | Status | Purpose | Provenance / release-only modification |
|---|---|---|---|---|---|---|
| `.gitattributes` | — | — | `d7e8be1286aa40446ae0bda0710b29133af1166e233851124448b5afef73f249` | generated | text normalisation | written for the release |
| `.gitignore` | — | — | `2d7d003b33a7617a21cc1841e5220d45154c2cce445b0424d85483c14533d235` | generated | ignore weights/outputs/upstream/candidate | written for the release |
| `cgsr/__init__.py` | — | — | `408a5b3f715b55563464ccb77dbc4d0527577c342c5c332848cfeb3a6f07ae07` | generated | package marker | written for the release |
| `cgsr/candidate/__init__.py` | — | — | `645ff7d4a383f7f3149537437eb0bff161da6f731eab221bef3e2da2d8385324` | generated | FixedCandidateGenerator alias of the separately distributed candidate class | written for the release |
| `cgsr/candidate/README.md` | — | — | `bc17a24bc6c0948caddf5368194b632550a0e9d397604d96341bd24658be9864` | generated | candidate generator distribution status | written for the release |
| `cgsr/context_h1.py` | [workspace]/experiments/context_h1_relock_v1/h1_common.py<br>[workspace]/experiments/context_h1_relock_v1/fixtures/official_59_labels.txt | `96e020f83d67616b5853c0232166b01cb78d1d270603007ae298b1e4c7f4ea5d`<br>`5c116adce83682e6163b6f01606a1765f60874935d5dbb5129a5f9f62838a0d7` | `085c2a53952789029a52481c1ecf438a6f0db99b5470f745472165fa3625b79e` | generated (re-implementation) | Context-59 official class order and name-based LUT | direct alphabetical->official LUT; equality with the frozen two-step LUT checked (RELEASE_CHECKS C6) |
| `cgsr/data.py` | — | — | `9d1662eeec85bfb7d3059cb4f0975cca146724691c2dec29f0ccc033b398903f` | generated | validation file lists | follows upstream read_file_list conventions (val side only); order checked equal to frozen runs (C8) |
| `cgsr/evaluation.py` | [fork]/tools/analysis/test_cgsr_integrated_parity.py<br>[workspace]/experiments/cgsr_research_program/run_miou_improvement_stage1.py<br>[workspace]/experiments/reclipp_parameter_study_v1/context_seeds/eval_seed_checkpoint.py | `f0b5274a29801dddc628b5a1d33077f2d80dff73eae2c451eb15139f80240021`<br>`ca5113d36a0e33966007cf6a8a13d137380f45007748264f972f8e78ee27e7d2`<br>`02d96e086b5965d973a6f7af3ad4b89d5e0c08bf348266aea334e50b3354fb33` | `1ce79e3a4a5d76d1a00506137264a9ef7093342788a3d5ce2687c8b270a826d8` | generated (functions moved) | shared evaluator, IoU metric, paired bootstrap | shared_postprocess (=final_prediction), raw_label, intersection_union(_from_label), miou, bootstrap(=bca): function bodies AST-identical to the sources; shared_postprocess re-implements the ReCLIP++ evaluator post-processing |
| `cgsr/frozen.py` | — | — | `097cb2c2c9c637ccaf788004a2746f5884f07f5f9e4ad3c823edda1d05ecfbb9` | generated | frozen settings, hashes, expected results | written for the release |
| `cgsr/model_cgsr.py` | [fork]/model/model_cgsr.py | `972d00bfa4cc2a4f9b8099fa58f5b32008879d48d789bba55b7fccfec5d7b909` | `c1fe3f05dddbbdd81974751440a03efa01922480ee6b7578186d44ef586175a9` | modified-for-release | CGSRPipeline: experimental integration layer (FrozenBasePredictor / FixedCandidateGenerator bindings) | original; refactored to role names (2026-10-10): class RECLIPPP -> CGSRPipeline, inline gate -> SelectiveRollbackVerifier, local_Q -> local_class_support (AST-identical body), debug keys renamed; verifier core moved to cgsr/verifier.py (2026-10-11); numerically identical to the source (CPU A/B source parity: 450/450, then 900/900 bit-identical vs v2 and audited) |
| `cgsr/upstream.py` | — | — | `af132acd658f0942cb570960ad2ee7903d33490179cf7d474a3aab8168ddd279` | generated | base-code checkout verification (verify_base_code) and pipeline loading (activate_base_code) | written for the release |
| `cgsr/verifier.py` | [fork]/model/model_cgsr.py | `972d00bfa4cc2a4f9b8099fa58f5b32008879d48d789bba55b7fccfec5d7b909` | `7022ea2b8ca54274076d7c3f6a589b1e4c89b76ded4e22bea50bed60cacf708e` | modified-for-release | SelectiveRollbackVerifier / local_class_support: the CGSR-owned core (PyTorch only) | original; moved verbatim out of cgsr/model_cgsr.py (2026-10-11); local_class_support AST-identical to the audited local_Q; verifier operations identical to the audited forward (CPU A/B 900/900 bit-identical vs v2 and audited) |
| `CITATION.cff` | — | — | `1a9eb24770a289cb74c8855b36e27365164fa8c5a88d87005fd5a8625d2b752c` | generated | citation metadata (TODO authors) | written for the release |
| `configs/ade20k/ade20k_cgsr.yaml` | [workspace]/experiments/official_reclippp_ours/configs/ade_official_ori.yaml | `ff231cfc7540a75c71767b1540b07a926c234f6edc442c74d8b4907cbe7e2bb1` | `69a8beb4d5c40ce05349c48f665dfda58310041faf478f588270028b34ce9abf` | modified-for-release | paper-facing evaluation config | values identical to the frozen run config except DATAROOT, TEXT_WEIGHT, LOAD_PATH, SAVE_DIR (placeholders); source comments dropped; header added; ReCLIP++ config schema |
| `configs/cityscapes/cityscapes_cgsr.yaml` | [workspace]/experiments/cityscapes_baseline/configs/cityscapes_test_verified.yaml | `ddcf23f3358fd7e85a2d8af9fab0eb65c104986c43024433eb8a06265299beda` | `8c8a97a8d661b61c8cf8d52d20b9e214312a423be26523896c61d8922d505cf2` | modified-for-release | paper-facing evaluation config | values identical to the frozen run config except DATAROOT, TEXT_WEIGHT, LOAD_PATH, SAVE_DIR (placeholders); source comments dropped; header added; ReCLIP++ config schema |
| `configs/coco_stuff27/coco_stuff27_cgsr.yaml` | [workspace]/experiments/coco_transfer/configs/coco_official_ori.yaml | `40e1108c2d494976f503da0ef21802a8923c993d3e669a149323d2ef07526859` | `8b5a5075b1f42601589a9b2244faf2ba9c87df036aacb2d7c475ac19aa06514d` | modified-for-release | paper-facing evaluation config | values identical to the frozen run config except DATAROOT, TEXT_WEIGHT, LOAD_PATH, SAVE_DIR (placeholders); source comments dropped; header added; ReCLIP++ config schema |
| `configs/context59/context59_cgsr.yaml` | [fork]/config/context_test_ori_cfg.yaml | `307a857fedf17181e2f832fdebeb91565ee4bf00f0dcff0ea36f7269dcc6fe6d` | `9d73b51c6b28f6f911ab348f62c17ab73a703687562699731b21acde6ba1e1c9` | modified-for-release | paper-facing evaluation config | values identical to the frozen run config except DATAROOT, TEXT_WEIGHT, LOAD_PATH, SAVE_DIR (placeholders); source comments dropped; header added; ReCLIP++ config schema |
| `configs/voc/voc_cgsr.yaml` | [workspace]/experiments/official_reclippp_ours/configs/voc_official_ori.yaml | `3afa9a34b7075a008b0ad5b75ba5a06923dbb687155f29afe0f2a3719e764b39` | `0dcb64a1cd445ddc8f757478e17225ce89d6d79446dd352e75c278c4a786d2e2` | modified-for-release | paper-facing evaluation config | values identical to the frozen run config except DATAROOT, TEXT_WEIGHT, LOAD_PATH, SAVE_DIR (placeholders); source comments dropped; header added; ReCLIP++ config schema |
| `docs/DATASETS.md` | — | — | `460276780aea0cf9b36f75e7dc29a5d95db6ab3a2d81245774edf3cc23fe974f` | generated | documentation | written for the release |
| `docs/GITHUB_PUSH_CHECKLIST.md` | — | — | `bf3d7009276ba06bd2a1aa172f78d3b2f22dca5b558f3b00c25858a88b275bdf` | generated | documentation | written for the release |
| `docs/LIMITATIONS.md` | — | — | `1a8fef88c78f6f915a3b2c8872784953eb70a59bd186f6604aa6b71911606527` | generated | documentation | written for the release |
| `docs/METHOD.md` | — | — | `c54b4bd01a29fdc5babd30633c2ad106a4f3930d5a4118258af2d3f626ad21bc` | generated | documentation | written for the release |
| `docs/RELEASE_CHECKS.md` | — | — | `02937849a902f52a3b3188db9290d9002215f64b6c0d746fbb0136c6fca4de83` | generated | documentation | written for the release |
| `docs/RESULTS.md` | — | — | `ce5e30725ec91e9b0b1bd9afe8c4d3bfc4ebebbaff114de0528d962683956106` | generated | documentation | written for the release |
| `docs/WEIGHTS.md` | — | — | `7aa00c66800a9c5bfc57103ccb6df5b4d298a442ae3e0aa0d94171ea2778f0d7` | generated | documentation | written for the release |
| `LICENSE_PENDING.md` | — | — | `167b24a848e526e8bf9b963808a869dd561b1c9cf49d6ba1b16160d71ab0f809` | generated | license decision pending | written for the release |
| `README.md` | — | — | `ece49a37264e9adb7834641c54867d120368e07a259e938359200656494aed40` | generated | overview, install, usage | written for the release |
| `RELEASE_MANIFEST.md` | — | — | (self) | generated | this file | — |
| `RELEASE_REPORT.md` | — | — | `6d32134ab35aa3b62012b0f829b9ed3b44f20f39309435d40dbd03c1697bce5e` | generated | release report | written for the release |
| `REPRODUCIBILITY.md` | — | — | `dae263d4d3b2217f5534b129e8d83f4a4665f8df464f65e52e95e649dda2e510` | generated | protocol, determinism, environment | written for the release |
| `requirements.txt` | — | — | `2c0c9cc2e53e2144a4602ed4b2571df826c7366b5c360594040ba18043c4c538` | generated | dependencies | written for the release |
| `results/tables/efficiency_cityscapes_author.csv` | — | — | `7ca63e97a6d0ac65a9953395b12cffd2f7e61b263fe5ed6fa58924298492f8e1` | generated (from frozen JSON) | paper-facing result table | generated by script from the frozen result files listed in section 2 |
| `results/tables/main_results.csv` | — | — | `8898e4208b51a286f7ad3361c53be43fb6b6fab2417bc53cc320beaaeeba2606` | generated (from frozen JSON) | paper-facing result table | generated by script from the frozen result files listed in section 2 |
| `results/tables/mechanism_controls.csv` | — | — | `ba9e146002ed8d98960bbcf1f826983bbf7bd0676e0e9cee8dbfc1b6e199d00c` | generated (from frozen JSON) | paper-facing result table | generated by script from the frozen result files listed in section 2 |
| `results/tables/transitions.csv` | — | — | `49b796ca627fe38e24a513a64e9e08b1227950411883208624b8fb7fa925620e` | generated (from frozen JSON) | paper-facing result table | generated by script from the frozen result files listed in section 2 |
| `results/tables/window_affinity.csv` | — | — | `f165f7f861c1751c4269427292480bc7db2034492945760f72dc8d650f179f42` | generated (from frozen JSON) | paper-facing result table | generated by script from the frozen result files listed in section 2 |
| `scripts/eval.ps1` | — | — | `5ce974e41410b42371a509cbb07d62e7c5692069c86e4cf3f547861854517926` | generated | per-dataset evaluation wrapper (Windows) | written for the release |
| `scripts/eval.sh` | — | — | `f4a03ee172f03849c361f77474af209265effed7061500c37c120f495e8afe87` | generated | per-dataset evaluation wrapper | written for the release |
| `scripts/setup_upstream.ps1` | — | — | `02dd878ce8e00f69a91a905b3793d09bea17394f6dfa3ee96477b33e87627837` | generated | clone upstream at pinned commit (Windows) | written for the release |
| `scripts/setup_upstream.sh` | — | — | `35eb886b6d04135ed51c1fd3ae1ac5293c83a50f1a51d0778b6348fdd8e3f0a3` | generated | clone upstream at pinned commit | written for the release |
| `tests/test_standalone_import.py` | — | — | `e6583ac0cb60739856c7a5c5646e79a509856ba7f639066dd28f439bd3300471` | generated | clean-environment import test of the CGSR core | written for the release |
| `THIRD_PARTY_NOTICES.md` | — | — | `775edb67bb28340d61b6b87a0f5b0b982d177ef03da3491d3a3bc3d587e4725c` | generated | third-party components and licenses | written for the release |
| `tools/convert_coco_stuff27_labels.py` | [fork]/tools/coco_convert_labels.py | `1b6c4bb18c8b3c5d362e150feadc3599fe60abf36411efeb82197fb96a5a27a5` | `435b9c908fe2ec5afd4f25f9a539e12ac6657f948d0dc4ce65e089440342566c` | modified-for-release | COCO-Stuff fine -> 27-class GT conversion | original + STEGO FINE_TO_COARSE table (MIT); default local paths replaced by required CLI args; default split val2017; docstring; mapping logic unchanged |
| `tools/eval_cgsr.py` | [workspace]/experiments/reclippp_author_checkpoint_transfer_v1/eval_author_checkpoint.py<br>[fork]/tools/analysis/test_cgsr_integrated_parity.py | `2f8a6c4a4fa1c05451ce98d2903d321daacdd0fdcb4bb1d1d31fccd122f76991`<br>`f0b5274a29801dddc628b5a1d33077f2d80dff73eae2c451eb15139f80240021` | `ee9ad7230ff34107581ab7f2068081def1f8c9ee5e2e7ee264644adb788b8e12` | generated (re-implementation) | main evaluation CLI (Base / Candidate / CGSR in one pass) | same model construction, forward call, evaluator and metric as the runner that produced the Context/COCO/Cityscapes rows; paths via CLI (--base-code-root); role names; hash/settings guards |
| `tools/make_text_embeddings.py` | [workspace]/experiments/reclippp_checkpoint_provenance_audit_v1/text_embeddings/regen_text_embeddings.py | `4dba16ef479e6ae2c5bfc22870c2f3dfa625ec26b122cf0adb4b308248b2f12f` | `a6223dd753d981ca102849d05f691ee42b215b08f4e87aaafe4443cd864a1d2c` | generated (re-implementation) | regenerate CLIP text embeddings from the upstream checkout | OpenAI CLIP zero-shot recipe (MIT); class lists/templates imported from upstream at run time |
| `tools/verify_assets.py` | — | — | `0ffec98713c986bfe6848d0d098fd31797cd881086beb1b26ec66b55b88f8330` | generated | hash checks for upstream and weights | written for the release |
| `weights/.gitkeep` | — | — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | generated | placeholder | written for the release |
| `weights/CHECKSUMS.sha256` | — | — | `43cd1b23a6d944a09dacc5cf6ebe479f1cbeaad370c9e6deb780b03b5911af8d` | generated | weight hashes | written for the release |
| `weights/text/.gitkeep` | — | — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | generated | placeholder | written for the release |

## 2. Frozen result files used to generate `results/tables/` and `docs/RESULTS.md`

| Source | SHA-256 |
|---|---|
| [cgsr-workspace]/phase2_2026-10-06/e3_prepared/selfcheck/voc.json | `ed4e14f3eef01461d1240a39dbcb543d60677b1d902c9a8dbf7d3b09dfc52f8b` |
| [cgsr-workspace]/phase2_2026-10-06/e3_prepared/selfcheck/context_h1.json | `a90749e1eeb8e8606ffd379172a704753c59db0b321189ce74cd2fc9c197e50f` |
| [cgsr-workspace]/phase2_2026-10-06/e3_prepared/selfcheck/ade20k.json | `8ae2df88b5d45ae286905ef395f90b2f34058c44b3658c95bb4c380a409aef40` |
| [cgsr-workspace]/phase2_2026-10-06/e3_prepared/selfcheck/coco27.json | `1969b8151496f7adf93225840d511167b10c342bb7313389682b935232845119` |
| [cgsr-workspace]/phase2_2026-10-06/e3_prepared/selfcheck/cityscapes.json | `23dc881428f9fdd61b08903b449a712fa005e2dbc5992fd42e48da07bce6ec38` |
| [cgsr-workspace]/paper_assets/intro_figure_v1/mining/raw_minus_base_ci_voc_ade.json | `8967379380a45489959a1c69effb0e0409d4b3739a2e4f2df9915b04ec7dc325` |
| [cgsr-workspace]/phase2_2026-10-06/e2/e2_voc.json | `bae384391bc11db4533136900f3f80b02485ce6c9f9ae1cd01e623090a99e6ca` |
| [cgsr-workspace]/phase2_2026-10-06/e2/e2_context_h1.json | `9629c13c2267fb6647df0010fde444155532e0143f48d50a436ce8a170a04fcd` |
| [cgsr-workspace]/phase2_2026-10-06/e2/e2_ade20k.json | `344af81180180399ac18d7dc9dbb5524cdd3b8ef27866191779f8ea6ae7cb656` |
| [cgsr-workspace]/phase2_2026-10-06/e2/e2_coco27.json | `e418a805fcfcb5ce981284531dd3bfe7a998f47f69c7a4553d61b264641e8586` |
| [cgsr-workspace]/phase2_2026-10-06/e2/e2_cityscapes.json | `9233ad5f2ba7b6a2d64e81b3495614c4cd0a183e8367525bd2bc275bd723c19c` |
| [workspace]/experiments/cgsr_mechanism_fullval_v1/scores/voc.json | `d26372524804fd5023d158bd6a7537568ae77748b172c18cc2a35aa2dd961811` |
| [workspace]/experiments/cgsr_mechanism_fullval_v1/scores/context_h1.json | `9245df9839bb206827e83c11c2780d97c9e70f460a7738e793e37db1727b6891` |
| [workspace]/experiments/cgsr_mechanism_fullval_v1/scores/ade20k.json | `62bcac6b0f91e17e52bb3e00de1121004a0ba6fdf253c8d9fec44e8a5580cb39` |
| [workspace]/experiments/cgsr_mechanism_fullval_v1/scores/coco27.json | `8485bd7074154a7975804a45ca91dd0207b6a17b2f5b94856d6732a62d86ddf8` |
| [workspace]/experiments/cgsr_mechanism_fullval_v1/scores/cityscapes.json | `896f3d9bc2037a006f1746c61132a0cdca08fbcf22e578d9eaa3ab576ca2bdbe` |
| [workspace]/experiments/cgsr_ablation_window_rgb_fullval_v1/scores/voc.json | `167d41d34f4150d5c6951c48efaf1483305cec3d7820506154c860df5c802c7b` |
| [workspace]/experiments/cgsr_ablation_window_rgb_fullval_v1/scores/context_h1.json | `4d07ce2539cb0c5925dfd85f578ef3f83045d3d7c82b1465689ab1d570c0df8e` |
| [workspace]/experiments/cgsr_ablation_window_rgb_fullval_v1/scores/ade20k.json | `dd1a31ea609869aaf5d32a9cb68a3ee3ec31a38d8589c6cdc1ee85d331625b7c` |
| [workspace]/experiments/cgsr_ablation_window_rgb_fullval_v1/scores/coco27.json | `dda54c463d68a3a6841fd9726c20c878002fb1ab3ef67ba2227ec04593737e8f` |
| [workspace]/experiments/cgsr_ablation_window_rgb_fullval_v1/scores/cityscapes.json | `5a3258eaa2cb1785aeeb0cc03785ce6a25149c187f0dc1837a119f81498f9b96` |
| [workspace]/experiments/cgsr_efficiency_author_v1/efficiency_results.json | `b8ee52eebd5004367d853fc431383f51d7db3a77e3ecaae8625d9a36ba80946f` |

## 3. Not included (by design)

* Upstream ReCLIP++ code: cloned by the user at commit `46298007fcfa75b0ca41c3a9a8420b2dfa800475` (no license).
* Candidate generator `cgsr/candidate/model_sfp_dtlr.py`: provenance unresolved. The paper-facing file has SHA-256 `7877892bfbd34e883a5b66bb7cd29cebd785140bf50fb0922b2793a809406c89`.
* Checkpoints, text embeddings, datasets, predictions and figures: see `docs/WEIGHTS.md`, `docs/DATASETS.md` and `RELEASE_REPORT.md`.
