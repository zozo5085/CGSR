# Third-party notices

This repository includes no third-party source **file**. It does contain four small third-party-derived items, each attributed below:
* the STEGO COCO-Stuff fine-to-coarse table, copied verbatim (MIT);
* `cgsr/evaluation.py::shared_postprocess`, a re-implementation of the ReCLIP++ evaluation post-processing;
* the CLIP zero-shot text-embedding recipe in `tools/make_text_embeddings.py`, following OpenAI CLIP (MIT);
* config values and colour palettes in `configs/`, which follow the ReCLIP++ configuration files.

All other third-party components are either fetched by the user from their own source or used as named dependencies. The table lists every component the release touches.

| Component | Upstream | License | How it is used here | Files in this repository derived from it | Modified? | Required attribution |
|---|---|---|---|---|---|---|
| ReCLIP++ (Base predictor, evaluator, configs) | https://github.com/dogehhh/ReCLIP, commit `46298007fcfa75b0ca41c3a9a8420b2dfa800475` | **None stated.** The repository has no license file in any commit, and the GitHub API reports no license. | **Not redistributed.** The user clones it (`scripts/setup_upstream.*`); `tools/eval_cgsr.py` imports `model/model.py`, `utils/preprocess.py`, `config/configs.py` and `utils/cls_token.pt` from that checkout, unmodified. | `cgsr/evaluation.py::shared_postprocess` re-implements the ReCLIP++ evaluation post-processing (class-presence filter `PD`, bilinear resize, softmax, argmax). The YAML files in `configs/` follow the ReCLIP++ config schema and keep its values, including the colour `PALETTE` lists. | n/a | Cite: J. Wang and G. Kang, "ReCLIP++: Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation", IJCV 133(11):7946–7968, 2025. |
| ReCLIP++ author checkpoints | released by the ReCLIP++ authors | not stated | **Not redistributed.** Users download them (`docs/WEIGHTS.md`). | none | n/a | as above |
| OpenAI CLIP | https://github.com/openai/CLIP, commit `dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1` | MIT, Copyright (c) 2021 OpenAI | pip dependency (`requirements.txt`). ViT-B/16 weights are downloaded by `clip.load`. | `tools/make_text_embeddings.py::zeroshot_classifier` follows the zero-shot classifier recipe of the CLIP prompt-engineering notebook. The text embeddings in `docs/WEIGHTS.md` are CLIP outputs. | adapted (device handling, output layout) | MIT, Copyright (c) 2021 OpenAI |
| STEGO COCO-Stuff fine-to-coarse table | https://github.com/mhamilton723/STEGO (`src/data.py`) | MIT, Copyright (c) Mark Hamilton | label conversion for COCO-Stuff-27 | `tools/convert_coco_stuff27_labels.py` (the `FINE_TO_COARSE` table only) | no (table copied verbatim) | MIT notice below |
| Candidate generator (SFP + DTLR) | ported from a third-party research script (author and license **not confirmed**) | unknown | **Not included** (`cgsr/candidate/README.md`) | none | n/a | pending |
| PASCAL-Context 59-class list | https://cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt | dataset terms | the class names are listed in `cgsr/context_h1.py` | `cgsr/context_h1.py::OFFICIAL_59_LABELS` | no | dataset citation |
| mmsegmentation (concept only) | https://github.com/open-mmlab/mmsegmentation | Apache-2.0 | none of its code is included. The per-class intersection/union convention in `cgsr/evaluation.py` matches the common mmsegmentation evaluation. | none | n/a | none |

Datasets (PASCAL VOC, PASCAL Context, ADE20K, COCO / COCO-Stuff, Cityscapes) are not redistributed; see `docs/DATASETS.md`.

## Role names used in the code

The code refers to components by architecture-neutral role names. The renaming changes no attribution: each role name below refers to a third-party component, which keeps its authorship and license status as listed above. None of them is new code of this work, except `SelectiveRollbackVerifier` / `local_class_support` and the `forward()` of `CGSRPipeline`.

| Role name in this repository | Refers to | Defined where | Origin |
|---|---|---|---|
| `FrozenBasePredictor` | the base predictor class `RECLIPPP` of the upstream ReCLIP++ `model/model.py` | user's upstream checkout, imported unmodified (`cgsr/model_cgsr.py`) | ReCLIP++ (Wang & Kang); no license |
| base predictor weights | the author-released ReCLIP++ checkpoints | downloaded by the user (`docs/WEIGHTS.md`) | ReCLIP++ authors |
| base config | `configs/*.yaml` (ReCLIP++ config schema and values) | this repository | follows ReCLIP++ configuration files |
| `shared_postprocess` | the ReCLIP++ evaluation post-processing | `cgsr/evaluation.py` (re-implementation) | ReCLIP++ (re-implemented) |
| `FixedCandidateGenerator` | the SFP + DTLR candidate generator (class `RECLIPPP` in `cgsr/candidate/model_sfp_dtlr.py`) | distributed separately; not included | third-party research script, provenance unresolved |
| `SelectiveRollbackVerifier`, `local_class_support` | the CGSR operator (CGSR-owned core) | `cgsr/verifier.py`; imports only PyTorch | this work |
| `CGSRPipeline` | the experimental integration: two-pass `forward()` connecting base, candidate and verifier | `cgsr/model_cgsr.py` | this work: the `forward()` integration only. It subclasses `FixedCandidateGenerator`, which subclasses `FrozenBasePredictor`; the inherited code keeps its origin as listed above. |

---

## MIT license text: STEGO (applies to the `FINE_TO_COARSE` table in `tools/convert_coco_stuff27_labels.py`)

```text
MIT License

Copyright (c) Mark Hamilton. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
