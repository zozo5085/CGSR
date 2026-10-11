# License: decision pending

No license has been chosen yet for the original code in this repository. **Until a `LICENSE` file is added, no license is granted.**

Original files that a license would cover:

* `cgsr/` (except `cgsr/candidate/`, see below)
* `tools/eval_cgsr.py`, `tools/verify_assets.py`, `tools/make_text_embeddings.py` (CLIP recipe: MIT, see notices), `tools/convert_coco_stuff27_labels.py`
* `scripts/`, `configs/` (values), `docs/`, `results/`, and the top-level documents

Not covered by a license chosen for this repository:

* **Upstream ReCLIP++ code.** It is not included; users clone it from its own repository, which currently carries no license (`THIRD_PARTY_NOTICES.md`).
* **The Candidate generator** (`cgsr/candidate/model_sfp_dtlr.py`). It is not included; its provenance is unresolved (`cgsr/candidate/README.md`).
* **The COCO-Stuff fine-to-coarse table** in `tools/convert_coco_stuff27_labels.py`. It is MIT-licensed (STEGO) and keeps its notice.

To finish: choose a license, add it as `LICENSE`, delete this file, and update `README.md` §License.
