# GitHub push checklist

Review this checklist before the first push. Items explicitly marked as public-release blockers may remain open for a private repository, but must be resolved before making the repository public. Items marked **(open)** were unresolved when the release folder was built (`RELEASE_REPORT.md` §19).

- [ ] **License chosen** (open; **public-release blocker**). Add `LICENSE`, delete `LICENSE_PENDING.md`, and update `README.md` §License.
- [ ] **Third-party redistribution checked** (open):
  - Candidate generator `model_sfp_dtlr.py` (**public-release blocker**): either rights are confirmed (then follow the README in the held-back staging folder kept outside this repository), or it stays out.
  - ReCLIP++ (base predictor): no code is included; confirm you are happy with the overlay design and the role-name wording in `THIRD_PARTY_NOTICES.md`.
  - STEGO table: MIT notice present.
- [x] No datasets in the tree.
- [x] No private or local checkpoints. No `.pth/.pt/.ckpt/.npz` files are in the tree, and `.gitignore` blocks them.
- [x] No secrets found by the scan (`docs/RELEASE_CHECKS.md`). Re-run the scan if you edit files.
- [x] No absolute local (drive-letter or home-directory) paths in code, configs or docs (`docs/RELEASE_CHECKS.md`).
- [x] No file > 100 MB; the largest file is listed in `docs/RELEASE_CHECKS.md`.
- [ ] **Text-embedding distribution decision** (open). The checkpoint hashes and source links are already verified (`docs/WEIGHTS.md`). The remaining decision is whether to attach the five exact text embeddings to a GitHub Release.
- [ ] **README commands tested**:
  - Done: CPU-side commands (`--help`, error paths, `verify_assets.py`, config parsing, module loading).
  - Not done (open): the GPU commands (`tools/eval_cgsr.py`, `tools/make_text_embeddings.py` on GPU). Suggested: a 20-image smoke per dataset.
- [x] Expected results documented (`docs/RESULTS.md`, `REPRODUCIBILITY.md` §5, `cgsr/frozen.py`).
- [x] Source/config SHA-256 recorded (`RELEASE_MANIFEST.md`).
- [ ] **Citation metadata reviewed** (open). The author names in `CITATION.cff` are TODO placeholders.
- [ ] **Paper figures safe to redistribute** (open). No figures are included. The paper figures embed dataset photographs (VOC, ADE20K, Cityscapes); decide before adding any.
- [ ] Git identity. Commits must use your own name/e-mail, with no AI co-author trailers. Check `git config user.name` / `user.email` before committing.
- [ ] Remote. Create the GitHub repository yourself, private first if unsure, then add it as `origin`.

## Commands (run yourself; nothing below has been executed)

```bash
cd <path-to-this-repository>
git status                      # repository already initialised, nothing staged
git add .
git status                      # review: no weights/, outputs/, third_party/, cgsr/candidate/model_sfp_dtlr.py
git commit -m "Initial CGSR release"
git branch -M main
git remote add origin <USER_REPO_URL>
git push -u origin main
```

If the Candidate generator is cleared for release, remove `cgsr/candidate/model_sfp_dtlr.py` from `.gitignore` before `git add .`, then copy the file in from the held-back staging folder.
