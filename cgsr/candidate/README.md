# Candidate generator (SFP + DTLR): distributed separately

`tools/eval_cgsr.py` expects the fixed Candidate generator at

```text
cgsr/candidate/model_sfp_dtlr.py
```

**Why the file is not in this repository yet.** Its core refinement logic was ported from a third-party research script whose authorship and license have not been confirmed. Until redistribution rights are confirmed, the file is not committed. It is listed in `.gitignore` so that it cannot be added by accident.

**What the file must be.** The paper-facing numbers were produced with a file whose SHA-256 is:

```text
7877892bfbd34e883a5b66bb7cd29cebd785140bf50fb0922b2793a809406c89
```

The algorithm and every setting are documented in `docs/METHOD.md` §3. The frozen settings are asserted at run time against `cgsr/frozen.py::CANDIDATE_ATTRIBUTES`.

**What happens without it.** `tools/eval_cgsr.py` and `tools/verify_assets.py` stop with a message pointing to this file. CGSR itself (`SelectiveRollbackVerifier` in `cgsr/verifier.py`) is independent of how the Candidate is produced; it only consumes `base_logits`, `candidate_logits` and `projected_features`. In the code, this generator is referred to as `FixedCandidateGenerator` (`cgsr/candidate/__init__.py`); the file keeps its original class name.

The Candidate generator is an existing refinement stage used as the object of verification. It is not part of the CGSR contribution.
