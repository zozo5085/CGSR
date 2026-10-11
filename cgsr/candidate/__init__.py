"""Fixed candidate generator (SFP + DTLR refinement stage), distributed separately.

``FixedCandidateGenerator`` is the class defined in ``model_sfp_dtlr.py`` (see
README.md in this folder for its distribution status and provenance). That
file is used byte-for-byte as frozen. It keeps its original class name, and it
imports the base predictor from the upstream checkout, so the upstream code
must be importable first (``cgsr.upstream.activate_base_code``).
"""

from pathlib import Path

_CANDIDATE_FILE = Path(__file__).with_name("model_sfp_dtlr.py")
if not _CANDIDATE_FILE.is_file():
    raise ImportError(
        f"Candidate generator not found: {_CANDIDATE_FILE}\n"
        "It is distributed separately; see cgsr/candidate/README.md."
    )

from .model_sfp_dtlr import RECLIPPP as FixedCandidateGenerator  # noqa: E402  (class name defined by the frozen file)

__all__ = ["FixedCandidateGenerator"]
