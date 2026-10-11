"""Locate and verify the base-predictor code checkout, then load the CGSR pipeline.

The base predictor code is the upstream ReCLIP++ repository (Wang & Kang). It
carries no license, so it is not redistributed in this repository. The user
clones it at the pinned commit (``scripts/setup_upstream.*``), and this module
makes it importable without modifying the checkout:

* ``<base_code_root>`` is put on ``sys.path`` and becomes the working directory,
  because the upstream code opens ``utils/cls_token.pt`` with a relative path.
* ``cgsr.model_cgsr`` then imports the base predictor (``model.model``) and the
  separately distributed candidate generator (``cgsr.candidate``).
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path

from . import frozen

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_FILE = REPO_ROOT / "cgsr" / "candidate" / "model_sfp_dtlr.py"

SETUP_HINT = (
    "Clone the base-predictor code (upstream ReCLIP++) at the pinned commit first:\n"
    "    bash scripts/setup_upstream.sh third_party/base_predictor      (Linux/macOS)\n"
    "    powershell -File scripts/setup_upstream.ps1 third_party/base_predictor   (Windows)\n"
    f"Upstream: {frozen.BASE_CODE_REPO} @ {frozen.BASE_CODE_COMMIT}"
)


class SetupError(SystemExit):
    """Raised for missing or unexpected inputs; carries a readable message."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_content_sha256(path: Path) -> str:
    """SHA-256 of a saved tensor's values (C-contiguous float32 bytes)."""
    import torch

    t = torch.load(str(path), map_location="cpu")
    return _sha256_bytes(t.detach().float().contiguous().numpy().tobytes())


def text_embedding_status(path: Path, name: str) -> tuple[str, str]:
    """Return (status, file_sha256) for a text embedding, accepting a content match."""
    digest = sha256_file(path)
    if digest == frozen.TEXT_EMBEDDING_SHA256.get(name):
        return "match", digest
    expected = frozen.TEXT_EMBEDDING_CONTENT_SHA256.get(name)
    if expected is not None and tensor_content_sha256(path) == expected:
        return "match (tensor content)", digest
    return "MISMATCH", digest


def verify_base_code(base_code_root: Path, allow_mismatch: bool = False) -> dict:
    """Check that ``base_code_root`` holds the pinned base-predictor files.

    Returns a report dict. Raises ``SetupError`` on a missing file, and on a
    hash mismatch unless ``allow_mismatch`` is set.
    """
    root = Path(base_code_root).resolve()
    if not root.is_dir():
        raise SetupError(f"--base-code-root does not exist: {root}\n{SETUP_HINT}")
    report, mismatches = {}, []
    for rel, expected in frozen.BASE_CODE_TEXT_SHA256.items():
        path = root / rel
        if not path.is_file():
            raise SetupError(f"base-predictor file missing: {path}\n{SETUP_HINT}")
        got = _sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        report[rel] = got
        if got != expected:
            mismatches.append(rel)
    for rel, expected in frozen.BASE_CODE_BINARY_SHA256.items():
        path = root / rel
        if not path.is_file():
            raise SetupError(f"base-predictor file missing: {path}\n{SETUP_HINT}")
        got = sha256_file(path)
        report[rel] = got
        if got != expected:
            mismatches.append(rel)
    if mismatches and not allow_mismatch:
        raise SetupError(
            "base-predictor files differ from the pinned commit "
            f"{frozen.BASE_CODE_COMMIT}: {mismatches}\n"
            "Numerical parity with the paper is only established for the pinned files. "
            "Re-run the setup script, or pass --allow-base-code-mismatch to continue anyway."
        )
    return {"root": str(root), "sha256": report, "mismatches": mismatches}


def activate_base_code(base_code_root: Path):
    """Make the base-predictor code importable and return ``CGSRPipeline``."""
    if not CANDIDATE_FILE.is_file():
        raise SetupError(
            f"Candidate generator not found: {CANDIDATE_FILE}\n"
            "It is distributed separately; see cgsr/candidate/README.md."
        )
    root = Path(base_code_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(root)  # the upstream code opens utils/cls_token.pt relative to its root
    importlib.import_module("model.model")
    return importlib.import_module("cgsr.model_cgsr").CGSRPipeline
