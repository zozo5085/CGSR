"""Check the base-code checkout and the downloaded weights against the paper-facing hashes.

No GPU and no model construction; safe to run anywhere.

    python tools/verify_assets.py --base-code-root third_party/base_predictor --weights-dir weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cgsr import frozen  # noqa: E402
from cgsr.upstream import CANDIDATE_FILE, sha256_file, text_embedding_status, verify_base_code  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-code-root", type=Path, default=None,
                   help="checkout of the base-predictor code (upstream ReCLIP++)")
    p.add_argument("--weights-dir", type=Path, default=None,
                   help="directory holding <ckpt>.pth and text/<embedding>.pth")
    p.add_argument("--skip-candidate", action="store_true",
                   help="do not require cgsr/candidate/model_sfp_dtlr.py (used by the setup scripts)")
    args = p.parse_args(argv)
    if args.base_code_root is None and args.weights_dir is None:
        p.error("give --base-code-root and/or --weights-dir")

    ok = True
    if args.base_code_root is not None:
        report = verify_base_code(args.base_code_root, allow_mismatch=True)
        for rel, digest in report["sha256"].items():
            flag = "MISMATCH" if rel in report["mismatches"] else "ok"
            print(f"[base code] {flag:8s} {rel}  {digest}")
        ok &= not report["mismatches"]
        if not args.skip_candidate:
            print(f"[candidate] {'present' if CANDIDATE_FILE.is_file() else 'MISSING'}  {CANDIDATE_FILE}")
            ok &= CANDIDATE_FILE.is_file()

    if args.weights_dir is not None:
        root = args.weights_dir
        for name, expected in frozen.CHECKPOINT_SHA256.items():
            path = root / name
            if not path.is_file():
                print(f"[weights] missing  {path}")
                ok = False
                continue
            got = sha256_file(path)
            ok &= got == expected
            print(f"[weights] {'ok' if got == expected else 'MISMATCH':8s} {path}  {got}")
        for name in frozen.TEXT_EMBEDDING_SHA256:
            path = root / "text" / name
            if not path.is_file():
                print(f"[weights] missing  {path}")
                ok = False
                continue
            status, got = text_embedding_status(path, name)
            ok &= status != "MISMATCH"
            print(f"[weights] {status:8s} {path}  {got}")

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
