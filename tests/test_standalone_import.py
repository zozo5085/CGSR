"""Clean-environment import test for the CGSR-owned core.

The verifier must be importable and runnable without the base-predictor
checkout (``third_party/base_predictor``) and without the separately
distributed candidate generator (``cgsr/candidate/model_sfp_dtlr.py``). The test
copies only the ``cgsr`` package (without the candidate file) to a temporary
directory and runs a fresh interpreter there, so nothing else on disk can be
picked up.

    python tests/test_standalone_import.py      (or: pytest tests/)
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

USER_COMMAND = "from cgsr import SelectiveRollbackVerifier, local_class_support; print('PASS')"

CHECKS = """
import importlib.util, sys
from pathlib import Path
import torch
import cgsr
from cgsr import SelectiveRollbackVerifier, local_class_support
assert importlib.util.find_spec("model") is None, "base-predictor code is importable"
assert not (Path(cgsr.__file__).parent / "candidate" / "model_sfp_dtlr.py").exists(), "candidate file is present"
assert not {"model", "cgsr.candidate", "cgsr.model_cgsr", "cgsr.evaluation", "cgsr.upstream"} & set(sys.modules)
torch.manual_seed(0)
base, cand = torch.randn(4, 6, 7), torch.randn(4, 6, 7)
feat = torch.nn.functional.normalize(torch.randn(8, 6, 7), dim=0)
final, states = SelectiveRollbackVerifier()(base, cand, feat)
accept = states["accept"]
assert final.shape == base.shape and accept.dtype == torch.bool
assert torch.equal(final, torch.where(accept.unsqueeze(0).expand(4, -1, -1), cand, base))
assert local_class_support(states["base_probs"], feat).shape == base.shape
print("PASS")
"""


def _run(code, cwd):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True)


def test_standalone_import():
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(REPO_ROOT / "cgsr", Path(tmp) / "cgsr",
                        ignore=shutil.ignore_patterns("model_sfp_dtlr.py", "__pycache__"))
        for code in (USER_COMMAND, CHECKS):
            out = _run(code, tmp)
            assert out.returncode == 0 and out.stdout.strip().splitlines()[-1] == "PASS", out.stderr


if __name__ == "__main__":
    test_standalone_import()
    print("PASS: standalone import of the CGSR core")
