"""Generate the CLIP ViT-B/16 text embeddings used by the evaluation.

Recipe: the standard CLIP zero-shot classifier (OpenAI CLIP prompt-engineering
notebook, MIT). For each class, every prompt template is formatted with the
class name, encoded, L2-normalised, averaged, and L2-normalised again. The
result is a ``[C, 512]`` float32 tensor. Class names and the prompt templates
are imported from the base-predictor code checkout (upstream ReCLIP++), so their
order and spelling match the base predictor.

On the paper environment (RTX 5090, PyTorch 2.11 nightly, CUDA 12.8) this
reproduces the VOC, Context-59, ADE20K and COCO-Stuff-27 files bit-exactly. The
Cityscapes file is reproduced only up to floating-point noise (max |diff|
4.9e-4). For bit-exact Cityscapes parity, use the original file
(docs/WEIGHTS.md). Other devices may differ in the last bits.

    python tools/make_text_embeddings.py --base-code-root third_party/base_predictor --out-dir weights/text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cgsr import frozen  # noqa: E402
from cgsr.upstream import text_embedding_status, verify_base_code  # noqa: E402

CLASS_LIST = {
    "voc": "voc_classes",
    "context59": "pascal_context_classes",
    "ade20k": "ade_classes",
    "coco_stuff27": "coco_stuff_classes_27",
    "cityscapes": "cityscapes_classes",
}


def zeroshot_classifier(model, classnames, templates, device):
    import clip
    import torch

    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [t.format(classname) for t in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights.permute(1, 0).float().cpu()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-code-root", required=True, type=Path,
                   help="checkout of the base-predictor code (upstream ReCLIP++)")
    p.add_argument("--out-dir", required=True, type=Path, help="output directory, e.g. weights/text")
    p.add_argument("--dataset", default="all", choices=["all", *sorted(frozen.DATASETS)])
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = p.parse_args(argv)

    verify_base_code(args.base_code_root, allow_mismatch=False)
    sys.path.insert(0, str(args.base_code_root.resolve()))
    import clip
    import torch
    import utils.preprocess as up

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = clip.load("ViT-B/16", device=device)
    model = model.to(device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(frozen.DATASETS) if args.dataset == "all" else [args.dataset]
    for ds in names:
        out = args.out_dir / frozen.DATASETS[ds]["text_embedding"]
        if out.exists():
            print(f"skip {out} (exists; delete it to regenerate)")
            continue
        classes = getattr(up, CLASS_LIST[ds])
        w = zeroshot_classifier(model, classes, up.prompt_templates, device)
        torch.save(w, out)
        status, digest = text_embedding_status(out, frozen.DATASETS[ds]["text_embedding"])
        flag = "matches the paper file" if status != "MISMATCH" else "differs from the paper file (see docstring)"
        print(f"{ds}: {tuple(w.shape)} -> {out}  file sha256 {digest}  [{flag}]")


if __name__ == "__main__":
    main()
