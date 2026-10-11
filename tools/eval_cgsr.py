"""Evaluate the base predictor, the fixed candidate generator and CGSR on one dataset.

One ``CGSRPipeline`` forward call per image returns all three logit states:
``base_logits`` (Base), ``candidate_logits`` (Candidate) and ``final_logits``
(CGSR). Each is passed through the same shared post-processing
(``shared_postprocess``) and scored on the full validation split. The output directory receives ``result.json`` (mIoU,
paired-bootstrap CIs, guards, environment) and ``per_image_iu.npz``.

Example (after scripts/setup_upstream.* and placing the weights):

    python tools/eval_cgsr.py --dataset voc \
        --base-code-root third_party/base_predictor \
        --data-root /path/to/VOCdevkit/VOC2012 \
        --checkpoint weights/voc.pth \
        --text-embedding weights/text/voc_ViT16_clip_text.pth \
        --output-dir outputs/voc
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cgsr import frozen  # noqa: E402
from cgsr.data import build_val_list  # noqa: E402

METHODS = ("base", "candidate", "cgsr")
STATE_KEY = {"base": "base_logits", "candidate": "candidate_logits", "cgsr": "final_logits"}
CONTRASTS = (
    ("candidate_minus_base", "candidate", "base"),
    ("cgsr_minus_base", "cgsr", "base"),
    ("cgsr_minus_candidate", "cgsr", "candidate"),
)
SMOKE_LABEL = "SMOKE ONLY -- NOT A SCIENTIFIC RESULT"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate Base / Candidate / CGSR with the frozen paper protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=sorted(frozen.DATASETS))
    p.add_argument("--base-code-root", required=True, type=Path,
                   help="checkout of the base-predictor code (upstream ReCLIP++) at the pinned commit "
                        "(scripts/setup_upstream.*)")
    p.add_argument("--data-root", required=True, type=Path,
                   help="dataset root, see docs/DATASETS.md")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="base predictor weights (author-released checkpoint), see docs/WEIGHTS.md")
    p.add_argument("--text-embedding", required=True, type=Path,
                   help="CLIP text embedding .pth for the dataset, see docs/WEIGHTS.md")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="new directory for result.json / per_image_iu.npz (must not exist)")
    p.add_argument("--config", type=Path, default=None,
                   help="override the dataset YAML (default: configs/<dataset>/...)")
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate only the first N images (smoke test; not a result)")
    p.add_argument("--bootstrap", type=int, default=frozen.BOOTSTRAP_REPLICATES,
                   help="paired bootstrap replicates (0 disables)")
    p.add_argument("--seed", type=int, default=frozen.BOOTSTRAP_SEED, help="bootstrap seed")
    p.add_argument("--save-predictions", action="store_true",
                   help="also write per-image label PNGs for each method")
    p.add_argument("--skip-hash-check", action="store_true",
                   help="do not stop on checkpoint / text-embedding SHA-256 mismatch")
    p.add_argument("--allow-base-code-mismatch", action="store_true",
                   help="do not stop when base-code files differ from the pinned commit")
    return p.parse_args(argv)


def check_inputs(args, spec):
    """Fail early, before any heavy import, on missing or unexpected inputs."""
    from cgsr.upstream import SetupError, sha256_file, text_embedding_status

    for flag, path in (("--checkpoint", args.checkpoint), ("--text-embedding", args.text_embedding)):
        if not path.is_file():
            raise SetupError(f"{flag} not found: {path}\nSee docs/WEIGHTS.md for how to obtain it.")
    ckpt_sha = sha256_file(args.checkpoint)
    text_status, text_sha = text_embedding_status(args.text_embedding, spec["text_embedding"])
    hashes = {"checkpoint": ckpt_sha, "text_embedding": text_sha}
    status = {
        "checkpoint": "match" if ckpt_sha == frozen.CHECKPOINT_SHA256[spec["checkpoint"]] else "MISMATCH",
        "text_embedding": text_status,
    }
    for key, reference in (("checkpoint", spec["checkpoint"]), ("text_embedding", spec["text_embedding"])):
        if status[key] == "MISMATCH":
            msg = (f"{key} {getattr(args, key)} does not match the paper-facing file {reference} "
                   "(SHA-256 in docs/WEIGHTS.md)")
            if not args.skip_hash_check:
                raise SetupError(msg + "\nUse --skip-hash-check to evaluate a different file anyway.")
            print("WARNING: " + msg, flush=True)
    return hashes, status


def build_pipeline(cfg_path, args, spec, pipeline_cls):
    import clip
    import torch
    from config.configs import cfg_from_file
    from utils.preprocess import prepare_dataset_cls_tokens

    cfg = cfg_from_file(str(cfg_path))
    assert cfg.DATASET.NAME == spec["config_name"], (cfg.DATASET.NAME, spec["config_name"])
    assert int(cfg.DATASET.NUM_CLASSES) == spec["num_classes"]
    assert bool(cfg.DATASET.REDUCE_ZERO_LABEL) == spec["reduce_zero_label"]
    assert float(cfg.TEST.PD) == spec["pd"]
    # The base predictor's text encoder (upstream code) reads the prompt token
    # from cfg.LOAD_PATH at construction time; point it at the evaluated checkpoint.
    cfg.LOAD_PATH = str(args.checkpoint)
    cfg.DATASET.DATAROOT = str(args.data_root).replace("\\", "/").rstrip("/") + "/"
    cfg.DATASET.TEXT_WEIGHT = str(args.text_embedding)

    device = torch.device("cuda")
    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)
    pipeline = pipeline_cls(cfg=cfg, clip_model=clip_model, rank=0)
    weights = torch.load(str(args.checkpoint), map_location="cpu")
    weights = {(k[7:] if k.startswith("module.") else k): v for k, v in weights.items()}
    missing, unexpected = pipeline.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match the pipeline: missing={list(missing)[:5]} "
                         f"unexpected={list(unexpected)[:5]}")
    pipeline = pipeline.to(device).eval()

    # Guard 1: every checkpoint tensor equals the loaded pipeline state.
    state = pipeline.state_dict()
    mismatched = [k for k, v in weights.items()
                  if k in state and not torch.equal(v.detach().cpu().to(state[k].dtype), state[k].detach().cpu())]
    if mismatched:
        raise SystemExit(f"checkpoint tensors differ from the loaded pipeline: {mismatched[:5]}")
    # Guard 2: the candidate generator runs with the frozen settings.
    attrs = {}
    for name, value in frozen.CANDIDATE_ATTRIBUTES.items():
        got = getattr(pipeline, name)
        got = tuple(got) if isinstance(value, tuple) else got
        attrs[name] = got
        if got != value:
            raise SystemExit(f"candidate setting {name}={got!r} differs from the frozen value {value!r}")

    text_weight = torch.load(str(args.text_embedding), map_location="cpu")
    cls_tokens, _ = prepare_dataset_cls_tokens(cfg)
    return cfg, pipeline, text_weight, cls_tokens, device, attrs


def env_info():
    import torch
    return dict(
        python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        platform=platform.platform(),
    )


def main(argv=None):
    args = parse_args(argv)
    spec = frozen.DATASETS[args.dataset]
    cfg_path = (args.config or (REPO_ROOT / spec["config"])).resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.text_embedding = args.text_embedding.resolve()
    args.data_root = args.data_root.resolve()
    args.base_code_root = args.base_code_root.resolve()
    out = args.output_dir.resolve()

    from cgsr.upstream import CANDIDATE_FILE, SetupError, activate_base_code, sha256_file, verify_base_code

    if not cfg_path.is_file():
        raise SetupError(f"config not found: {cfg_path}")
    if out.exists() and any(out.iterdir()):
        raise SetupError(f"refusing to overwrite a non-empty output directory: {out}")
    ids, images, labels = build_val_list(args.dataset, args.data_root)
    if args.limit:
        ids, images, labels = ids[:args.limit], images[:args.limit], labels[:args.limit]
    elif len(ids) != spec["num_val_images"]:
        print(f"WARNING: {len(ids)} validation images found, the paper protocol uses "
              f"{spec['num_val_images']}", flush=True)
    hashes, hash_status = check_inputs(args, spec)
    base_code_report = verify_base_code(args.base_code_root, args.allow_base_code_mismatch)
    if not CANDIDATE_FILE.is_file():
        raise SetupError(f"Candidate generator not found: {CANDIDATE_FILE}\n"
                         "It is distributed separately; see cgsr/candidate/README.md.")

    import numpy as np
    import torch
    from PIL import Image
    from cgsr.evaluation import bootstrap, intersection_union, miou, raw_label, shared_postprocess

    if not torch.cuda.is_available():
        raise SetupError("A CUDA GPU is required: the paper-facing numbers were produced on CUDA, "
                         "and the base-predictor code loads CUDA-saved tensors.")

    pipeline_cls = activate_base_code(args.base_code_root)
    from utils.preprocess import val_preprocess

    cfg, pipeline, text_weight, cls_tokens, device, attrs = build_pipeline(cfg_path, args, spec, pipeline_cls)

    lut = None
    if args.dataset == "context59":
        from utils.preprocess import pascal_context_classes
        from cgsr.context_h1 import alphabetical_to_official_lut
        lut = alphabetical_to_official_lut(pascal_context_classes)

    out.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        for m in METHODS:
            (out / "pred" / m).mkdir(parents=True, exist_ok=True)

    n, K = len(ids), spec["num_classes"]
    rzl, pd = spec["reduce_zero_label"], spec["pd"]
    I = {m: np.zeros((n, K + 1), np.int64) for m in METHODS}
    U = {m: np.zeros((n, K + 1), np.int64) for m in METHODS}
    t0 = time.time()
    for i, (image_path, label_path) in enumerate(zip(images, labels)):
        with open(image_path, "rb") as stream:
            image = val_preprocess(cfg, stream.read()).unsqueeze(0).to(device)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            _, states = pipeline(image, [], text_weight, cls_tokens, training=False, return_debug=True)
        label = raw_label(label_path, rzl)
        shape = tuple(label.shape[:2])
        for m in METHODS:
            pred = shared_postprocess(states[STATE_KEY[m]], tuple(image.shape[-2:]), shape, pd)
            if lut is not None:
                pred = torch.from_numpy(lut[pred.cpu().numpy()])
            I[m][i], U[m][i] = intersection_union(pred, label, K)
            if args.save_predictions:
                Image.fromarray(pred.cpu().numpy().astype(np.uint8)).save(out / "pred" / m / f"{ids[i]}.png")
        if (i + 1) % 250 == 0 or i + 1 == n:
            print(f"[{args.dataset}] {i + 1}/{n} images, {time.time() - t0:.0f} s", flush=True)

    np.savez_compressed(out / "per_image_iu.npz", image_ids=np.asarray(ids),
                        **{f"I_{m}": I[m] for m in METHODS}, **{f"U_{m}": U[m] for m in METHODS})

    point = {m: miou(I[m].sum(0), U[m].sum(0), K) for m in METHODS}
    deltas = {}
    if args.bootstrap > 0:
        for name, a, b in CONTRASTS:
            deltas[name] = bootstrap(
                lambda ix, a=a, b=b: 100 * (miou(I[a][ix].sum(0), U[a][ix].sum(0), K)
                                            - miou(I[b][ix].sum(0), U[b][ix].sum(0), K)),
                n, args.bootstrap, args.seed,
            )

    smoke = bool(args.limit)
    parity = None
    if not smoke:
        exp = dict(zip(METHODS, frozen.EXPECTED_MIOU[args.dataset]))
        diff = {m: abs(point[m] - exp[m]) for m in METHODS}
        parity = dict(expected=exp, abs_diff=diff, passed=max(diff.values()) <= 1e-9)

    result = dict(
        status=SMOKE_LABEL if smoke else "full validation split",
        dataset=args.dataset, n_images=n, num_classes=K, pd=pd,
        miou={m: point[m] for m in METHODS},
        miou_pct={m: 100 * point[m] for m in METHODS},
        deltas_pp=deltas,
        bootstrap=dict(replicates=args.bootstrap, seed=args.seed, unit="image", paired=True,
                       primary_interval="95% percentile"),
        paper_parity=parity,
        inputs=dict(config=str(cfg_path), config_sha256=sha256_file(cfg_path),
                    checkpoint_sha256=hashes["checkpoint"], text_embedding_sha256=hashes["text_embedding"],
                    hash_status=hash_status),
        base_code=dict(repo=frozen.BASE_CODE_REPO, commit=frozen.BASE_CODE_COMMIT,
                       sha256=base_code_report["sha256"], mismatches=base_code_report["mismatches"]),
        candidate_settings={k: list(v) if isinstance(v, tuple) else v for k, v in attrs.items()},
        context_label_order="official 59_labels.txt (H1)" if lut is not None else None,
        env=env_info(),
        seconds=time.time() - t0,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    tag = f"  [{SMOKE_LABEL}]" if smoke else ""
    print(f"[{args.dataset}] mIoU  Base {100 * point['base']:.4f}  Candidate {100 * point['candidate']:.4f}"
          f"  CGSR {100 * point['cgsr']:.4f}{tag}")
    for name, d in deltas.items():
        lo, hi = d["percentile"]
        print(f"  {name:<22} {d['point']:+.4f} mIoU points  95% CI [{lo:+.4f}, {hi:+.4f}]{tag}")
    if parity is not None:
        print(f"  paper parity (|diff| <= 1e-9): {'PASS' if parity['passed'] else 'FAIL'}")
    print(f"  wrote {out / 'result.json'}")


if __name__ == "__main__":
    main()
