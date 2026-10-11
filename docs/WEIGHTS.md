# Weights and model assets

No weight file is distributed in this repository. Download or generate the files below, and place them at the expected paths. Then verify them:

```bash
python tools/verify_assets.py --base-code-root third_party/base_predictor --weights-dir weights
```

`tools/eval_cgsr.py` checks the same hashes. It stops on a mismatch unless `--skip-hash-check` is given.

## Summary

| Asset | Dataset | Required | Size (bytes) | SHA-256 | Expected path | Distribution |
|---|---|:---:|---:|---|---|---|
| Base predictor weights: ReCLIP++ author checkpoint (rectification stage) | PASCAL VOC | yes | 658,427,542 | `f5e68096ed6ed364ab395c22b116932bc37b5c6d3af8901169b07a6f77faf23b` | `weights/voc.pth` | NOT REDISTRIBUTED; [author link](https://drive.google.com/file/d/1FjTNj8PPGlce4xAQIXwjhi020glX3fSW/view?usp=drive_link) |
| Base predictor weights: ReCLIP++ author checkpoint | PASCAL Context | yes | 660,733,078 | `21d78dc19ff3b8d3527e83da65c04cd9e178ac507712e89d57fc7f2288ba2f97` | `weights/context.pth` | NOT REDISTRIBUTED; [author link](https://drive.google.com/file/d/1gYfh2fY5EyWCH8RXLgEq-bzcGToT40SM/view?usp=drive_link) |
| Base predictor weights: ReCLIP++ author checkpoint | ADE20K | yes | 667,296,086 | `8b05b8c5f7d93bcb52dac134ea1f58a313421f47474702d3f2e80101e43380d9` | `weights/ade.pth` | NOT REDISTRIBUTED; [author link](https://drive.google.com/file/d/1p1b7WaaFHiNbq6ssvY1KZgqJ9G5XUKBr/view?usp=drive_link) |
| Base predictor weights: ReCLIP++ author checkpoint | COCO-Stuff | yes | 658,818,838 | `2c5b55de933df0a7d1a01716c2c55a07e9a954127a0f35934a7667dc24519371` | `weights/coco.pth` | NOT REDISTRIBUTED; [author link](https://drive.google.com/file/d/1kqvNi_2J-IE0Tg9Rs2T2kkM5S_KRZsUz/view?usp=drive_link) |
| Base predictor weights: ReCLIP++ author checkpoint | Cityscapes | yes | 658,372,438 | `a7daa5dad6e2f4132e3b9c24fdd8027ef22155b08d1f15295953aa383f7d3399` | `weights/city.pth` | NOT REDISTRIBUTED; [author link](https://drive.google.com/file/d/18Hdorz5nbLS5DFbXFcXKvgknzyD0W4sQ/view?usp=drive_link) |
| CLIP ViT-B/16 text embedding (20 × 512) | VOC | yes | 42,621 | `561365cf5569d24334d4a3da5014813fcee2a91d90d1bfcf97ebfded0e654207` | `weights/text/voc_ViT16_clip_text.pth` | not redistributed (yet); generate with `tools/make_text_embeddings.py` |
| CLIP ViT-B/16 text embedding (59 × 512, alphabetical class order) | Context-59 | yes | 122,585 | `ae450fd92d5a63a5ab5b091fdfb06f355c70e9d60c940f4911c8ecef747e956a` | `weights/text/context_ViT16_clip_text_alpha.pth` | as above |
| CLIP ViT-B/16 text embedding (150 × 512) | ADE20K | yes | 308,861 | `5410010b4df79fb87bad93cc7f6db607b8399164406821d78f4a8d32e564578e` | `weights/text/ade_ViT16_clip_text.pth` | as above |
| CLIP ViT-B/16 text embedding (27 × 512) | COCO-Stuff-27 | yes | 57,049 | `34044aa7c3c1a16f2b04e59727fa5b40ad80188d7f8a505431783c6c31c81f7f` | `weights/text/stuff27_ViT16_clip_text.pth` | as above |
| CLIP ViT-B/16 text embedding (19 × 512) | Cityscapes | yes | 40,580 | `28b83c5465091a51884fee1b1d6e211a0c928d2b612318d08562dfc9ae7d1169` | `weights/text/city_ViT16_clip_text.pth` | as above (regeneration is not bit-exact; see below) |
| Learned reference-prompt token and decoder | all | yes | — | — | inside each author checkpoint (`text_encoder.prompt_token`, `decoder*`) | comes with the checkpoint |
| CLIP ViT-B/16 image/text encoder | all | yes | — | verified by the `clip` package from its download URL | `~/.cache/clip/ViT-B-16.pt` | downloaded automatically by `clip.load("ViT-B/16")` |
| `utils/cls_token.pt` | all | yes | — | `153b2bdc5fa225e78adaea34675db71fae8f6c4e74437af3987ed0c4197745ab` | inside the upstream checkout | comes with the upstream clone |

The candidate generator and CGSR add **no** weights.

## Checkpoints

The five checkpoints are the **author-released ReCLIP++ rectification-stage checkpoints**.
* **Source.** The links come from the "Rectification" table of the official ReCLIP++ README (https://github.com/dogehhh/ReCLIP, `README.md` lines 139–143 at commit `46298007fcfa75b0ca41c3a9a8420b2dfa800475`).
* **Verification.** Files downloaded from these links on 2026-09-26 have exactly the SHA-256 values listed above.
* **Redistribution.** The upstream repository states no license or redistribution terms, so the files are **NOT REDISTRIBUTED** here. Obtain each checkpoint from the upstream release and place it at the expected path. This repository does not provide a download script, because download automation is not covered by any stated terms.
* **Locally trained checkpoints.** Some were used during development, for robustness checks only. They are not author weights and are not released. Every paper-facing number uses the author checkpoints above.

## Text embeddings

Each embedding is a `[C, 512]` float32 tensor of CLIP ViT-B/16 text features. Every class is encoded with 85 prompt templates; each template embedding is L2-normalised, the templates are averaged, and the mean is L2-normalised again. Class names and templates come from the upstream ReCLIP++ code. The Context-59 embedding uses the upstream **alphabetical** class order, which is the order the Context checkpoint was trained with. `tools/eval_cgsr.py` maps predictions to the official order afterwards.

Generate all five from the upstream checkout:

```bash
python tools/make_text_embeddings.py --base-code-root third_party/base_predictor --out-dir weights/text
```

Reproduction status, measured on the paper environment (RTX 5090, PyTorch 2.11 nightly, CUDA 12.8):

| Dataset | Regenerated tensor vs paper file |
|---|---|
| VOC, Context-59, ADE20K, COCO-Stuff-27 | bit-identical tensor values |
| Cityscapes | not bit-identical: max abs diff 4.9e-4, min per-class cosine 0.999999 |

**File hashes vs content hashes.** Re-saving an identical tensor changes the `.pth` file hash. The tools therefore also accept a **tensor-content** SHA-256 match (`cgsr/frozen.py::TEXT_EMBEDDING_CONTENT_SHA256`).

**Exactness caveats:**
* Other GPUs or library versions may change the last bits.
* For bit-exact Cityscapes parity you need the original file; the regenerated one is close but not identical.

Whether the five exact files will be attached to a GitHub Release is an open decision (`RELEASE_REPORT.md`).

## Expected layout

```text
weights/
├── voc.pth  context.pth  ade.pth  coco.pth  city.pth
└── text/
    ├── voc_ViT16_clip_text.pth
    ├── context_ViT16_clip_text_alpha.pth
    ├── ade_ViT16_clip_text.pth
    ├── stuff27_ViT16_clip_text.pth
    └── city_ViT16_clip_text.pth
```

`weights/CHECKSUMS.sha256` lists the same file hashes in `sha256sum -c` format, relative to `weights/`.
