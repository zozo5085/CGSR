#!/usr/bin/env bash
# Evaluate Base / Candidate / CGSR on one dataset with the paper-facing protocol.
# Usage: bash scripts/eval.sh <dataset> <data_root> [extra tools/eval_cgsr.py args]
#   dataset: voc | context59 | ade20k | coco_stuff27 | cityscapes
# Expects base predictor weights in weights/<ckpt>.pth, text embeddings in
# weights/text/<embedding>.pth (docs/WEIGHTS.md) and the base-code checkout in
# third_party/base_predictor (scripts/setup_upstream.sh); override with BASE_CODE_ROOT.
set -euo pipefail

DATASET="${1:?dataset}"
DATA_ROOT="${2:?data root}"
shift 2
BASE_CODE_ROOT="${BASE_CODE_ROOT:-third_party/base_predictor}"

case "$DATASET" in
    voc)          CKPT=voc.pth;     TEXT=voc_ViT16_clip_text.pth ;;
    context59)    CKPT=context.pth; TEXT=context_ViT16_clip_text_alpha.pth ;;
    ade20k)       CKPT=ade.pth;     TEXT=ade_ViT16_clip_text.pth ;;
    coco_stuff27) CKPT=coco.pth;    TEXT=stuff27_ViT16_clip_text.pth ;;
    cityscapes)   CKPT=city.pth;    TEXT=city_ViT16_clip_text.pth ;;
    *) echo "unknown dataset: $DATASET" >&2; exit 2 ;;
esac

python tools/eval_cgsr.py \
    --dataset "$DATASET" \
    --base-code-root "$BASE_CODE_ROOT" \
    --data-root "$DATA_ROOT" \
    --checkpoint "weights/$CKPT" \
    --text-embedding "weights/text/$TEXT" \
    --output-dir "outputs/$DATASET" \
    "$@"
