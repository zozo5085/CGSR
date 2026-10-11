#!/usr/bin/env bash
# Clone the base-predictor code at the commit used for the paper-facing results,
# then verify the files CGSR imports. The base predictor is ReCLIP++ (Wang & Kang);
# its code is not redistributed in this repository.
# Usage: bash scripts/setup_upstream.sh [destination]   (default: third_party/base_predictor)
set -euo pipefail

DEST="${1:-third_party/base_predictor}"
URL="https://github.com/dogehhh/ReCLIP.git"
COMMIT="46298007fcfa75b0ca41c3a9a8420b2dfa800475"

if [ -e "$DEST" ]; then
    echo "Using existing directory: $DEST"
else
    git clone "$URL" "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout "$COMMIT"
python tools/verify_assets.py --base-code-root "$DEST" --skip-candidate
