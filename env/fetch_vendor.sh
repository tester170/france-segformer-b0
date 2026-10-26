#!/usr/bin/env bash
# Original DINOv3 code (github.com/facebookresearch/dinov3) at the commit of the experiment, into vendor/dinov3.
# Needed only for the ConvNeXt-B + UPerNet config (DINOv3 ConvNeXt backbone); SegFormer-B0 does not use it.
# Fetched at exactly that commit (git fetch by hash). Idempotent. Usage: bash env/fetch_vendor.sh
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL=https://github.com/facebookresearch/dinov3.git
COMMIT=6876159a11b4df116f30f667f8c9888617df0751
d="$BUNDLE/vendor/dinov3"
if [ "$(git -C "$d" rev-parse HEAD 2>/dev/null)" = "$COMMIT" ]; then
  echo "dinov3: already at ${COMMIT:0:7}"; exit 0
fi
echo "== dinov3: $URL @ ${COMMIT:0:7}"
rm -rf "$d"; mkdir -p "$d"
git -C "$d" init -q
git -C "$d" fetch -q --depth 1 "$URL" "$COMMIT"
git -C "$d" checkout -q FETCH_HEAD
