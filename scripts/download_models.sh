#!/usr/bin/env bash
set -euo pipefail
checkpoint_dir="${1:-checkpoints}"
mkdir -p "$checkpoint_dir"

fetch() {
  local name="$1" url="$2" hash="$3"
  local target="$checkpoint_dir/$name"
  if [[ ! -f "$target" ]]; then
    curl --fail --location --retry 2 --continue-at - --output "$target.part" "$url"
    printf '%s  %s\n' "$hash" "$target.part" | sha256sum --check
    mv -- "$target.part" "$target"
  else
    printf '%s  %s\n' "$hash" "$target" | sha256sum --check
  fi
}

fetch ffhq.pkl \
  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl' \
  a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c
fetch e4e_ffhq_encode.pt \
  'https://drive.usercontent.google.com/download?id=1cUv_reLE6k3604or78EranS7XzuVMWeO&export=download&confirm=t' \
  2ace1d9a8c05c10a399bcd500b8dda118f759ff1aac89dbdab7435f2136a0999
fetch model_ir_se50.pth \
  'https://drive.usercontent.google.com/download?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn&export=download&confirm=t' \
  a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28
fetch FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  'https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth' \
  ab3534c4c2377a853926b99456fc4a10140ea092213b9c8d32458e06ec0961b7
