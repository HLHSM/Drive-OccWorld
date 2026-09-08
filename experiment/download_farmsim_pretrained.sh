#!/usr/bin/env bash
# Download the two external baseline initializations required by FarmSim.
# The destinations are checked before they are made visible to the configs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

download() {
  local url="$1"
  local destination="$2"
  local expected_bytes="$3"
  local partial="${destination}.part"

  mkdir -p "$(dirname "${destination}")"
  if [[ -f "${destination}" ]] && [[ "$(stat -c%s "${destination}")" == "${expected_bytes}" ]]; then
    echo "Verified: ${destination}"
    return
  fi
  rm -f "${destination}"
  echo "Downloading ${destination} (resumable)"
  curl --fail --location --retry 5 --continue-at - --output "${partial}" "${url}"
  [[ "$(stat -c%s "${partial}")" == "${expected_bytes}" ]] || {
    echo "Incomplete download: ${partial}" >&2
    exit 1
  }
  mv "${partial}" "${destination}"
  echo "Verified: ${destination}"
}

download \
  'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth' \
  "${ROOT}/pretrained/sparseocc/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth" \
  308515644

download \
  'https://huggingface.co/Dobbin/OccStudio/resolve/main/pretrain/bevdet-r50-4d-stereo-cbgs.pth?download=true' \
  "${ROOT}/pretrained/cotr/bevdet-r50-4d-stereo-cbgs.pth" \
  233059365

