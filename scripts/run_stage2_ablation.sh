#!/bin/bash
# Stage 2 resolution / model-size ablation: extract frozen features, then
# linear probe (mean of the 8 rotation/flip feature copies).
# Run from the project root: bash scripts/run_stage2_ablation.sh
set -e
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_DISABLE_TELEMETRY=1
PY=.venv/bin/python
run() {  # model res
  f="out/features/$1_$2.npz"
  [ -f "$f" ] || $PY scripts/11_extract_features.py --model "$1" --res "$2" 2>&1 | grep -v -i warn
  $PY scripts/12_linear_probe.py --feat "$1_$2" --mode mean8 2>&1 | grep -v -i warn
}
run vit_small_patch14_dinov2.lvd142m 336
run vit_base_patch14_dinov2.lvd142m 224
run convnext_tiny.fb_in22k_ft_in1k 384
run vit_small_patch14_dinov2.lvd142m 448
run vit_base_patch14_dinov2.lvd142m 336
