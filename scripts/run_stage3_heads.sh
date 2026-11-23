#!/bin/bash
# Stage 3a: better heads on the saved frozen features (no raw images needed).
# aug8 for the DINOv2 probes, feature concatenation across backbones, RBF SVM.
# Run from the project root: bash scripts/run_stage3_heads.sh
set -e
PY=.venv/bin/python; [ -x "$PY" ] || PY=.venv/Scripts/python.exe     # macOS / Windows
P() { $PY scripts/12_linear_probe.py "$@" 2>&1 | grep -v -i warn; }
S336=vit_small_patch14_dinov2.lvd142m_336
B224=vit_base_patch14_dinov2.lvd142m_224
S224=vit_small_patch14_dinov2.lvd142m_224
CNX=convnext_tiny.fb_in22k_ft_in1k_224
P --feat $B224 --mode t0
P --feat $S336 --mode aug8
P --feat $B224 --mode aug8
P --feat $S336,$B224 --mode mean8
P --feat $S336,$B224,$CNX --mode mean8
P --feat $S336 --mode mean8 --clf svm
P --feat $B224 --mode mean8 --clf svm
P --feat $S336,$B224 --mode mean8 --clf svm
