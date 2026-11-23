#!/bin/bash
# Stage 3c: end-to-end fine-tuning (5 folds each), then the lecture CNN with D4
# augmentation (interrupted on the M1; ~20 min on an RTX 4060).
# Run from the project root: bash scripts/run_stage3_finetune.sh
# ViT-B at lr 1e-4 failed to train on fold 1 (loss stuck at the class prior;
# out/logs/ft_dinov2_b336_lr1e-4.log), so ViT-B uses lr 5e-5 on every fold.
set -e
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_DISABLE_TELEMETRY=1
PY=.venv/bin/python; [ -x "$PY" ] || PY=.venv/Scripts/python.exe     # macOS / Windows
F() { $PY -u scripts/15_finetune.py "$@" 2>&1 | grep --line-buffered -v -i "warn\|symlink" || true; }
F --model vit_small_patch14_dinov2.lvd142m --res 336 > out/logs/ft_dinov2_s336.log
F --model vit_base_patch14_dinov2.lvd142m --res 336 --lr 5e-5 --tag lr5e-5 > out/logs/ft_dinov2_b336_lr5e-5.log
$PY -u scripts/10_lecture_cnn.py --aug d4 2>&1 | grep --line-buffered -v -i warn > out/logs/lecture_cnn_d4.log
