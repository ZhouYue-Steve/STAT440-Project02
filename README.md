# Project 2: Kilns — satellite image classification of brick kilns

**Task.** Given a 256×256 satellite image, predict the kiln type: 1 = zigzag kiln, 0 = fixed chimney kiln (FCK).
Submissions are scored by test-set AUC (score = 2×(AUC − 0.5)).
Training set: 1617 images (64.4% zigzag). Test set: 724 images. Deadline: December 9, 17:00.

**Contest rules.** The contest data must not be uploaded to an AI, and must not be augmented with outside data (pretrained networks are allowed). All scripts run locally and print only aggregate statistics. The data is **not** included in this repository (see [Data](#3-data)).

**Final submissions** (both nested-CV AUC 0.9984, contest score ≈ 99.7):

| File | What it is |
|---|---|
| `submissions/ens_stack.csv` | **Primary.** Logistic stacking of 29 models' out-of-fold predictions |
| `submissions/ft_dinov2_b336.csv` | **Backup.** Fine-tuned DINOv2 ViT-B/14 at 336 px alone (mean of the 5 fold models) |

A Chinese version of an earlier README is kept in [`README_zh.md`](README_zh.md).

---

## 1. Environment (stage 0)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### 1.1 macOS (Apple M1, 8 GB, macOS 14.6)
- **torch is pinned to 2.6.0 / torchvision 0.21.0.** torch 2.15 crashes as soon as it uses MPS on macOS 14 (it calls an API that only exists in macOS 15).
- MPS training speed (synthetic data, ~1294 training images per fold):

| Model | Resolution | Batch | Per epoch | 5 folds × 15 epochs |
|---|---|---|---|---|
| ResNet50 | 224 | 32 | ~50 s | ~1 h |
| EfficientNet-B0 | 256 | 32 | ~34 s | ~45 min |
| ConvNeXt-Tiny | 224 | 16 | **out of memory** | — |

- Feature extraction (inference only) works for every model on MPS. Fine-tuning (`15_finetune.py`) assumes CUDA (bf16 autocast on `cuda`) and is not supported on the M1.

### 1.2 Windows + NVIDIA GPU (second half of stage 2, and stage 3)
Stage 2 (second half) and stage 3 ran on Windows 11 + RTX 4060 (8 GB). torch is still pinned to 2.6.0, but the CUDA build is needed. Install the other dependencies first, then force the cu124 torch; otherwise timm pulls in the CPU build:

```bash
uv venv --python 3.11 .venv
grep -v -E '^(torch|torchvision)==' requirements.txt > req_notorch.txt
uv pip install --python .venv/Scripts/python.exe -r req_notorch.txt
uv pip install --python .venv/Scripts/python.exe --reinstall-package torch --reinstall-package torchvision \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

On Windows, replace `.venv/bin/python` below with `.venv/Scripts/python.exe` (the `scripts/*.sh` files pick the right one). `src/dl.py` selects the device automatically: CUDA, then MPS, then CPU.
Results from the M1 reproduce on CUDA (e.g. the DINOv2-S@336 linear probe: OOF AUC 0.932, same C chosen in every fold).
CUDA speed: lecture CNN with D4 augmentation, 5 folds in ~19 min (over 11 h on the M1); DINOv2-S@336 fine-tuning, 5 folds in ~10 min.

## 2. Directory layout

```
project02/
├── data/            contest data (not in the repo; see section 3)
├── src/common.py    paths, image loading, folds, writing and validating submissions
├── src/dl.py        deep-learning helpers (D4 transforms, loading all images, saving OOF/test predictions, experiment log)
├── scripts/         run in numbered order
├── out/             intermediate results
│   ├── folds.csv        the fixed 5-fold split
│   ├── features/        frozen features (each image × 8 D4 transforms)
│   ├── oof/             out-of-fold predictions of every model (training set)
│   ├── test_preds/      test-set predictions (mean of the 5 fold models)
│   ├── logs/            training logs
│   └── experiments.csv  summary of all experiments (printed by 13_summary.py)
└── submissions/     submission files
```

## 3. Data

Download the contest data and unzip it so that the layout is:

```
data/train/data.csv      training labels
data/train/<index>.png   1617 training images
data/test/<index>.png    724 test images
```

`data/` is listed in `.gitignore`. Scripts that only use saved features or predictions (`12_`, `13_`, `14_`) run without `data/`: `common.list_test_ids()` then reads the test ids stored in `out/features/*.npz`.

## 4. Pipeline and run order

Run every script from the project root: `.venv/bin/python scripts/<script>`.

### 4.1 Stage 1: data checks and CV split

| # | Script | Purpose | Output | Time |
|---|---|---|---|---|
| 0 | `00_test_submission.py` | Tests submission writing and validation (should print `ALL SUBMISSION TESTS PASSED`) | `submissions/constant_0.5.csv` | seconds |
| 1 | `01_near_duplicates.py` | Exact and rotated/flipped near-duplicates (pHash + thumbnail correlation, minimum over the 8 D4 transforms) | `out/near_dup_pairs.csv`, `out/nn_distance.csv` | ~15 s |
| 1b | `01b_shifted_overlap.py` | Partly overlapping (shifted) tiles: 64 px phase-correlation screen over all 2.7 M image pairs | `out/overlap_pairs.csv`, `out/overlap_nn.csv` | ~1.5 min |
| 1c | `01c_verify_overlap.py` | Full-resolution check of the screened candidates: pixel correlation on the overlap, compared with random pairs and calibrated on synthetic overlapping tiles | `out/overlap_verified.csv` | ~5 s |
| 2 | `02_folds.py` | Fixed stratified 5-fold split (seed 440); refuses to overwrite a different existing split | **`out/folds.csv`** | seconds |
| 3 | `03_shortcut_check.py` | Shortcut check with simple image features, a lower bound from simple models, train/test adversarial validation | `out/simple_features.csv` | ~25 s |

**Every submission is written with `common.write_submission(scores, name)`**, which validates the format: header `index,score`, 724 rows, sorted by index, scores in [0, 1].

### 4.2 Stages 2–3: models and ensembles

| # | Script | Purpose | Output |
|---|---|---|---|
| 10 | `10_lecture_cnn.py --aug none\|d4` | The small CNN from lecture 14 (rewritten in PyTorch), 5 folds; `d4` = every epoch trains on all 8 rotations/flips of each image | `out/oof/lecture_cnn_*.csv` etc. |
| 11 | `11_extract_features.py --model M --res R` | Frozen features of a pretrained network (timm), each image × 8 D4 transforms | `out/features/<M>_<R>.npz` |
| 12 | `12_linear_probe.py --feat F[,F2,...] --mode t0\|mean8\|aug8 [--clf svm]` | Logistic regression on frozen features (C chosen by inner CV); comma-separated features are concatenated; `--clf svm` uses an RBF SVM instead | `out/oof/probe_*.csv` etc. |
| 13 | `13_summary.py` | Prints `out/experiments.csv` sorted by OOF AUC | — |
| 14 | `14_ensemble.py` | Ensembles of the OOF predictions (top-k rank average, greedy selection, logistic stacking), scored by **nested CV**; writes the best one as a submission | `submissions/ens_<method>.csv`, `out/ensemble_weights.csv` |
| 15 | `15_finetune.py --model M --res R [--lr --llrd --epochs]` | End-to-end DINOv2 fine-tuning, 5 folds; last-epoch model + D4 TTA (CUDA only) | `out/oof/ft_*.csv`, `out/ft_*_curve.csv` |
| sh | `run_stage2_ablation.sh` / `run_stage3_heads.sh` / `run_stage3_finetune.sh` | Run the experiments above in order | `out/logs/*.log` |

`submissions/ft_dinov2_b336.csv` is a copy of `out/test_preds/ft_vit_base_patch14_dinov2.lvd142m_336_lr5e-5.csv`.

**Evaluation.** Every AUC below is the **pooled out-of-fold (OOF) AUC** on `out/folds.csv`: each training image is scored by the fold model that did not see it. Hyperparameters are either fixed in advance or chosen inside the training folds (inner CV); the outer validation fold is never used for tuning.

## 5. Stage 1 findings

### 5.1 No duplicate or overlapping images → plain stratified 5-fold CV
- Identical files: 0.
- Rotated/flipped near-duplicates: the smallest nearest-neighbour pHash distance is 9 (64-bit hash), with a continuous distribution and no "very close" cluster. The highest thumbnail correlation is only 0.876. Among the 49 training pairs with pHash ≤ 12, the label agreement is 0.61, close to 0.54 for random pairs.
- Shifted overlaps: calibration shows that at full resolution, overlaps of ≥ 50% are fully recovered (correlation 1.0), and the 64 px screen reliably catches overlaps of ≥ 70%. The 314 candidate pairs have a maximum overlap correlation of 0.57, no different from the random-pair maximum of 0.62.
- **Limitations:** tiles overlapping by less than ~50% may be missed by the screen; the same site imaged on another date or under different lighting would also have low pixel correlation.

### 5.2 The 5-fold split
323–324 images per fold, zigzag share 0.64 in every fold, matching the full set.

### 5.3 No obvious shortcut
- Every single simple feature has AUC between 0.48 and 0.55.
- Simple features + logistic regression / gradient boosting: CV AUC **0.61–0.63**. A CNN must clearly beat this lower bound.
- "Image property" features only (file size, sharpness, dark/bright pixels): logistic regression 0.50, gradient boosting 0.575. The signal is weak and may just reflect the land cover around the kiln.
- The image index is unrelated to the label: neighbouring indices agree in label 0.552 of the time (permutation test p = 0.18); in 5 index blocks, the zigzag share is 0.63–0.68.

### 5.4 Train and test come from the same distribution
- Adversarial validation (without the index): logistic regression AUC 0.505, gradient boosting 0.548; training and test images are essentially indistinguishable.
- With the index, AUC = 1.0, only because all test indices come after the training ones; this is meaningless.

## 6. Stage 2 findings: frozen features + linear probe

| Model (mean8 = mean of the 8 D4 feature copies) | 224 | 336 | 384/448 |
|---|---|---|---|
| ResNet50 (ImageNet) | 0.870 | — | — |
| EfficientNet-B0 | 0.850 | — | — |
| ConvNeXt-Tiny (in22k) | 0.875 | — | 0.880 (384) |
| DINOv2 ViT-S/14 | 0.890 | 0.932 | 0.927 (448) |
| DINOv2 ViT-B/14 | 0.922 | 0.950 | 0.952 (448) |
| DINOv2 ViT-L/14 | 0.942 | **0.958** | — |

- **DINOv2 (self-supervised) is clearly better than supervised ImageNet CNNs**, and larger is better (S < B < L).
- **Going from 224 to 336 px helps a lot** (+0.03–0.04); 448 adds almost nothing. Upsampling the 256 px image lets the 14 px patches cover finer structure.
- D4 averaging (TTA) consistently adds about 0.03 (t0 → mean8); training on the 8 transforms as extra samples (aug8) adds only +0.002 more.
- Concatenating features (S336 + B224) gives 0.945, better than either alone; an RBF SVM is worse than logistic regression (0.90–0.92).
- The lecture-14 CNN: 0.616 without augmentation, 0.735 with D4 augmentation. A small network trained from scratch is far behind pretrained features.

## 7. Stage 3 findings: end-to-end fine-tuning and ensembles

### 7.1 DINOv2 fine-tuning (`15_finetune.py`)
Setup: 336 px, random D4 + ±10% brightness/contrast jitter, AdamW (weight decay 0.05), layer-wise LR decay 0.75, 1 warm-up epoch + cosine decay, stochastic depth 0.1, bf16, batch 16, 10 epochs. The last-epoch model is used (no early stopping on the validation fold), with 8-way D4 TTA at prediction time.

| Model | LR (head) | OOF AUC | Per fold | Time (RTX 4060) |
|---|---|---|---|---|
| DINOv2 ViT-S/14 @336 | 1e-4 | 0.9952 | 0.999 / 0.998 / 0.996 / 0.996 / 0.993 | 10.5 min |
| DINOv2 ViT-B/14 @336 | 1e-4 | 0.9771 | 1.000 / **0.623** / 0.999 / 1.000 / 0.995 | 29.5 min |
| **DINOv2 ViT-B/14 @336** | **5e-5** | **0.9984** | 1.000 / 0.999 / 0.998 / 1.000 / 0.995 | 30.1 min |

- **Fine-tuning beats the best frozen features (0.958) by about 0.04**, raising the score from 91.6 to 99.7 (contest score = 200×(AUC − 0.5)).
- ViT-B at lr 1e-4 **failed to train on fold 1**: the training loss stayed around 0.65 (about the entropy of the 64/36 class prior), i.e. it only learned the base rate. This is visible from the training loss alone, independent of the validation set. ViT-B's learning rate was therefore lowered to 5e-5 and **all 5 folds were rerun** (not just the failed one). The failed run is kept as `*_lr1e-4` in the logs but excluded from ensembles.
- Learning curves (`out/ft_*_curve.csv`): every fold plateaus by epoch 7–8 and does not degrade afterwards, so using the last epoch costs nothing; D4 TTA adds about +0.003.

### 7.2 Ensembles (`14_ensemble.py`, nested CV)
Selecting models and scoring them on the same OOF predictions is optimistic, so for each method the models/weights are chosen on 4 folds' OOF predictions and scored on the 5th.

| Method | Nested AUC | Fold SD | On the same OOF (optimistic) |
|---|---|---|---|
| Logistic stacking (logits of 29 models) | **0.9984** | **0.0008** | 0.9990 |
| Greedy selection (picked only the ViT-B fine-tune) | 0.9984 | 0.0017 | 0.9984 |
| Top-2 rank average (the two fine-tunes) | 0.9978 | 0.0018 | 0.9978 |
| Top-3 … top-6 | 0.9964 → 0.9910 | | |

- The best single model (ViT-B fine-tune) already reaches 0.9984; ensembling brings no measurable gain on the OOF predictions. The differences (≤ 0.0006) are far below the noise: the public and private test halves have about 362 images each, and the AUC sampling error is about ±0.002.
- Stacking puts the largest coefficient on the ViT-B fine-tune (0.43, on the logit scale), then the DINOv2-L@336 probe (0.13); the rest goes to weaker models that are less correlated with them.

### 7.3 Submissions
| File | Content | Nested/OOF AUC |
|---|---|---|
| **`submissions/ens_stack.csv`** | **Primary**: logistic stacking, highest nested AUC and most stable across folds | 0.9984 |
| `submissions/ft_dinov2_b336.csv` | Backup: ViT-B fine-tune only (mean of the 5 fold models) | 0.9984 |

Both files pass `validate_submission`; on the test set, 64% of scores are > 0.5 in both, matching the training zigzag share. They tie on OOF AUC but rank the test images differently (Spearman 0.856), so they are two genuinely different bets. Submit both and look at the public leaderboard, but the public leaderboard covers only half the test images, so do not over-select on it.

### 7.4 Limitations
- OOF AUC is close to 1 and few errors remain, so a handful of images decides the ranking; differences of 0.001 between methods are not reliable.
- Fine-tuning used a single random seed; averaging several seeds might stabilise the test predictions, but no further gain is detectable on the OOF predictions.

## 8. Open-source code and models

- [timm](https://github.com/huggingface/pytorch-image-models): model definitions and pretrained weights (ResNet50, EfficientNet-B0, ConvNeXt-Tiny, DINOv2 ViT-S/B/L).
- [DINOv2](https://github.com/facebookresearch/dinov2) (Oquab et al., 2023): the self-supervised pretrained ViTs.
- Greedy ensemble selection follows Caruana et al. (2004), *Ensemble Selection from Libraries of Models*.
- The lecture CNN follows the Keras code from STAT 440 lecture 14.
