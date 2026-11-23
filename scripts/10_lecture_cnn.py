"""Baseline: the small CNN from lecture 14, re-implemented in PyTorch.

Architecture and training follow the lecture's Keras code:
  Rescaling(1/255) -> Conv2D(32, 5, relu) -> MaxPool(2) -> Conv2D(64, 5, relu)
  -> GlobalMaxPooling -> Dense(1, sigmoid); Adam (lr 1e-3), batch 32, 15 epochs,
  Glorot-uniform weights and zero biases (the Keras defaults).
Differences from the lecture: 5-fold CV on out/folds.csv instead of one 80/20
split, and PyTorch instead of Keras.

--aug d4 reproduces the lecture's augmentation: each epoch sees every training
image under all 8 rotations/flips (8x the data); validation is unchanged.

Outputs: out/lecture_cnn_<aug>_curve.csv (validation AUC per fold and epoch),
         out/oof/lecture_cnn_<aug>.csv, out/test_preds/lecture_cnn_<aug>.csv,
         a row in out/experiments.csv.
Run: .venv/bin/python scripts/10_lecture_cnn.py --aug none   (then --aug d4)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, SEED, load_folds  # noqa: E402
from dl import DEVICE, d4, fold_aucs, load_split, log_experiment, save_preds, set_seed, to_tensor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--aug", choices=["none", "d4"], default="none")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--folds", type=str, default="0,1,2,3,4")
args = ap.parse_args()
folds_to_run = [int(f) for f in args.folds.split(",")]
name = f"lecture_cnn_{args.aug}"


class LectureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 5)
        self.c2 = nn.Conv2d(32, 64, 5)
        self.out = nn.Linear(64, 1)
        for m in (self.c1, self.c2, self.out):      # Keras defaults
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):                              # x in [0, 1]
        x = torch.max_pool2d(torch.relu(self.c1(x)), 2)
        x = torch.relu(self.c2(x))
        x = torch.amax(x, dim=(-2, -1))                # global max pooling
        return self.out(x).squeeze(-1)                 # logit


@torch.no_grad()
def predict(model, X, bs=64):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(X[i:i + bs].to(DEVICE))).cpu())
    return torch.cat(out).numpy()


folds = load_folds()
y = folds["label"].values
fold = folds["fold"].values
_, Xtr_u8 = load_split("train")
_, Xte_u8 = load_split("test")
X = to_tensor(Xtr_u8); del Xtr_u8
Xte = to_tensor(Xte_u8); del Xte_u8

oof = np.full(len(y), np.nan)
test_pred = np.zeros(len(Xte))
curve = []
t0 = time.time()
for k in folds_to_run:
    set_seed(SEED + k)
    tr_idx = np.where(fold != k)[0]
    va_idx = np.where(fold == k)[0]
    model = LectureCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    lossf = nn.BCEWithLogitsLoss()
    n_t = 8 if args.aug == "d4" else 1
    for ep in range(args.epochs):
        model.train()
        # every (image, transform) pair once per epoch, shuffled
        pairs = np.array([(i, t) for i in tr_idx for t in range(n_t)])
        pairs = pairs[np.random.permutation(len(pairs))]
        for b in range(0, len(pairs), 32):
            bi, bt = pairs[b:b + 32, 0], pairs[b:b + 32, 1]
            xb = X[bi]
            for t in np.unique(bt):
                m = torch.from_numpy(bt == t)
                xb[m] = d4(xb[m], int(t))
            xb = xb.to(DEVICE)
            yb = torch.from_numpy(y[bi]).float().to(DEVICE)
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        p = predict(model, X[va_idx])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[va_idx], p)
        curve.append({"fold": k, "epoch": ep + 1, "val_auc": auc})
        print(f"fold {k} epoch {ep + 1:2d}: val AUC {auc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    oof[va_idx] = p                                    # final epoch, as in the lecture
    test_pred += predict(model, Xte) / len(folds_to_run)

pd.DataFrame(curve).to_csv(OUT / f"{name}_curve.csv", index=False)
if len(folds_to_run) == 5:
    auc, sd, per = fold_aucs(y, oof, fold)
    print(f"\n{name}: pooled OOF AUC {auc:.3f}; per-fold {np.round(per, 3).tolist()} (SD {sd:.3f})")
    save_preds(name, oof, test_pred)
    log_experiment({"name": name, "stage": 2, "model": "lecture CNN", "features": "",
                    "res": 256, "aug": args.aug, "tta": "none", "classifier": "end-to-end",
                    "oof_auc": round(auc, 4), "fold_sd": round(sd, 4),
                    "minutes": round((time.time() - t0) / 60, 1)})
