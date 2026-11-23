"""Linear probe: logistic regression on frozen pretrained features (from 11_).

How the 8 rotation/flip (D4) feature copies are used (--mode):
  t0     only the original orientation (no augmentation, no TTA)
  mean8  average the 8 copies into one vector (TTA in feature space)
  aug8   train on all 8 copies as separate rows (augmentation), predict by
         averaging the 8 predicted probabilities (TTA)

Within each outer fold of out/folds.csv, features are standardised and the
L2 penalty C is chosen by an inner 5-fold CV on AUC (copies of one image are
kept in the same inner fold), so the outer-fold AUC is never used for tuning.
Test predictions are the mean of the 5 fold models.

--feat may list several feature files separated by commas; their features are
concatenated (each is standardised, so every dimension gets equal weight).
--clf svm swaps logistic regression for an RBF-kernel SVM (C chosen the same
way; t0/mean8 only); its decision value is squashed by a sigmoid into [0, 1].

Outputs: out/oof/<name>.csv, out/test_preds/<name>.csv, row in out/experiments.csv
Run: .venv/bin/python scripts/12_linear_probe.py --feat resnet50.a1_in1k_224 --mode mean8
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, SEED, load_folds  # noqa: E402
from dl import fold_aucs, log_experiment, save_preds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True, help="file stem(s) in out/features/, comma-separated")
ap.add_argument("--mode", choices=["t0", "mean8", "aug8"], default="mean8")
ap.add_argument("--clf", choices=["logistic", "svm"], default="logistic")
args = ap.parse_args()
feats = args.feat.split(",")
assert not (args.clf == "svm" and args.mode == "aug8"), "svm: use t0 or mean8"

t0 = time.time()
folds = load_folds()
Ftr, Fte = [], []
for f in feats:
    z = np.load(OUT / "features" / f"{f}.npz")
    assert z["train_ids"].tolist() == folds["index"].tolist()
    Ftr.append(z["train"].astype(np.float32))          # (N, 8, D)
    Fte.append(z["test"].astype(np.float32))
Ftr, Fte = np.concatenate(Ftr, -1), np.concatenate(Fte, -1)
y = folds["label"].values
fold = folds["fold"].values
N, T, D = Ftr.shape
Cs = np.logspace(-5, 1, 13)


def rows(F, idx):
    """Design matrix, labels, groups for images idx under the chosen mode."""
    if args.mode == "t0":
        return F[idx, 0], idx
    if args.mode == "mean8":
        return F[idx].mean(1), idx
    return F[idx].reshape(-1, D), np.repeat(idx, T)          # aug8


def predict(model, scaler, F, idx):
    if args.mode == "aug8":
        p = model.predict_proba(scaler.transform(F[idx].reshape(-1, D)))[:, 1]
        return p.reshape(len(idx), T).mean(1)
    X, _ = rows(F, idx)
    if args.clf == "svm":
        return 1 / (1 + np.exp(-model.decision_function(scaler.transform(X))))
    return model.predict_proba(scaler.transform(X))[:, 1]


oof = np.zeros(N)
test = np.zeros(len(Fte))
chosen = []
for k in range(5):
    tr = np.where(fold != k)[0]
    va = np.where(fold == k)[0]
    X, g = rows(Ftr, tr)
    yy = y[g]
    scaler = StandardScaler().fit(X)
    inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, yy, groups=g))
    if args.clf == "svm":
        gs = GridSearchCV(SVC(kernel="rbf", gamma="scale"), {"C": np.logspace(-1, 2, 7)},
                          cv=inner, scoring="roc_auc", n_jobs=-1)
        model = gs.fit(scaler.transform(X), yy).best_estimator_
        chosen.append(model.C)
    else:
        model = LogisticRegressionCV(Cs=Cs, cv=inner, scoring="roc_auc", max_iter=5000, n_jobs=-1)
        model.fit(scaler.transform(X), yy)
        chosen.append(model.C_[0])
    oof[va] = predict(model, scaler, Ftr, va)
    test += predict(model, scaler, Fte, np.arange(len(Fte))) / 5

auc, sd, per = fold_aucs(y, oof, fold)
name = f"probe_{'+'.join(feats)}_{args.mode}" + ("_svm" if args.clf == "svm" else "")
print(f"{name}: OOF AUC {auc:.3f} (fold SD {sd:.3f}); per-fold {np.round(per, 3).tolist()}; "
      f"C chosen {[f'{c:.0e}' for c in chosen]}; {time.time() - t0:.0f}s")
save_preds(name, oof, test)
model_name = "+".join(f.rsplit("_", 1)[0] for f in feats)
res = "/".join(f.rsplit("_", 1)[1] for f in feats)
clf = "RBF SVM (C by inner CV)" if args.clf == "svm" else "logistic (C by inner CV)"
log_experiment({"name": name, "stage": 2 if len(feats) == 1 and args.clf == "logistic" else 3, "model": model_name, "features": "frozen",
                "res": res, "aug": "d4" if args.mode == "aug8" else "none",
                "tta": "none" if args.mode == "t0" else "d4", "classifier": clf,
                "oof_auc": round(auc, 4), "fold_sd": round(sd, 4),
                "minutes": round((time.time() - t0) / 60, 1)})
