"""Stage 3b: ensembles of the saved out-of-fold (OOF) predictions.

Every model in out/oof/ has an OOF score for each training image (from the fold
model that did not see it) and a test score (mean of its 5 fold models). Scores
are turned into ranks in [0, 1] (AUC only depends on order, and ranks put
models with different calibration on one scale), then combined by:

  top-k   mean rank of the k models with the best OOF AUC (k = 2..6)
  greedy  forward selection with replacement (Caruana et al., 2004): start
          empty, repeatedly add the model that most raises the OOF AUC of the
          mean, stop when nothing helps (max 20 steps)
  stack   logistic regression on the models' logit scores (L2, C by inner CV)

Choosing models on the same OOF predictions that are then scored is optimistic,
so each method is also scored by nested CV over the 5 folds of out/folds.csv:
for held-out fold k, the selection/weights are fitted on the other 4 folds'
OOF scores only and applied to fold k; the pooled AUC of those held-out scores
is the honest estimate ("nested"). The method with the best nested AUC is
refitted on all folds and written to submissions/.

Outputs: out/oof/ens_<method>.csv, out/test_preds/ens_<method>.csv,
         submissions/ens_<method>.csv, out/ensemble_weights.csv,
         rows in out/experiments.csv
Base models whose name contains an --exclude pattern are left out; by default
the ViT-B fine-tune at lr 1e-4, whose fold-1 model never trained (train loss
stayed at the class prior), so its test predictions are partly broken.

Run: .venv/bin/python scripts/14_ensemble.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, SEED, load_folds, write_submission  # noqa: E402
from dl import fold_aucs, log_experiment, save_preds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exclude", default="_lr1e-4", help="comma-separated name substrings to skip")
args = ap.parse_args()
excl = [e for e in args.exclude.split(",") if e]

t_start = time.time()
folds = load_folds()
y = folds["label"].values
fold = folds["fold"].values

# ---- load every base model (skip earlier ensembles) -------------------------
names, O, Tt = [], [], []
test_ids = None
for p in sorted((OUT / "oof").glob("*.csv")):
    if p.stem.startswith("ens_") or any(e in p.stem for e in excl):
        continue
    oof = pd.read_csv(p, dtype={"index": str})
    tp = pd.read_csv(OUT / "test_preds" / p.name, dtype={"index": str})
    assert oof["index"].tolist() == folds["index"].tolist()
    assert test_ids is None or tp["index"].tolist() == test_ids
    test_ids = tp["index"].tolist()
    names.append(p.stem)
    O.append(oof["score"].values)
    Tt.append(tp["score"].values)
O, Tt = np.array(O).T, np.array(Tt).T                  # (N, M), (Nte, M)
M = len(names)


def ranks(a):
    """Column-wise rank in (0, 1]."""
    return np.column_stack([rankdata(c) / len(c) for c in a.T])


def logit(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


R, Rte = ranks(O), ranks(Tt)
single = np.array([roc_auc_score(y, O[:, j]) for j in range(M)])

print(f"{M} base models (OOF AUC):")
for j in np.argsort(-single):
    print(f"  {single[j]:.4f}  {names[j]}")
corr = np.corrcoef(R[:, np.argsort(-single)[:6]].T)
print("\nSpearman correlation of the 6 best models' OOF scores:")
print(np.array2string(corr, precision=2, suppress_small=True))


# ---- methods: fit(idx) -> weights over models (for rank-mean) or a model ----
def fit_topk(idx, k):
    a = np.array([roc_auc_score(y[idx], O[idx, j]) for j in range(M)])
    w = np.zeros(M)
    w[np.argsort(-a)[:k]] = 1
    return w / w.sum()


def fit_greedy(idx, steps=20):
    counts = np.zeros(M)
    s = np.zeros(len(idx))
    best = -1
    for _ in range(steps):
        cand = [roc_auc_score(y[idx], (s + R[idx, j]) / (counts.sum() + 1)) for j in range(M)]
        j = int(np.argmax(cand))
        if cand[j] <= best + 1e-5:
            break
        best = cand[j]
        counts[j] += 1
        s += R[idx, j]
    return counts / counts.sum()


def fit_stack(idx):
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    m = LogisticRegressionCV(Cs=np.logspace(-3, 2, 11), cv=inner, scoring="roc_auc",
                             max_iter=5000)
    return m.fit(logit(O[idx]), y[idx])


def apply(method, fitted, idx=None, test=False):
    if method == "stack":
        X = logit(Tt) if test else logit(O[idx])
        return fitted.predict_proba(X)[:, 1]
    return (Rte if test else R[idx]) @ fitted


methods = {f"top{k}": (lambda idx, k=k: fit_topk(idx, k)) for k in range(2, 7)}
methods["greedy"] = fit_greedy
methods["stack"] = fit_stack

# ---- nested CV score and full-data refit for each method --------------------
rows = []
results = {}
for meth, fit in methods.items():
    kind = "stack" if meth == "stack" else "rank"
    nested = np.zeros(len(y))
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        nested[va] = apply(kind, fit(tr), idx=va)
    full = fit(np.arange(len(y)))
    in_sample = apply(kind, full, idx=np.arange(len(y)))
    test = apply(kind, full, test=True)
    n_auc, n_sd, n_per = fold_aucs(y, nested, fold)
    results[meth] = (nested, test, full, n_auc, n_sd)
    rows.append({"method": meth, "nested_auc": round(n_auc, 4), "nested_fold_sd": round(n_sd, 4),
                 "in_sample_auc": round(roc_auc_score(y, in_sample), 4)})

tab = pd.DataFrame(rows).sort_values("nested_auc", ascending=False)
print(f"\nBest single model OOF AUC: {single.max():.4f} ({names[single.argmax()]})")
print("Ensembles (nested = honest; in_sample = selected and scored on the same OOF):")
print(tab.to_string(index=False))

# weights of the rank-based ensembles, refitted on all folds
wrows = []
for meth, (_, _, full, _, _) in results.items():
    if meth == "stack":
        for j, c in enumerate(full.coef_[0]):
            wrows.append({"method": meth, "model": names[j], "weight": round(float(c), 4)})
    else:
        for j in np.nonzero(full)[0]:
            wrows.append({"method": meth, "model": names[j], "weight": round(float(full[j]), 4)})
pd.DataFrame(wrows).to_csv(OUT / "ensemble_weights.csv", index=False)
g = results["greedy"][2]
print("\nGreedy ensemble weights (all folds):")
for j in np.argsort(-g):
    if g[j] > 0:
        print(f"  {g[j]:.3f}  {names[j]}")

# ---- save every ensemble; write a submission for the best nested one --------
best = tab.iloc[0]["method"]
for meth, (nested, test, _, n_auc, n_sd) in results.items():
    name = f"ens_{meth}"
    save_preds(name, nested, test)        # OOF file holds the nested (honest) scores
    log_experiment({"name": name, "stage": 3, "model": f"ensemble of {M} models", "features": "",
                    "res": "", "aug": "", "tta": "", "classifier": f"{meth} (nested CV)",
                    "oof_auc": round(n_auc, 4), "fold_sd": round(n_sd, 4),
                    "minutes": round((time.time() - t_start) / 60, 1)})
nested, test, *_ = results[best]
path = write_submission(pd.Series(test, index=test_ids), f"ens_{best}")
print(f"\nBest by nested AUC: {best} -> {path.relative_to(OUT.parent)}")
