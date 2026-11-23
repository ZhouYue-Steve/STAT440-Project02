"""Shortcut and distribution-shift checks with simple, hand-made image features.

1. Single-feature AUC on the training labels (|AUC - 0.5| is what matters).
   Features that describe the *imagery* rather than the ground (file size,
   sharpness, blank/saturated pixels, image index) would be shortcuts: they
   could differ by class only because the two classes came from different
   sources or dates.
2. 5-fold CV AUC (out/folds.csv) of logistic regression and gradient boosting
   on all simple features: a floor that any CNN should beat clearly.
3. Adversarial validation: can the same features tell train from test?
   AUC near 0.5 means the test images look like the training images.

Output: out/simple_features.csv (one row per image), printed aggregates only.
Run: .venv/bin/python scripts/03_shortcut_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, SEED, image_path, list_test_ids, load_folds  # noqa: E402

IMAGERY = ["file_kb", "sharpness", "frac_dark", "frac_bright", "frac_flat_rows", "index_num"]


def features(idx, split):
    p = image_path(idx, split)
    with Image.open(p) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255
    grey = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    f = {
        "index": idx, "split": split,
        "index_num": int(idx[1:]),
        "file_kb": p.stat().st_size / 1024,
        "sharpness": ndimage.laplace(grey).var(),
        "frac_dark": (grey < 0.05).mean(),
        "frac_bright": (grey > 0.95).mean(),
        "frac_flat_rows": (grey.std(1) < 0.01).mean(),
        "edge_density": (np.hypot(ndimage.sobel(grey, 0), ndimage.sobel(grey, 1)) > 0.5).mean(),
        # reddish/orange bare soil and brick: red clearly above green and blue
        "frac_brick": ((r > g + 0.06) & (r > b + 0.10)).mean(),
        # vegetation: green above red
        "frac_green": (g > r + 0.02).mean(),
    }
    for c, name in enumerate("rgb"):
        ch = rgb[..., c]
        f[f"{name}_mean"] = ch.mean()
        f[f"{name}_std"] = ch.std()
        f[f"{name}_p10"], f[f"{name}_p90"] = np.percentile(ch, [10, 90])
    f["sat_mean"] = hsv[..., 1].mean()
    f["val_mean"] = hsv[..., 2].mean()
    f["grey_std"] = grey.std()
    # centre vs border: kilns are usually near the centre of the tile
    c = grey[64:192, 64:192]
    f["centre_minus_all_mean"] = c.mean() - grey.mean()
    f["centre_brick"] = ((r[64:192, 64:192] > g[64:192, 64:192] + 0.06)
                         & (r[64:192, 64:192] > b[64:192, 64:192] + 0.10)).mean()
    return f


folds = load_folds()
rows = [features(i, "train") for i in folds["index"]] + [features(i, "test") for i in list_test_ids()]
F = pd.DataFrame(rows)
F.to_csv(OUT / "simple_features.csv", index=False)
feat_cols = [c for c in F.columns if c not in ("index", "split")]

tr = F[F["split"] == "train"].reset_index(drop=True)
assert tr["index"].tolist() == folds["index"].tolist()
y = folds["label"].values

# ---- 1. single-feature AUCs ----
auc1 = pd.Series({c: roc_auc_score(y, tr[c]) for c in feat_cols})
tab = pd.DataFrame({"AUC": auc1, "strength": (auc1 - 0.5).abs(),
                    "type": ["imagery" if c in IMAGERY else "content" for c in feat_cols]})
tab = tab.sort_values("strength", ascending=False)
print("1. Single-feature AUC for zigzag (0.5 = no signal; <0.5 = higher value means fck)")
print(tab[["AUC", "type"]].round(3).to_string())

# ---- 2. CV AUC of simple models, using the fixed folds ----
class Folds:  # sklearn-compatible splitter that reuses out/folds.csv
    def split(self, X, y=None, groups=None):
        for k in sorted(folds["fold"].unique()):
            yield np.where(folds["fold"] != k)[0], np.where(folds["fold"] == k)[0]
    def get_n_splits(self, *a, **k):
        return folds["fold"].nunique()


def cv_auc(model, cols):
    p = cross_val_predict(model, tr[cols].values, y, cv=Folds(), method="predict_proba")[:, 1]
    per_fold = [roc_auc_score(y[folds["fold"] == k], p[folds["fold"] == k]) for k in range(5)]
    return roc_auc_score(y, p), np.std(per_fold)


logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)
content = [c for c in feat_cols if c not in IMAGERY]
print("\n2. 5-fold CV AUC on simple features (pooled OOF AUC; SD across folds)")
for name, model in [("logistic", logit), ("gradient boosting", gbm)]:
    for label, cols in [("all features", feat_cols), ("content only", content), ("imagery only", IMAGERY)]:
        a, s = cv_auc(model, cols)
        print(f"  {name:18s} {label:14s}: AUC {a:.3f} (fold SD {s:.3f})")

# ---- 3. adversarial validation: train vs test ----
y_adv = (F["split"] == "test").astype(int).values
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
print("\n3. Adversarial validation (predict test vs train; 0.5 = same distribution)")
for name, model in [("logistic", logit), ("gradient boosting", gbm)]:
    for label, cols in [("all features", feat_cols), ("all but index", [c for c in feat_cols if c != "index_num"])]:
        p = cross_val_predict(model, F[cols].values, y_adv, cv=skf, method="predict_proba")[:, 1]
        print(f"  {name:18s} {label:14s}: AUC {roc_auc_score(y_adv, p):.3f}")
# label drift along the index within the training set
q = pd.qcut(folds["index"].str[1:].astype(int), 5, labels=False)
print("\nShare zigzag by training-index quintile:", folds.groupby(q)["label"].mean().round(3).tolist())
