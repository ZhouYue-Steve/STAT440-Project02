"""Make the fixed 5-fold CV split used by every model: out/folds.csv.

01_near_duplicates.py / 01b / 01c found no duplicate, rotated/flipped or
shifted-overlap tiles, so each image is its own group and a plain stratified
5-fold split is valid. The `group` column is kept so that a grouped split can
be swapped in later without changing downstream code.

Columns: index, label, group, fold (0-4).
Run: .venv/bin/python scripts/02_folds.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, SEED, load_train_labels  # noqa: E402

K = 5
df = load_train_labels()
df["group"] = df["index"]
df["fold"] = -1
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
for f, (_, va) in enumerate(skf.split(df, df["label"])):
    df.loc[df.index[va], "fold"] = f
assert (df["fold"] >= 0).all()

out = OUT / "folds.csv"
if out.exists():
    import pandas as pd
    old = pd.read_csv(out, dtype={"index": str})
    if not old[["index", "fold"]].equals(df[["index", "fold"]]):
        raise SystemExit("out/folds.csv exists with a different split; delete it deliberately to regenerate")
df.to_csv(out, index=False)

summary = df.groupby("fold")["label"].agg(n="size", zigzag="sum", share_zigzag="mean")
print(summary.round(3).to_string())
print(f"overall share zigzag: {df['label'].mean():.3f}")
print(f"wrote {out.relative_to(OUT.parent)}")
