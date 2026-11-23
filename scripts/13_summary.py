"""Print out/experiments.csv sorted by OOF AUC (latest run of each name).

Run: .venv/bin/python scripts/13_summary.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT  # noqa: E402

df = pd.read_csv(OUT / "experiments.csv").drop_duplicates("name", keep="last")
df["score_pct"] = (200 * (df["oof_auc"] - 0.5)).round(1)     # contest score scale
cols = ["name", "res", "aug", "tta", "oof_auc", "fold_sd", "score_pct", "minutes"]
with pd.option_context("display.width", 200, "display.max_colwidth", 60):
    print(df.sort_values("oof_auc", ascending=False)[cols].to_string(index=False))
