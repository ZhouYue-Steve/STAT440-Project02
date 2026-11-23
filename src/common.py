"""Shared paths, data loading and submission helpers for Project 2 (Kilns).

All scripts import from here so that paths, ids and the submission format
are defined in exactly one place.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
OUT = ROOT / "out"
SUB_DIR = ROOT / "submissions"

N_TRAIN = 1617
N_TEST = 724
IMG_SIZE = 256
SEED = 440


def load_train_labels() -> pd.DataFrame:
    """train/data.csv as a DataFrame with columns index (str), label (int)."""
    df = pd.read_csv(TRAIN_DIR / "data.csv", dtype={"index": str, "label": int})
    assert len(df) == N_TRAIN, len(df)
    assert df["index"].is_unique
    assert set(df["label"]) == {0, 1}
    return df


def list_test_ids() -> list[str]:
    """Test ids from the filenames in data/test, sorted ascending."""
    ids = sorted(p.stem for p in TEST_DIR.glob("*.png"))
    assert len(ids) == N_TEST, len(ids)
    return ids


def image_path(idx: str, split: str) -> Path:
    return (TRAIN_DIR if split == "train" else TEST_DIR) / f"{idx}.png"


def load_image(idx: str, split: str) -> np.ndarray:
    """uint8 array of shape (256, 256, 3)."""
    with Image.open(image_path(idx, split)) as im:
        arr = np.asarray(im.convert("RGB"))
    assert arr.shape == (IMG_SIZE, IMG_SIZE, 3), arr.shape
    return arr


def load_folds() -> pd.DataFrame:
    """out/folds.csv: index, label, group, fold (made by scripts/02_folds.py)."""
    return pd.read_csv(OUT / "folds.csv", dtype={"index": str})


# --------------------------------------------------------------------------
# Submission format (see Kilns.pdf): header exactly "index,score", one row per
# test image, sorted ascending by index, score a real number in [0, 1].
# --------------------------------------------------------------------------

def validate_submission(path) -> None:
    """Raise AssertionError if the file at `path` breaks the submission format."""
    path = Path(path)
    with open(path) as f:
        header = f.readline().rstrip("\r\n")
    assert header == "index,score", f"bad header: {header!r}"

    df = pd.read_csv(path, dtype={"index": str})
    test_ids = list_test_ids()
    assert len(df) == N_TEST, f"expected {N_TEST} rows, got {len(df)}"
    assert df["index"].tolist() == test_ids, "ids missing, extra, or not sorted ascending"
    s = df["score"]
    assert s.notna().all(), "missing scores"
    assert np.isfinite(s).all(), "non-finite scores"
    assert ((s >= 0) & (s <= 1)).all(), "scores outside [0, 1]"


def write_submission(scores: dict | pd.Series, name: str) -> Path:
    """Write submissions/<name>.csv from {test id: score}, then validate it."""
    scores = pd.Series(scores, dtype=float)
    test_ids = list_test_ids()
    missing = set(test_ids) - set(scores.index)
    extra = set(scores.index) - set(test_ids)
    assert not missing and not extra, f"missing {len(missing)}, extra {len(extra)} ids"
    df = pd.DataFrame({"index": test_ids, "score": scores.loc[test_ids].clip(0, 1).values})
    SUB_DIR.mkdir(exist_ok=True)
    path = SUB_DIR / f"{name}.csv"
    df.to_csv(path, index=False, float_format="%.6f")
    validate_submission(path)
    return path
