"""Deep-learning helpers shared by the model scripts (PyTorch, Apple MPS)."""
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from common import OUT, list_test_ids, load_folds, load_image

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

# The 8 symmetries of the square (dihedral group D4): 4 rotations x optional flip.
# A satellite photo under any of them is still a valid photo of the same kiln.
D4 = [(k, flip) for flip in (False, True) for k in range(4)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def d4(x: torch.Tensor, t: int) -> torch.Tensor:
    """Apply transform t (0-7) of D4 to a batch (..., H, W) tensor."""
    k, flip = D4[t]
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


def load_split(split: str) -> tuple[list[str], np.ndarray]:
    """All images of a split as one uint8 array (N, 256, 256, 3), with ids.

    Train order follows out/folds.csv; test order is ascending id.
    """
    ids = load_folds()["index"].tolist() if split == "train" else list_test_ids()
    return ids, np.stack([load_image(i, split) for i in ids])


def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 in [0, 1]."""
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float().div_(255)


def fold_aucs(y: np.ndarray, p: np.ndarray, fold: np.ndarray) -> tuple[float, float, list[float]]:
    """Pooled OOF AUC, SD of per-fold AUC, and the per-fold AUCs."""
    per = [roc_auc_score(y[fold == k], p[fold == k]) for k in sorted(np.unique(fold))]
    return roc_auc_score(y, p), float(np.std(per)), per


def save_preds(name: str, oof: np.ndarray, test: np.ndarray | None) -> None:
    """out/oof/<name>.csv (train, in folds.csv order) and out/test_preds/<name>.csv."""
    folds = load_folds()
    (OUT / "oof").mkdir(exist_ok=True)
    pd.DataFrame({"index": folds["index"], "fold": folds["fold"], "label": folds["label"],
                  "score": oof}).to_csv(OUT / "oof" / f"{name}.csv", index=False)
    if test is not None:
        (OUT / "test_preds").mkdir(exist_ok=True)
        pd.DataFrame({"index": list_test_ids(), "score": test}).to_csv(
            OUT / "test_preds" / f"{name}.csv", index=False)


def log_experiment(row: dict) -> None:
    """Append one result row to out/experiments.csv."""
    path = OUT / "experiments.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
