"""Find exact and near-duplicate images among all 2,341 train + test photos.

Satellite tiles of the same kiln may appear more than once (re-captures,
rotations, flips). Near-duplicates inside the training set would make random
CV folds optimistic, so they must share a fold (GroupKFold in 02_folds.py).

Two independent measures, each the minimum over the 8 rotations/flips (D4):
  * pHash Hamming distance (64-bit perceptual hash; imagehash library)
  * Pearson correlation of 32x32 greyscale thumbnails

Outputs (only aggregate counts are printed):
  out/near_dup_pairs.csv   every pair with pHash distance <= 16 or corr >= 0.90
  out/nn_distance.csv      each image's nearest-neighbour distance/correlation
Run: .venv/bin/python scripts/01_near_duplicates.py
"""
import hashlib
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, image_path, list_test_ids, load_train_labels  # noqa: E402

D4 = [(k, flip) for flip in (False, True) for k in range(4)]  # 8 transforms


def d4(arr, k, flip):
    a = arr[:, ::-1] if flip else arr
    return np.rot90(a, k)


train = load_train_labels()
ids = train["index"].tolist() + list_test_ids()
split = np.array(["train"] * len(train) + ["test"] * (len(ids) - len(train)))
n = len(ids)

md5 = []
phash = np.zeros((n, 8), dtype=np.uint64)   # phash[i, t] = hash of transform t of image i
thumb = np.zeros((n, 8, 32 * 32), dtype=np.float32)
for i, (idx, s) in enumerate(zip(ids, split)):
    p = image_path(idx, s)
    md5.append(hashlib.md5(p.read_bytes()).hexdigest())
    with Image.open(p) as im:
        rgb = np.asarray(im.convert("RGB"))
    for t, (k, flip) in enumerate(D4):
        a = np.ascontiguousarray(d4(rgb, k, flip))
        img = Image.fromarray(a)
        bits = imagehash.phash(img).hash.flatten()
        phash[i, t] = np.packbits(bits).view(">u8")[0]
        g = np.asarray(img.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
        g = g - g.mean()
        thumb[i, t] = g / (np.linalg.norm(g) + 1e-8)

# min over transforms of image j (image i kept as is)
dist = np.full((n, n), 64, dtype=np.uint8)
corr = np.full((n, n), -1.0, dtype=np.float32)
for t in range(8):
    x = np.bitwise_xor(phash[:, 0][:, None], phash[:, t][None, :])
    dist = np.minimum(dist, np.bitwise_count(x).astype(np.uint8))
    corr = np.maximum(corr, thumb[:, 0] @ thumb[:, t].T)
dist = np.minimum(dist, dist.T)        # symmetrise
corr = np.maximum(corr, corr.T)
np.fill_diagonal(dist, 255)
np.fill_diagonal(corr, -1)

# ---------------- exact duplicates ----------------
md5 = pd.Series(md5)
n_exact = int(md5.duplicated(keep=False).sum())
print(f"Images: {n} ({(split == 'train').sum()} train, {(split == 'test').sum()} test)")
print(f"Images whose file bytes exactly match another image: {n_exact}")

# ---------------- nearest-neighbour distributions ----------------
nn_d = dist.min(1)
nn_c = corr.max(1)
pd.DataFrame({"index": ids, "split": split, "nn_phash": nn_d, "nn_corr": nn_c}).to_csv(
    OUT / "nn_distance.csv", index=False)

print("\nNearest-neighbour pHash distance (0 = identical, ~32 = unrelated):")
bins = [0, 1, 3, 5, 7, 9, 11, 13, 17, 21, 65]
h = pd.cut(pd.Series(nn_d), bins=bins, right=False).value_counts().sort_index()
for interval, c in h.items():
    print(f"  {str(interval):>10s}: {c:5d}")
print("\nNearest-neighbour thumbnail correlation:")
for q in [0.5, 0.9, 0.99, 0.999]:
    print(f"  {q:>5} quantile: {np.quantile(nn_c, q):.3f}")
for c in [0.90, 0.95, 0.98]:
    print(f"  images with a neighbour at corr >= {c}: {(nn_c >= c).sum()}")

# ---------------- candidate pairs ----------------
iu, ju = np.triu_indices(n, 1)
keep = (dist[iu, ju] <= 16) | (corr[iu, ju] >= 0.90)
pairs = pd.DataFrame({
    "a": np.array(ids)[iu[keep]], "b": np.array(ids)[ju[keep]],
    "split_a": split[iu[keep]], "split_b": split[ju[keep]],
    "phash": dist[iu[keep], ju[keep]], "corr": corr[iu[keep], ju[keep]],
})
pairs.to_csv(OUT / "near_dup_pairs.csv", index=False)

lab = dict(zip(train["index"], train["label"]))
print("\nPairs by threshold (pHash distance <= T):")
print(f"  {'T':>3s} {'train-train':>12s} {'train-test':>11s} {'test-test':>10s} "
      f"{'labels agree (train-train)':>27s} {'median corr':>12s}")
for T in [0, 4, 6, 8, 10, 12, 16]:
    sub = pairs[pairs["phash"] <= T]
    kinds = (sub["split_a"] + "-" + sub["split_b"]).replace({"test-train": "train-test"})
    tt = sub[kinds == "train-train"]
    agree = (tt["a"].map(lab) == tt["b"].map(lab)).mean() if len(tt) else float("nan")
    print(f"  {T:>3d} {(kinds == 'train-train').sum():>12d} {(kinds == 'train-test').sum():>11d} "
          f"{(kinds == 'test-test').sum():>10d} {agree:>27.2f} "
          f"{sub['corr'].median() if len(sub) else float('nan'):>12.3f}")

# Base rate for comparison: label agreement of random training pairs
p1 = train["label"].mean()
print(f"\nLabel agreement expected for random training pairs: {p1**2 + (1-p1)**2:.2f}")
