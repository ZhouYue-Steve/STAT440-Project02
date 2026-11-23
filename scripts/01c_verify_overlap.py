"""Verify candidate overlaps from 01b at full resolution.

For each candidate pair (peak >= 0.2 in out/overlap_pairs.csv) and for 300
random control pairs: estimate the shift by full-resolution (256x256) phase
correlation, then compute the Pearson correlation of the greyscale pixels in
the overlapping region. A genuine overlap of the same ground gives a
correlation near 1; look-alike scenes do not. Calibrated on synthetic overlaps
cut from training images (two 128x128 tiles from one scene, shifted).

Output: out/overlap_verified.csv (pair, shift, overlap fraction, pixel corr).
Run: .venv/bin/python scripts/01c_verify_overlap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, image_path, list_test_ids, load_train_labels  # noqa: E402

N = 256


def grey(idx, split):
    with Image.open(image_path(idx, split)) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def best_overlap(a, b, min_frac=0.2):
    """Shift of b relative to a by phase correlation; pixel corr on the overlap.

    a and b are square greyscale arrays of the same size."""
    N = a.shape[0]
    win = np.outer(np.hanning(N), np.hanning(N))
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r = np.fft.ifft2(r / (np.abs(r) + 1e-6)).real
    dy, dx = np.unravel_index(r.argmax(), r.shape)
    dy = dy - N if dy > N // 2 else dy
    dx = dx - N if dx > N // 2 else dx
    # overlap: a[y, x] ~ b[y - dy, x - dx]
    ya, yb = (slice(dy, N), slice(0, N - dy)) if dy >= 0 else (slice(0, N + dy), slice(-dy, N))
    xa, xb = (slice(dx, N), slice(0, N - dx)) if dx >= 0 else (slice(0, N + dx), slice(-dx, N))
    pa, pb = a[ya, xa].ravel(), b[yb, xb].ravel()
    frac = pa.size / (N * N)
    if frac < min_frac:
        return dy, dx, frac, np.nan
    return dy, dx, frac, float(np.corrcoef(pa, pb)[0, 1])


train = load_train_labels()
split_of = {i: "train" for i in train["index"]} | {i: "test" for i in list_test_ids()}
lab = dict(zip(train["index"], train["label"]))
rng = np.random.default_rng(0)

# --- calibration: two 128x128 tiles cut from the same 256x256 scene, the second
# shifted by `off` pixels (diagonally), i.e. genuine overlaps of known size ---
calib = []
T = 128
for idx in train["index"].sample(40, random_state=1):
    g = grey(idx, "train")
    for off in (16, 32, 48, 64):
        a = g[:T, :T]
        b = g[off:off + T, off // 2:off // 2 + T]
        calib.append((off, *best_overlap(a, b)))
calib = pd.DataFrame(calib, columns=["offset", "dy", "dx", "frac", "corr"])
calib["shift_ok"] = (calib["dy"] == calib["offset"]) & (calib["dx"] == calib["offset"] // 2)
print("Calibration (true overlaps, 128px tiles): share of shifts recovered exactly, pixel corr")
print(calib.groupby("offset").agg(overlap=("frac", "mean"), shift_ok=("shift_ok", "mean"),
                                  corr_min=("corr", "min"), corr_median=("corr", "median")).round(3).to_string())

# --- candidates and random controls ---
cand = pd.read_csv(OUT / "overlap_pairs.csv", dtype={"a": str, "b": str})
cand = cand[cand["peak"] >= 0.2][["a", "b", "peak"]].assign(kind="candidate")
all_ids = list(split_of)
ctrl = pd.DataFrame([rng.choice(all_ids, 2, replace=False) for _ in range(300)], columns=["a", "b"])
ctrl = ctrl.assign(peak=np.nan, kind="random")
pairs = pd.concat([cand, ctrl], ignore_index=True)

res = [best_overlap(grey(a, split_of[a]), grey(b, split_of[b])) for a, b in zip(pairs["a"], pairs["b"])]
pairs[["dy", "dx", "frac", "corr"]] = pd.DataFrame(res, index=pairs.index)
pairs["split_a"] = pairs["a"].map(split_of)
pairs["split_b"] = pairs["b"].map(split_of)
pairs.to_csv(OUT / "overlap_verified.csv", index=False)

print("\nPixel corr on overlap (NaN = overlap < 20% of the tile):")
for kind, g in pairs.groupby("kind"):
    c = g["corr"].dropna()
    print(f"  {kind:9s}: n={len(g)}, with overlap>=20%: {len(c)}, "
          f"median={c.median():.3f}, 99th pct={c.quantile(0.99):.3f}, max={c.max():.3f}")
for T in (0.6, 0.8, 0.9):
    hit = pairs[(pairs["kind"] == "candidate") & (pairs["corr"] >= T)]
    kinds = (hit["split_a"] + "-" + hit["split_b"]).replace({"test-train": "train-test"})
    tt = hit[kinds == "train-train"]
    agree = (tt["a"].map(lab) == tt["b"].map(lab)).mean() if len(tt) else float("nan")
    print(f"  candidates with corr >= {T}: {len(hit)} "
          f"(train-train {int((kinds == 'train-train').sum())}, train-test {int((kinds == 'train-test').sum())}, "
          f"test-test {int((kinds == 'test-test').sum())}; train-train label agreement {agree:.2f})")
