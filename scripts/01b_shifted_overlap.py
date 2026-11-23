"""Check for partially overlapping (translated) tiles, which global hashes miss.

For every pair of the 2,341 images, phase correlation of Hann-windowed 64x64
greyscale versions: a translated copy gives a sharp peak at the shift, unrelated
images give a low, flat surface. Orientation is assumed unchanged (tiles cut
from the same scene); rotated/flipped copies are covered by 01_near_duplicates.py.

Outputs (only aggregates are printed):
  out/overlap_pairs.csv   pairs with peak >= 0.15, with the estimated shift
  out/overlap_nn.csv      each image's highest peak over all other images
Run: .venv/bin/python scripts/01b_shifted_overlap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, image_path, list_test_ids, load_train_labels  # noqa: E402

S = 64
train = load_train_labels()
ids = train["index"].tolist() + list_test_ids()
split = np.array(["train"] * len(train) + ["test"] * (len(ids) - len(train)))
n = len(ids)

win = np.outer(np.hanning(S), np.hanning(S)).astype(np.float32)
F = np.zeros((n, S, S), dtype=np.complex64)
for i, (idx, s) in enumerate(zip(ids, split)):
    with Image.open(image_path(idx, s)) as im:
        g = np.asarray(im.convert("L").resize((S, S), Image.BILINEAR), dtype=np.float32)
    g = (g - g.mean()) * win
    f = np.fft.fft2(g)
    F[i] = f / (np.abs(f) + 1e-6)          # whitened spectrum

peak = np.zeros((n, n), dtype=np.float32)
shift = np.zeros((n, n, 2), dtype=np.int8)
B = 64
for i0 in range(0, n, B):
    Fi = F[i0:i0 + B]                                      # (b, S, S)
    for j0 in range(i0, n, B):
        Fj = F[j0:j0 + B]
        r = np.fft.ifft2(Fi[:, None] * np.conj(Fj[None]), axes=(-2, -1)).real  # (b, b', S, S)
        flat = r.reshape(r.shape[0], r.shape[1], -1)
        am = flat.argmax(-1)
        pk = np.take_along_axis(flat, am[..., None], -1)[..., 0]
        peak[i0:i0 + B, j0:j0 + B] = pk
        dy, dx = np.unravel_index(am, (S, S))
        shift[i0:i0 + B, j0:j0 + B, 0] = np.where(dy > S // 2, dy - S, dy)
        shift[i0:i0 + B, j0:j0 + B, 1] = np.where(dx > S // 2, dx - S, dx)
iu, ju = np.triu_indices(n, 1)
peak_u = peak[iu, ju]
full = np.zeros((n, n), dtype=np.float32)
full[iu, ju] = peak_u
full = full + full.T

nn = full.max(1)
pd.DataFrame({"index": ids, "split": split, "max_peak": nn}).to_csv(OUT / "overlap_nn.csv", index=False)
print("Phase-correlation peak (1 = identical up to shift; unrelated images are low)")
print("Each image's highest peak over all other images:")
for q in [0.5, 0.9, 0.99, 0.999, 1.0]:
    print(f"  {q:>5} quantile: {np.quantile(nn, q):.3f}")
print("All pairs:")
for q in [0.5, 0.999, 0.99999, 1.0]:
    print(f"  {q:>7} quantile: {np.quantile(peak_u, q):.3f}")

keep = peak_u >= 0.15
pairs = pd.DataFrame({
    "a": np.array(ids)[iu[keep]], "b": np.array(ids)[ju[keep]],
    "split_a": split[iu[keep]], "split_b": split[ju[keep]],
    "peak": peak_u[keep], "dy": shift[iu[keep], ju[keep], 0] * (256 // S),
    "dx": shift[iu[keep], ju[keep], 1] * (256 // S),
})
pairs.to_csv(OUT / "overlap_pairs.csv", index=False)

lab = dict(zip(train["index"], train["label"]))
print("\nPairs with peak >= T:")
print(f"  {'T':>5s} {'train-train':>12s} {'train-test':>11s} {'test-test':>10s} {'labels agree':>13s}")
for T in [0.15, 0.2, 0.3, 0.5]:
    sub = pairs[pairs["peak"] >= T]
    kinds = (sub["split_a"] + "-" + sub["split_b"]).replace({"test-train": "train-test"})
    tt = sub[kinds == "train-train"]
    agree = (tt["a"].map(lab) == tt["b"].map(lab)).mean() if len(tt) else float("nan")
    print(f"  {T:>5.2f} {(kinds == 'train-train').sum():>12d} {(kinds == 'train-test').sum():>11d} "
          f"{(kinds == 'test-test').sum():>10d} {agree:>13.2f}")
