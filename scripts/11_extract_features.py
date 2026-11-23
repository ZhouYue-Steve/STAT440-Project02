"""Extract frozen features from ImageNet-pretrained networks (timm).

For every train and test image and each of the 8 rotations/flips (D4), the
pooled penultimate-layer features of a pretrained backbone, with no training.
Pretrained networks are allowed by the contest rules; weights come from the
timm / Hugging Face hub (open licences, see each model card).

Images are resized from 256 to --res (bilinear, antialiased; no cropping, so a
kiln near the edge is kept) and normalised with the model's own mean/std.

Output: out/features/<model>_<res>.npz with
  train (1617, 8, D) float16 in out/folds.csv order, test (724, 8, D) float16,
  train_ids, test_ids.
Run: .venv/bin/python scripts/11_extract_features.py --model resnet50.a1_in1k --res 224
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT  # noqa: E402
from dl import DEVICE, d4, load_split, to_tensor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--bs", type=int, default=32)
args = ap.parse_args()

# ViTs: fix the input size at creation so the position embeddings are resampled
# once on CPU (MPS cannot do the antialiased bicubic resize used at run time)
kw = {"img_size": args.res} if "vit" in args.model else {}
model = timm.create_model(args.model, pretrained=True, num_classes=0, **kw).to(DEVICE).eval()
cfg = timm.data.resolve_data_config({}, model=model)
mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def extract(X_u8):
    feats = []
    for i in range(0, len(X_u8), args.bs):
        x = to_tensor(X_u8[i:i + args.bs])
        if args.res != x.shape[-1]:      # on CPU: MPS has no antialiased resize
            x = F.interpolate(x, size=(args.res, args.res), mode="bilinear", antialias=True,
                              align_corners=False)
        x = (x.to(DEVICE) - mean) / std
        per_t = [model(d4(x, t)).float().cpu() for t in range(8)]
        feats.append(torch.stack(per_t, 1).half())       # (b, 8, D)
    return torch.cat(feats).numpy()


t0 = time.time()
train_ids, Xtr = load_split("train")
test_ids, Xte = load_split("test")
ftr = extract(Xtr)
fte = extract(Xte)
(OUT / "features").mkdir(exist_ok=True)
safe = args.model.replace("/", "_")
path = OUT / "features" / f"{safe}_{args.res}.npz"
np.savez(path, train=ftr, test=fte, train_ids=np.array(train_ids), test_ids=np.array(test_ids))
print(f"{args.model} @ {args.res}: features {ftr.shape[-1]}-d, "
      f"{time.time() - t0:.0f}s -> {path.relative_to(OUT.parent)}")
