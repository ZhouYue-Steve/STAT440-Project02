"""Stage 3c: fine-tune a pretrained ViT (DINOv2) end to end, 5-fold CV.

The frozen-feature probes (12_) showed DINOv2 features are by far the best;
here the whole network is trained on the kiln labels.

Recipe (standard for fine-tuning ViTs on small datasets):
  - input: the 256x256 image resized to --res (bilinear, antialiased, no crop),
    normalised with the model's mean/std
  - augmentation: a random one of the 8 rotations/flips (D4) per image, plus mild
    brightness/contrast jitter (+-10%); all on the GPU
  - AdamW, weight decay 0.05, layer-wise LR decay (--llrd: block i of L gets
    lr * llrd^(L - i), the patch embedding the smallest, the head the full lr),
    1 warm-up epoch then cosine decay, stochastic depth 0.1, bf16 autocast
  - the model after the last epoch is used (no early stopping on the validation
    fold, so the OOF AUC is not tuned on itself); predictions average the 8 D4
    transforms (TTA). Test predictions average the 5 fold models.

Outputs: out/<name>_curve.csv (val AUC per fold/epoch, no TTA),
         out/oof/<name>.csv, out/test_preds/<name>.csv, row in out/experiments.csv
Run: .venv/bin/python scripts/15_finetune.py --model vit_small_patch14_dinov2.lvd142m --res 336
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import OUT, SEED, load_folds  # noqa: E402
from dl import DEVICE, d4, fold_aucs, load_split, log_experiment, save_preds, set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
ap.add_argument("--res", type=int, default=336)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4, help="head LR; blocks get lr * llrd^depth")
ap.add_argument("--llrd", type=float, default=0.75)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--folds", type=str, default="0,1,2,3,4")
ap.add_argument("--tag", type=str, default="")
args = ap.parse_args()
folds_to_run = [int(f) for f in args.folds.split(",")]
name = f"ft_{args.model}_{args.res}" + (f"_{args.tag}" if args.tag else "")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def make_model():
    m = timm.create_model(args.model, pretrained=True, num_classes=1, img_size=args.res,
                          drop_path_rate=0.1)
    return m.to(DEVICE)


def param_groups(model):
    """Layer-wise LR decay for timm ViTs: depth 0 = embeddings, 1..L = blocks, L+1 = head."""
    L = len(model.blocks)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("blocks."):
            depth = int(n.split(".")[1]) + 1
        elif n.startswith(("cls_token", "reg_token", "pos_embed", "patch_embed", "mask_token")):
            depth = 0
        else:                                       # final norm, head
            depth = L + 1
        no_wd = p.ndim == 1 or n.endswith(".bias") or "token" in n or "pos_embed" in n
        key = (depth, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr_scale": args.llrd ** (L + 1 - depth),
                           "weight_decay": 0.0 if no_wd else args.wd}
        groups[key]["params"].append(p)
    return list(groups.values())


folds = load_folds()
y = folds["label"].values
fold = folds["fold"].values
_, Xtr = load_split("train")                         # uint8 (N, 256, 256, 3), kept on CPU
_, Xte = load_split("test")
Xtr = torch.from_numpy(Xtr).permute(0, 3, 1, 2).contiguous()
Xte = torch.from_numpy(Xte).permute(0, 3, 1, 2).contiguous()

_probe = timm.create_model(args.model, pretrained=False, num_classes=1, img_size=args.res)
cfg = timm.data.resolve_data_config({}, model=_probe)
del _probe
MEAN = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)


def prep(xb_u8, train):
    """uint8 (b, 3, 256, 256) CPU -> normalised float (b, 3, res, res) on the GPU."""
    x = xb_u8.to(DEVICE, non_blocking=True).float().div_(255)
    if train:
        b = len(x)
        t = torch.randint(0, 8, (b,))
        for k in t.unique().tolist():               # random D4 element per image
            m = (t == k).to(DEVICE)
            x[m] = d4(x[m], k)
        bright = 1 + 0.1 * (2 * torch.rand(b, 1, 1, 1, device=DEVICE) - 1)
        contrast = 1 + 0.1 * (2 * torch.rand(b, 1, 1, 1, device=DEVICE) - 1)
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mu) * contrast + mu) * bright
        x = x.clamp_(0, 1)
    if args.res != x.shape[-1]:
        x = F.interpolate(x, size=(args.res, args.res), mode="bilinear", antialias=True,
                          align_corners=False)
    return (x - MEAN) / STD


@torch.no_grad()
def predict(model, X, tta=True, bs=32):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs], train=False)
        ts = range(8) if tta else [0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.stack([torch.sigmoid(model(d4(x, t)).float().squeeze(-1)) for t in ts])
        out.append(p.mean(0).cpu())
    return torch.cat(out).numpy()


oof = np.full(len(y), np.nan)
test_pred = np.zeros(len(Xte))
curve = []
t0 = time.time()
for k in folds_to_run:
    set_seed(SEED + k)
    tr_idx = np.where(fold != k)[0]
    va_idx = np.where(fold == k)[0]
    model = make_model()
    groups = param_groups(model)
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(tr_idx) / args.bs)
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch
    lossf = torch.nn.BCEWithLogitsLoss()
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = tr_idx[np.random.permutation(len(tr_idx))]
        losses = []
        for b in range(0, len(perm), args.bs):
            bi = perm[b:b + args.bs]
            if len(bi) < 2:
                continue
            f = (step + 1) / warm if step < warm else \
                0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = args.lr * g["lr_scale"] * f
            x = prep(Xtr[bi], train=True)
            yb = torch.from_numpy(y[bi]).float().to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x).float().squeeze(-1)
            loss = lossf(logit, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        p = predict(model, Xtr[va_idx], tta=False)
        auc = roc_auc_score(y[va_idx], p)
        curve.append({"fold": k, "epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_auc": auc})
        print(f"fold {k} epoch {ep + 1:2d}: loss {np.mean(losses):.3f}  val AUC {auc:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    oof[va_idx] = predict(model, Xtr[va_idx], tta=True)
    test_pred += predict(model, Xte, tta=True) / len(folds_to_run)
    print(f"fold {k}: final val AUC with D4 TTA {roc_auc_score(y[va_idx], oof[va_idx]):.3f}", flush=True)
    del model, opt
    torch.cuda.empty_cache()

pd.DataFrame(curve).to_csv(OUT / f"{name}_curve.csv", index=False)
if len(folds_to_run) == 5:
    auc, sd, per = fold_aucs(y, oof, fold)
    print(f"\n{name}: pooled OOF AUC {auc:.4f}; per-fold {np.round(per, 3).tolist()} (SD {sd:.3f}); "
          f"{(time.time() - t0) / 60:.1f} min")
    save_preds(name, oof, test_pred)
    log_experiment({"name": name, "stage": 3, "model": args.model, "features": "fine-tuned",
                    "res": args.res, "aug": "d4+jitter", "tta": "d4",
                    "classifier": f"end-to-end (AdamW lr {args.lr:g}, LLRD {args.llrd}, {args.epochs} ep)",
                    "oof_auc": round(auc, 4), "fold_sd": round(sd, 4),
                    "minutes": round((time.time() - t0) / 60, 1)})
