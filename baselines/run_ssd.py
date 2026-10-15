"""
ssd_baseline.py — SSD (Selective Synaptic Dampening), Foster et al., AAAI 2024.
Run as a SOTA comparison baseline against BEAIS on CIFAR-10 and SVHN.

How SSD works (no retraining at all):
  1. Estimate the diagonal Fisher Information for every weight, twice:
     once over the FULL training set D, once over the forget set D_f.
     (Fisher is approximated by the mean squared gradient.)
  2. A weight is "forget-specialised" if its Fisher on D_f is much larger
     than on D, i.e.  FIM_f[i] > alpha * FIM_D[i].
  3. Those weights are DAMPENED (scaled down), not retrained:
         beta_i = min( lambda * FIM_D[i] / FIM_f[i],  1 )
         w_i   <- beta_i * w_i
     Weights important to the forget class shrink; everything else is
     untouched. Because there is no fine-tuning, SSD runs in seconds.

Uses the SAME AllCNN model and cached base checkpoints as the BEAIS
experiments, so the numbers drop straight into Tables 1 and 2.

Kaggle/Colab: paste and run. Set cfg.dataset to "cifar10" or "svhn".
"""

import copy
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. Model — identical AllCNN to the BEAIS experiments
# ============================================================================

class AllCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()

        def block(c_in, c_out, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, k, stride=s, padding=p),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            nn.Dropout(0.2),
            block(in_channels, 96), block(96, 96), block(96, 96, s=2),
            nn.Dropout(0.5),
            block(96, 192), block(192, 192), block(192, 192, s=2),
            nn.Dropout(0.5),
            block(192, 192, k=3, p=0),
            block(192, 192, k=1, p=0),
        )
        self.classifier = nn.Conv2d(192, num_classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

    def forward(self, x):
        return self.pool(self.classifier(self.features(x))).flatten(1)


# ============================================================================
# 2. Data — identical splits to the BEAIS experiments
# ============================================================================

def get_dataset(name, data_dir="./data"):
    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        tr = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
        te = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
        return tr, te, np.array(tr.targets), np.array(te.targets)

    if name == "svhn":
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        tr = datasets.SVHN(data_dir, split="train", download=True, transform=tfm)
        te = datasets.SVHN(data_dir, split="test", download=True, transform=tfm)
        labels = np.array(tr.labels)
        # same 40,000-image stratified subsample as the BEAIS runs
        rng = np.random.RandomState(42)
        keep = []
        for c in range(10):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            keep.extend(idx[:4000].tolist())
        rng.shuffle(keep)
        return Subset(tr, keep), te, labels[keep], np.array(te.labels)

    raise ValueError(f"Unknown dataset {name}")


def split_forget_retain(labels, forget_class):
    return (np.where(labels == forget_class)[0].tolist(),
            np.where(labels != forget_class)[0].tolist())


def train_base(model, loader, epochs, lr, momentum, device):
    """Trains AllCNN exactly as the BEAIS experiments did, so SSD starts
    from an equivalent base model."""
    import torch.optim as optim
    model.to(device).train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tot = cor = 0; ls = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x); loss = crit(out, y)
            loss.backward(); opt.step()
            ls += loss.item()*x.size(0); cor += (out.argmax(1)==y).sum().item(); tot += x.size(0)
        print(f"    epoch {ep+1}/{epochs}  loss={ls/tot:.4f}  acc={100*cor/tot:.2f}%")
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.to(device).eval()
    cor = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        cor += (model(x).argmax(1) == y).sum().item()
        tot += x.size(0)
    return 100.0 * cor / max(tot, 1)


# ============================================================================
# 3. SSD
# ============================================================================

def fisher_diagonal(model, loader, device, max_batches=None):
    """Diagonal Fisher Information, approximated as the mean squared
    gradient of the loss w.r.t. each parameter."""
    model.to(device).eval()
    crit = nn.CrossEntropyLoss()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}

    nb = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        crit(model(x), y).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        nb += 1

    for n in fisher:
        fisher[n] /= max(nb, 1)
    model.zero_grad()
    return fisher


def ssd_unlearn(model, fim_full, fim_forget, alpha, lam, device):
    """Dampen weights that are disproportionately important to the forget
    set. Returns the modified model plus the count of dampened weights."""
    model = model.to(device)
    n_damp = n_total = 0

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n not in fim_forget:
                continue
            f_f = fim_forget[n]
            f_d = fim_full[n]

            # selection: forget-Fisher far exceeds full-dataset Fisher
            sel = f_f > alpha * f_d

            # dampening factor, capped at 1 so nothing is ever amplified
            beta = torch.ones_like(p)
            safe = f_f > 0
            ratio = torch.zeros_like(p)
            ratio[safe] = lam * f_d[safe] / f_f[safe]
            beta[sel & safe] = torch.clamp(ratio[sel & safe], max=1.0)

            p.mul_(beta)
            n_damp += int((beta < 1).sum().item())
            n_total += p.numel()

    return model, n_damp, n_total


# ============================================================================
# 4. Main
# ============================================================================

def main():
    cfg = SimpleNamespace(
        dataset="cifar10",       # "cifar10" or "svhn"
        forget_class=0,
        batch_size=32,
        base_epochs=20,      # same as the BEAIS runs
        base_lr=0.01,
        momentum=0.9,
        # SSD hyperparameters (Foster et al. defaults; alpha selects how
        # aggressively weights are flagged, lambda how hard they're damped)
        alpha=10.0,
        lam=1.0,
        fisher_batches=200,      # cap Fisher batches on the full set for speed
        seed=42,
        data_dir="./data",
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n=== Loading {cfg.dataset.upper()} ===")
    train_ds, test_ds, tr_lab, te_lab = get_dataset(cfg.dataset, cfg.data_dir)
    f_idx, r_idx = split_forget_retain(tr_lab, cfg.forget_class)
    tf_idx, tr_idx = split_forget_retain(te_lab, cfg.forget_class)
    print(f"D_f (train): {len(f_idx)} | D_r (train): {len(r_idx)}")

    eval_loaders = {
        "train_dr": DataLoader(Subset(train_ds, r_idx), batch_size=256),
        "train_df": DataLoader(Subset(train_ds, f_idx), batch_size=256),
        "test_dr": DataLoader(Subset(test_ds, tr_idx), batch_size=256),
        "test_df": DataLoader(Subset(test_ds, tf_idx), batch_size=256),
    }

    # ---- load the SAME cached base model the BEAIS runs used ----
    ckpt = os.path.join(cfg.data_dir, f"{cfg.dataset}_model0.pt")
    model0 = AllCNN(num_classes=10)
    if os.path.exists(ckpt):
        print(f"\n=== Loading cached base model from {ckpt} ===")
        model0.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        print(f"\n=== Training base AllCNN, {cfg.base_epochs} epochs ===")
        model0 = train_base(model0, DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
                            cfg.base_epochs, cfg.base_lr, cfg.momentum, device)
        torch.save(model0.state_dict(), ckpt)
        print(f"Saved checkpoint to {ckpt}")

    print("\n--- Original model accuracy ---")
    orig = {}
    for k, v in eval_loaders.items():
        orig[k] = evaluate(model0, v, device)
        print(f"  {k}: {orig[k]:.2f}%")

    # ---- SSD ----
    print("\n=== SSD: computing Fisher information ===")
    t0 = time.perf_counter()

    full_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    forget_loader = DataLoader(Subset(train_ds, f_idx), batch_size=cfg.batch_size, shuffle=True)

    print(f"  full training set (capped at {cfg.fisher_batches} batches)...")
    fim_full = fisher_diagonal(copy.deepcopy(model0), full_loader, device, cfg.fisher_batches)
    print("  forget set...")
    fim_forget = fisher_diagonal(copy.deepcopy(model0), forget_loader, device)

    print(f"\n=== SSD: dampening (alpha={cfg.alpha}, lambda={cfg.lam}) ===")
    unlearned, n_damp, n_total = ssd_unlearn(
        copy.deepcopy(model0), fim_full, fim_forget, cfg.alpha, cfg.lam, device)
    elapsed = time.perf_counter() - t0
    print(f"  dampened {n_damp:,} / {n_total:,} weights ({100*n_damp/n_total:.2f}%)")

    print("\n=== SSD (unlearned model) accuracy ===")
    final = {}
    for k, v in eval_loaders.items():
        final[k] = evaluate(unlearned, v, device)
        print(f"  {k}: {final[k]:.2f}%")
    print(f"  unlearning time: {elapsed:.2f}s")

    # ---- comparison ----
    beais = {"cifar10": {"train_dr": 88.56, "train_df": 0.02, "test_dr": 78.70, "test_df": 0.00},
             "svhn":    {"train_dr": 82.80, "train_df": 0.85, "test_dr": 82.74, "test_df": 0.52}}[cfg.dataset]
    print(f"\n=== SSD vs BEAIS on {cfg.dataset.upper()} ===")
    print(f"{'Metric':<12}{'SSD':>10}{'BEAIS':>10}")
    for k in ["train_dr", "train_df", "test_dr", "test_df"]:
        print(f"{k:<12}{final[k]:>10.2f}{beais[k]:>10.2f}")


main()
