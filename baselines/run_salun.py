"""
salun_baseline.py — SalUn (Saliency Unlearning) as a head-to-head baseline
against our BEAIS reproduction, on the identical CIFAR-10 / AllCNN setup.

Why this exists: your supervisor asked for comparative analysis against
recent work. Rather than just citing papers, this implements one
(SalUn-style saliency unlearning) ourselves so we get real, directly
comparable numbers on our own pipeline.

How SalUn differs from BEAIS (both unlearn a trained model, but):
  - BEAIS fine-tunes the ENTIRE network (Boundary Expanding relabels and
    fine-tunes on D_f; the Impair Step fine-tunes on D_r-sub + noise) -
    every weight is free to move.
  - SalUn instead computes a SALIENCY MASK first: which weights actually
    matter most for correctly classifying the forget set. Only those
    weights (a configurable top-k% by gradient magnitude) are updated
    during unlearning; everything else is frozen. The forget set is
    given RANDOM incorrect labels ("Random Labeling", a classic
    unlearning signal), and fine-tuning only moves the salient weights.

This reuses the exact same AllCNN architecture, cached base model
checkpoint, and evaluation protocol as beais_single_file.py, so the
numbers are directly comparable to our BEAIS results, no re-training
needed.

Run in Colab: paste the whole file into a cell and run. Uses the same
Drive cache folder, so the CIFAR-10 base model loads instantly.
"""

import copy
import time
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. Model: identical AllCNN to the BEAIS reproduction
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
        x = self.features(x)
        x = self.classifier(x)
        x = self.pool(x)
        return x.flatten(1)


# ============================================================================
# 2. Data (identical to beais_single_file.py)
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
        rng = np.random.RandomState(42)          # same 40,000-image stratified subsample
        keep = []
        for c in range(10):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            keep.extend(idx[:4000].tolist())
        rng.shuffle(keep)
        return Subset(tr, keep), te, labels[keep], np.array(te.labels)

    raise ValueError(f"Unknown dataset {name}")


def train_base(model, loader, epochs, lr, momentum, device):
    """Trains AllCNN exactly as the BEAIS experiments did."""
    model.to(device).train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tot = cor = 0; ls = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); out = model(x); loss = crit(out, y)
            loss.backward(); opt.step()
            ls += loss.item()*x.size(0); cor += (out.argmax(1)==y).sum().item(); tot += x.size(0)
        print(f"    epoch {ep+1}/{epochs}  loss={ls/tot:.4f}  acc={100*cor/tot:.2f}%")
    return model


def split_forget_retain(labels, forget_class):
    forget_idx = np.where(labels == forget_class)[0].tolist()
    retain_idx = np.where(labels != forget_class)[0].tolist()
    return forget_idx, retain_idx


@torch.no_grad()
def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / max(total, 1)


# ============================================================================
# 3. SalUn Step 1: compute the saliency mask
# ============================================================================

def compute_saliency_mask(model, forget_loader, device, sparsity=0.5):
    """Accumulate |gradient| of the forget-set loss w.r.t. every weight,
    then keep only the top `sparsity` fraction (by magnitude) as
    'salient' - these are the only weights SalUn will update. Everything
    else gets frozen during unlearning."""
    model.to(device).eval()
    model.zero_grad()
    crit = nn.CrossEntropyLoss()

    grad_accum = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    n_batches = 0
    for x, y in forget_loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_accum[name] += p.grad.abs()
        n_batches += 1

    # Global threshold: pool every parameter's averaged |grad| together,
    # take the top `sparsity` fraction across the WHOLE network at once
    # (matches SalUn's global saliency ranking, not a per-layer one).
    all_vals = torch.cat([(g / n_batches).flatten() for g in grad_accum.values()])
    k = int(len(all_vals) * sparsity)
    threshold = torch.topk(all_vals, k, largest=True).values.min() if k > 0 else float("inf")

    mask = {}
    for name, g in grad_accum.items():
        mask[name] = ((g / n_batches) >= threshold).float()

    kept = sum(m.sum().item() for m in mask.values())
    total = sum(m.numel() for m in mask.values())
    print(f"  [SalUn] saliency mask: {kept:.0f}/{total} weights ({100*kept/total:.1f}%) marked salient")
    return mask


# ============================================================================
# 4. SalUn Step 2: Random Labeling + saliency-restricted fine-tuning
# ============================================================================

class RandomLabeled(Dataset):
    """Sec. 'Random Labeling' signal: forget-set samples get a fixed
    random INCORRECT label (never their true class)."""
    def __init__(self, base, idx, num_classes, seed=42):
        self.base, self.idx = base, idx
        rng = np.random.RandomState(seed)
        self.fake_labels = []
        for i in idx:
            true_y = base[i][1]
            choices = [c for c in range(num_classes) if c != true_y]
            self.fake_labels.append(int(rng.choice(choices)))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, _ = self.base[self.idx[i]]
        return x, self.fake_labels[i]


def salun_unlearn(model, mask, train_ds, forget_idx, retain_idx, device,
                   epochs, lr, momentum, batch_size, seed=42):
    num_classes = model.num_classes
    # Normalise labels on both sides: torchvision returns python ints while
    # some wrappers return tensors, and mixing them breaks default collate.
    class _IntLabel(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            return x, int(y)

    combined = torch.utils.data.ConcatDataset([
        _IntLabel(Subset(train_ds, retain_idx)),
        _IntLabel(RandomLabeled(train_ds, forget_idx, num_classes, seed)),
    ])
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True)

    model.to(device).train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = nn.CrossEntropyLoss()

    print("  [SalUn] fine-tuning (only salient weights receive gradient updates)...")
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()

            # THE core SalUn mechanism: zero out the gradient for every
            # non-salient weight BEFORE the optimizer step, so only the
            # masked (salient) weights actually move.
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p.grad *= mask[name]

            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"    epoch {ep+1}/{epochs}  loss={loss_sum/total:.4f}  acc={100*correct/total:.2f}%")
    return model


# ============================================================================
# 5. Main
# ============================================================================

def main():
    args = SimpleNamespace(
        dataset="cifar10",          # "cifar10" or "svhn"
        forget_class=0,
        batch_size=32,
        base_epochs=20, base_lr=0.01,
        sparsity=0.5,                # SalUn default: top 50% of weights by saliency
        salun_epochs=3,              # matched to BEAIS's best impair_epochs=3 for a fair comparison
        salun_lr=0.01,               # matched to BEAIS's SGD lr for a fair comparison
        momentum=0.9,
        seed=42,
        data_dir="./data",
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n=== Loading {args.dataset.upper()} ===")
    train_ds, test_ds, train_labels, test_labels = get_dataset(args.dataset, args.data_dir)
    forget_idx, retain_idx = split_forget_retain(train_labels, args.forget_class)
    test_forget_idx, test_retain_idx = split_forget_retain(test_labels, args.forget_class)
    print(f"D_f (train): {len(forget_idx)} | D_r (train): {len(retain_idx)}")

    eval_loaders = {
        "train_dr": DataLoader(Subset(train_ds, retain_idx), batch_size=256),
        "train_df": DataLoader(Subset(train_ds, forget_idx), batch_size=256),
        "test_dr": DataLoader(Subset(test_ds, test_retain_idx), batch_size=256),
        "test_df": DataLoader(Subset(test_ds, test_forget_idx), batch_size=256),
    }
    forget_loader = DataLoader(Subset(train_ds, forget_idx), batch_size=args.batch_size, shuffle=True)

    # ---- load the SAME cached base model BEAIS used ----
    ckpt_path = os.path.join(args.data_dir, f"{args.dataset}_model0.pt")
    model0 = AllCNN(num_classes=10)
    if os.path.exists(ckpt_path):
        print(f"\n=== Loading cached base model from {ckpt_path} ===")
        model0.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        print(f"\n=== Training base AllCNN, {args.base_epochs} epochs ===")
        model0 = train_base(model0, DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
                            args.base_epochs, args.base_lr, args.momentum, device)
        torch.save(model0.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print("\n--- Original model accuracy ---")
    for k, v in eval_loaders.items():
        print(f"  {k}: {evaluate(model0, v, device):.2f}%")

    # ---- SalUn Step 1: saliency mask ----
    print("\n=== SalUn Step 1: Computing saliency mask ===")
    mask = compute_saliency_mask(copy.deepcopy(model0), forget_loader, device, sparsity=args.sparsity)

    # ---- SalUn Step 2: masked fine-tuning ----
    print("\n=== SalUn Step 2: Saliency-restricted unlearning ===")
    t0 = time.perf_counter()
    unlearned = salun_unlearn(copy.deepcopy(model0), mask, train_ds, forget_idx, retain_idx,
                               device, args.salun_epochs, args.salun_lr, args.momentum,
                               args.batch_size, args.seed)
    elapsed = time.perf_counter() - t0

    print("\n=== SalUn (unlearned model) accuracy ===")
    for k, v in eval_loaders.items():
        print(f"  {k}: {evaluate(unlearned, v, device):.2f}%")
    print(f"  unlearning time: {elapsed:.2f}s")

    print(f"\n=== SalUn vs BEAIS on {args.dataset.upper()} ===")
    beais = {"cifar10": {"train_dr": 88.56, "train_df": 0.02, "test_dr": 78.70, "test_df": 0.00},
             "svhn":    {"train_dr": 82.80, "train_df": 0.85, "test_dr": 82.74, "test_df": 0.52}}[args.dataset]
    print(f"{'Metric':<12}{'SalUn':>10}{'BEAIS':>10}")
    for k in ["train_dr", "train_df", "test_dr", "test_df"]:
        print(f"{k:<12}{evaluate(unlearned, eval_loaders[k], device):>10.2f}{beais[k]:>10.2f}")


main()
