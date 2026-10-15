"""
amun_baseline.py — AMUN (Adversarial Machine UNlearning)
Ebrahimpour-Boroojeny, Sundaram & Chandrasekaran, ICML 2025
("Not All Wrong is Bad: Using Adversarial Examples for Unlearning")

Run as a SOTA comparison baseline against BEAIS on CIFAR-10 and SVHN.

How AMUN works:
  1. For every forget sample x (true label y), find the CLOSEST adversarial
     example x' that the model already misclassifies as some y' != y.
     "Closest" matters: a minimal perturbation keeps x' on the model's own
     data manifold, so training on it only nudges the decision boundary
     locally around x rather than distorting the model globally.
  2. Fine-tune the model on the pairs (x', y'), mixed with a subset of the
     retain data to hold accuracy steady.
  The model therefore stops being confident about the forget samples,
  while its behaviour on everything else is largely untouched.

NOTE ON SETTING: AMUN was designed for *random-sample* unlearning (e.g.
a random 10% of CIFAR-10). Here it is applied to *class-wise* unlearning
to match the BEAIS experimental setup, which is a harder case for it.
This is stated in the paper write-up rather than glossed over.

Uses the SAME AllCNN and cached base checkpoints as the BEAIS runs, so
the numbers drop straight into Tables 1 and 2.

Kaggle: paste and run. Set cfg.dataset to "cifar10" or "svhn".
"""

import copy
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset, Dataset
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
    """Trains AllCNN exactly as the BEAIS experiments did."""
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
# 3. AMUN step 1 — closest adversarial examples
# ============================================================================

def make_adversarial(model, loader, device, max_steps, step_size):
    """Minimal-perturbation untargeted attack.

    Takes small gradient-ascent steps on the true-label loss and stops the
    moment each sample flips to a different class, which approximates the
    'closest adversarial example' AMUN relies on. Samples are frozen as
    soon as they flip, so no sample is perturbed more than necessary.
    """
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    crit = nn.CrossEntropyLoss(reduction="sum")
    adv_x, adv_y = [], []
    n_flipped = n_total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = x.clone()
        frozen = torch.zeros(x.size(0), dtype=torch.bool, device=device)

        for _ in range(max_steps):
            if frozen.all():
                break
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = crit(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]

            with torch.no_grad():
                # only move samples that have not flipped yet
                upd = (~frozen).float().view(-1, 1, 1, 1)
                x_adv = x_adv + step_size * grad.sign() * upd
                frozen = frozen | (model(x_adv).argmax(1) != y)
            x_adv = x_adv.detach()

        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        keep = pred != y                      # only genuinely adversarial ones
        n_flipped += int(keep.sum().item())
        n_total += x.size(0)

        if keep.any():
            adv_x.append(x_adv[keep].cpu())
            adv_y.append(pred[keep].cpu())    # train on the adversarial label

    for p in model.parameters():
        p.requires_grad_(True)

    if not adv_x:
        raise RuntimeError("No adversarial examples found - increase max_steps or step_size.")

    print(f"  flipped {n_flipped}/{n_total} forget samples ({100*n_flipped/n_total:.1f}%)")
    return TensorDataset(torch.cat(adv_x), torch.cat(adv_y))


# ============================================================================
# 4. AMUN step 2 — fine-tune on adversarial pairs + retain subset
# ============================================================================

class _IntLabel(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(y)


def amun_finetune(model, adv_ds, train_ds, retain_idx, labels, device,
                  epochs, lr, momentum, batch_size, retain_per_class, seed=42):
    rng = np.random.RandomState(seed)
    ri = np.array(retain_idx)
    rl = labels[ri]
    sub = []
    for c in np.unique(rl):
        idx = ri[rl == c]
        rng.shuffle(idx)
        sub.extend(idx[:retain_per_class].tolist())

    # Normalise labels on BOTH sides. Torchvision returns python ints while
    # the adversarial TensorDataset returns tensors; mixing the two in one
    # batch breaks the default collate function.
    combined = ConcatDataset([_IntLabel(Subset(train_ds, sub)), _IntLabel(adv_ds)])
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True)

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


# ============================================================================
# 5. Main
# ============================================================================

def main():
    cfg = SimpleNamespace(
        dataset="cifar10",        # "cifar10" or "svhn"
        forget_class=0,
        batch_size=32,
        base_epochs=20, base_lr=0.01, momentum=0.9,
        # adversarial search
        adv_max_steps=20,
        adv_step_size=0.01,
        # unlearning fine-tune
        ft_epochs=3, ft_lr=0.01, retain_per_class=1000,
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
    for k, v in eval_loaders.items():
        print(f"  {k}: {evaluate(model0, v, device):.2f}%")

    # ---- AMUN ----
    t0 = time.perf_counter()
    print("\n=== AMUN step 1: generating closest adversarial examples ===")
    forget_loader = DataLoader(Subset(train_ds, f_idx), batch_size=cfg.batch_size)
    adv_ds = make_adversarial(copy.deepcopy(model0), forget_loader, device,
                              cfg.adv_max_steps, cfg.adv_step_size)

    print("\n=== AMUN step 2: fine-tuning on adversarial pairs + retain subset ===")
    unlearned = amun_finetune(copy.deepcopy(model0), adv_ds, train_ds, r_idx, tr_lab,
                              device, cfg.ft_epochs, cfg.ft_lr, cfg.momentum,
                              cfg.batch_size, cfg.retain_per_class, cfg.seed)
    elapsed = time.perf_counter() - t0

    print("\n=== AMUN (unlearned model) accuracy ===")
    final = {}
    for k, v in eval_loaders.items():
        final[k] = evaluate(unlearned, v, device)
        print(f"  {k}: {final[k]:.2f}%")
    print(f"  unlearning time: {elapsed:.2f}s")

    beais = {"cifar10": {"train_dr": 88.56, "train_df": 0.02, "test_dr": 78.70, "test_df": 0.00},
             "svhn":    {"train_dr": 82.80, "train_df": 0.85, "test_dr": 82.74, "test_df": 0.52}}[cfg.dataset]
    print(f"\n=== AMUN vs BEAIS on {cfg.dataset.upper()} ===")
    print(f"{'Metric':<12}{'AMUN':>10}{'BEAIS':>10}")
    for k in ["train_dr", "train_df", "test_dr", "test_df"]:
        print(f"{k:<12}{final[k]:>10.2f}{beais[k]:>10.2f}")


main()
