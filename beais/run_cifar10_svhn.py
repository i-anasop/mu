"""
beais_single_file.py

A single, self-contained reproduction of BEAIS (Boundary Expanding and
Impair Step) from the paper "Rethinking Image Classification with Machine
Unlearning" (Tariq et al.). No other files needed - just this one script.

What it does, step by step:
  1. Loads CIFAR-10 (or SVHN) via torchvision.
  2. Trains the ORIGINAL model M_theta0 on the full dataset  ->  Eq. 1.
  3. Splits the data into Forget set (D_f, one whole class) and
     Retain set (D_r, everything else)                       ->  Sec 3.1.
  4. Runs BEAIS Algorithm 1 on top of M_theta0:
       Step 1 - Boundary Expanding (Eq. 2)
       Step 2 - Impair Step (Eq. 3): make error-maximizing noise,
                fine-tune on Retain-subset + noise
  5. Evaluates accuracy on D_r and D_f (train + test), same layout as
     Table 1 / Table 2 in the paper, and prints a side-by-side comparison
     against the paper's own reported numbers so you can eyeball the match.

Run it:
    python beais_single_file.py                     # CIFAR-10, forgets class 0
    python beais_single_file.py --dataset svhn
    python beais_single_file.py --forget_class 3 --epochs 20

Everything the paper leaves unspecified is called out with an
"ASSUMPTION:" comment right where it's used - nothing is silently guessed.
"""

import copy
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms


# ============================================================================
# 0. Reproducibility
# ============================================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. Model: AllCNN (Sec. 4.2: "the AllCNN model is used for all experiments")
# ============================================================================
# ASSUMPTION: the paper cites AllCNN without giving exact layer widths; this
# is the standard fully-convolutional AllCNN used across this exact line of
# unlearning papers (Golatkar 2020, Tarun 2023, Chen 2023). It's "headless"
# (no fully-connected layer) - the final 1x1 conv's channel count IS the
# number of classes, which is what makes Boundary Expanding's "add a pseudo
# class neuron, then discard it" trick (Algorithm 1, lines 2 & 4) trivial.

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
            block(in_channels, 96), block(96, 96), block(96, 96, s=2),   # 32->16
            nn.Dropout(0.5),
            block(96, 192), block(192, 192), block(192, 192, s=2),      # 16->8
            nn.Dropout(0.5),
            block(192, 192, k=3, p=0),                                   # 8->6
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

    def expand_classes(self, extra=1):
        """Algorithm 1, line 2: add a pseudo-class output neuron."""
        new_model = AllCNN(num_classes=self.num_classes + extra)
        new_model.features.load_state_dict(self.features.state_dict())
        with torch.no_grad():
            new_model.classifier.weight[: self.num_classes] = self.classifier.weight
            new_model.classifier.bias[: self.num_classes] = self.classifier.bias
        return new_model

    def shrink_classes(self, num_classes):
        """Algorithm 1, line 4: discard the pseudo-class neuron."""
        new_model = AllCNN(num_classes=num_classes)
        new_model.features.load_state_dict(self.features.state_dict())
        with torch.no_grad():
            new_model.classifier.weight[:] = self.classifier.weight[:num_classes]
            new_model.classifier.bias[:] = self.classifier.bias[:num_classes]
        return new_model


# ============================================================================
# 2. Data: CIFAR-10 / SVHN, Forget/Retain split (Sec. 3.1, 4.1)
# ============================================================================

def get_dataset(name, data_dir="./data"):
    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
        return train, test, np.array(train.targets), np.array(test.targets)

    if name == "svhn":
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train = datasets.SVHN(data_dir, split="train", download=True, transform=tfm)
        test = datasets.SVHN(data_dir, split="test", download=True, transform=tfm)
        train_labels = np.array(train.labels)

        # Sec. 4.1: SVHN train is subsampled to 40,000 images.
        # ASSUMPTION: sampling scheme unspecified -> stratified (class-balanced)
        # random subsample with a fixed seed.
        rng = np.random.RandomState(42)
        keep = []
        per_class = 40000 // 10
        for c in range(10):
            idx = np.where(train_labels == c)[0]
            rng.shuffle(idx)
            keep.extend(idx[:per_class].tolist())
        rng.shuffle(keep)
        train = Subset(train, keep)
        return train, test, train_labels[keep], np.array(test.labels)

    raise ValueError(f"Unknown dataset {name}")


def split_forget_retain(labels, forget_class):
    """Sec. 3.1: D_f = samples whose label is in the forget class(es),
    D_r = everything else. Class-wise unlearning only (whole class)."""
    forget_idx = np.where(labels == forget_class)[0].tolist()
    retain_idx = np.where(labels != forget_class)[0].tolist()
    return forget_idx, retain_idx


# ============================================================================
# 3. Generic train / evaluate loop (Eq. 1, and reused for every fine-tune
#    stage: SGD, cross-entropy - Sec. 4.2)
# ============================================================================

def train(model, loader, epochs, lr, momentum, device):
    model.to(device).train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"    epoch {ep+1}/{epochs}  loss={loss_sum/total:.4f}  acc={100*correct/total:.2f}%")
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """Sec. 4.4.1: plain classification accuracy."""
    model.to(device).eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / max(total, 1)


# ============================================================================
# 4. BEAIS Step 1: Boundary Expanding (Sec. 3.2.1, Eq. 2, Algorithm 1 L1-4)
# ============================================================================

def boundary_expanding(model, train_ds, forget_idx, device, epochs, lr, momentum, batch_size):
    """
        theta' = argmin_theta  sum_{(x,y_pseudo) in D_f}  L(x, y_pseudo, theta_0)
    Reassign every forget-set sample to a brand-new pseudo class, fine-tune
    on JUST that relabeled forget set, then discard the pseudo neuron.
    """
    num_classes = model.num_classes
    y_pseudo = num_classes  # the new class index

    expanded = model.expand_classes(extra=1)

    # relabel forget samples -> y_pseudo
    class Relabeled(torch.utils.data.Dataset):
        def __init__(self, base, idx, label):
            self.base, self.idx, self.label = base, idx, label
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            x, _ = self.base[self.idx[i]]
            return x, self.label

    relabeled_loader = DataLoader(Relabeled(train_ds, forget_idx, y_pseudo),
                                   batch_size=batch_size, shuffle=True)

    print("  [Boundary Expanding] fine-tuning on relabeled forget set...")
    expanded = train(expanded, relabeled_loader, epochs, lr, momentum, device)

    # Algorithm 1, line 4: discard the pseudo-class neuron.
    return expanded.shrink_classes(num_classes)


# ============================================================================
# 5. BEAIS Step 2: Impair Step (Sec. 3.2.2, Eq. 3, Algorithm 1 L5-8)
# ============================================================================

def make_error_maximizing_noise(original_model, input_shape, forget_label, device,
                                 batch_size, repetitions, lam, noise_steps, noise_lr):
    """
        argmin_Z  [ -L(M_theta0(Z), y_f)  +  lambda * ||Z|| ]
    Freeze the ORIGINAL model, start from random noise Z ~ N(0,1), and push
    it (via gradient descent on Z, not on the model) to maximize the
    classification loss against the forget label - i.e. make noise that the
    frozen model is forced to associate with the forgotten class.

    ASSUMPTION: paper gives no numeric lambda, no optimizer/lr for Z, and no
    step count per batch (only "repeated 120 times" for how many BATCHES of
    noise to generate). We default to lambda=0.1, Adam(lr=0.1), 10 steps -
    same order of magnitude as (Tarun et al., 2023), which this step is
    explicitly built on.
    """
    original_model.to(device).eval()
    for p in original_model.parameters():
        p.requires_grad_(False)  # "the weights of the original model are frozen"

    crit = nn.CrossEntropyLoss()
    noise_batches = []

    print(f"  [Impair Step] generating {repetitions} noise batches...")
    for _ in range(repetitions):
        z = torch.randn(batch_size, *input_shape, device=device, requires_grad=True)
        y_f = torch.full((batch_size,), forget_label, dtype=torch.long, device=device)
        z_opt = optim.Adam([z], lr=noise_lr)

        for _ in range(noise_steps):
            z_opt.zero_grad()
            logits = original_model(z)
            # minus sign: MAXIMIZE the forget-class loss (i.e. minimize its negative)
            loss = -crit(logits, y_f) + lam * z.norm()
            loss.backward()
            z_opt.step()

        noise_batches.append(z.detach().cpu())

    for p in original_model.parameters():
        p.requires_grad_(True)

    Z = torch.cat(noise_batches, dim=0)
    labels = torch.full((Z.size(0),), forget_label, dtype=torch.long)
    return TensorDataset(Z, labels)


def impair_step(theta_prime, noisy_dataset, train_ds, retain_idx, labels,
                 device, epochs, lr, momentum, batch_size, retain_sub_per_class):
    """
    Algorithm 1, line 6: partition D_r into D_r-sub (1,000 samples/class).
    Algorithm 1, line 8: fine-tune theta' on D_r-sub UNION noise.
    """
    rng = np.random.RandomState(42)
    retain_idx = np.array(retain_idx)
    retain_labels = labels[retain_idx]
    retain_sub_idx = []
    for c in np.unique(retain_labels):
        idx = retain_idx[retain_labels == c]
        rng.shuffle(idx)
        retain_sub_idx.extend(idx[:retain_sub_per_class].tolist())

    # BUG FIX: CIFAR10/SVHN samples yield (tensor, python-int) labels, while
    # the noise TensorDataset yields (tensor, torch-tensor) labels. Mixing
    # both types inside one DataLoader batch crashes the default collate
    # function ("expected Tensor ... but got int"). Normalize the noise
    # labels to plain Python ints so every sample in the combined dataset
    # has the exact same (tensor, int) shape.
    class _IntLabel(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            return x, int(y)

    combined = ConcatDataset([Subset(train_ds, retain_sub_idx), _IntLabel(noisy_dataset)])
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True)

    print("  [Impair Step] fine-tuning on retain-subset + noise...")
    return train(theta_prime, loader, epochs, lr, momentum, device)


# ============================================================================
# 6. Paper's reported numbers, for a quick side-by-side sanity check
#    (Table 1 = CIFAR-10, Table 2 = SVHN, "Our" row)
# ============================================================================

PAPER_RESULTS = {
    "cifar10": {"train_dr": 88.56, "train_df": 0.02, "test_dr": 78.70, "test_df": 0.00},
    "svhn":    {"train_dr": 82.80, "train_df": 0.85, "test_dr": 82.74, "test_df": 0.52},
}


# ============================================================================
# 7. Main
# ============================================================================

def main():
    # ---- Edit these directly instead of passing CLI flags (Colab-friendly) ----
    args = SimpleNamespace(
        dataset="cifar10",          # "cifar10" or "svhn"
        forget_class=0,             # ASSUMPTION: paper never says which class is forgotten
        epochs=20,                  # Sec. 4.2: 20 epochs for base model
        base_lr=0.01,
        momentum=0.9,
        batch_size=32,              # Sec. 4.2
        # EXPERIMENT 2 (see experiment_log.md): boundary_epochs 1 -> 3.
        # Untested axis so far. Boundary Expanding only trains on D_f (5,000
        # samples) - it never touches D_r, so more epochs here carry near-
        # zero risk to retain accuracy, unlike pushing the Impair Step
        # further (which already hurt retain acc in Exp 1). Hypothesis:
        # giving Boundary Expanding more time to fully separate the forget
        # class gives the Impair Step a better starting point, closing more
        # of the forget-accuracy gap without the retain-accuracy cost.
        boundary_epochs=3,
        boundary_lr=0.01,
        # TUNED: paper reports ~0% forget accuracy. 1 impair epoch with 10
        # noise-optimization steps left us at ~4% forget acc (see chat) -
        # bumping epochs/steps pushes the noise harder and gives it more
        # fine-tuning passes to actually erase the forget class. This is
        # still within the paper's stated "Impair Step" mechanism (Eq. 3),
        # just spending more compute on the same objective (ASSUMPTION,
        # since the paper gives no exact step/epoch count - see
        # assumptions.md item A6).
        # Sec. 4.2 gives no exact epoch/step count for this stage.
        # Empirically: 3 epochs / 25 noise-opt steps beat 5 epochs / 40
        # steps on EVERY metric (retain went down, forget went up when
        # pushed further) - matches the paper's own ablation finding
        # (Sec. 5.5, Fig. 5d) that more Impair passes give diminishing/
        # negative returns. Settling here rather than chasing exact 0%.
        impair_epochs=3,
        impair_lr=0.01,
        retain_sub_per_class=1000,  # Sec. 4.2
        noise_repetitions=120,      # Sec. 4.2
        noise_steps=25,
        noise_lr=0.05,              # TUNED (was 0.1): more stable noise optimization
        lambda_noise=0.05,          # TUNED (was 0.1): less regularization -> noise can push harder
        seed=42,
        # ---- Dataset caching (avoids re-downloading every runtime) --------
        # Point this at Google Drive so the ~170MB download only ever
        # happens ONCE, even across separate Colab sessions/runtime resets.
        # If you're not on Colab, or don't want Drive, just set this to
        # "./data" instead.
        data_dir="./data",
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 0. Mount Google Drive so the dataset persists across sessions -----

    # ---- 1. Data -----------------------------------------------------------
    print(f"\n=== Loading {args.dataset.upper()} (from {args.data_dir}) ===")
    train_ds, test_ds, train_labels, test_labels = get_dataset(args.dataset, args.data_dir)
    forget_idx, retain_idx = split_forget_retain(train_labels, args.forget_class)
    test_forget_idx, test_retain_idx = split_forget_retain(test_labels, args.forget_class)
    print(f"D_f (train): {len(forget_idx)} samples | D_r (train): {len(retain_idx)} samples")

    full_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loaders = {
        "train_dr": DataLoader(Subset(train_ds, retain_idx), batch_size=256),
        "train_df": DataLoader(Subset(train_ds, forget_idx), batch_size=256),
        "test_dr": DataLoader(Subset(test_ds, test_retain_idx), batch_size=256),
        "test_df": DataLoader(Subset(test_ds, test_forget_idx), batch_size=256),
    }

    # ---- 2. Train original model M_theta0 (Eq. 1) - CACHED to Drive --------
    # This is the expensive 20-epoch step. Once trained, it's saved to
    # args.data_dir so re-running this cell (e.g. after tweaking the Impair
    # Step below) skips straight to loading it instead of retraining.
    import os
    ckpt_path = os.path.join(args.data_dir, f"{args.dataset}_model0.pt")
    model0 = AllCNN(num_classes=10)
    if os.path.exists(ckpt_path):
        print(f"\n=== Loading cached original model from {ckpt_path} ===")
        model0.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        print(f"\n=== Training original model (Eq. 1), {args.epochs} epochs ===")
        model0 = train(model0, full_loader, args.epochs, args.base_lr, args.momentum, device)
        os.makedirs(args.data_dir, exist_ok=True)
        torch.save(model0.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print("\n--- Original model accuracy ---")
    orig_acc = {k: evaluate(model0, v, device) for k, v in eval_loaders.items()}
    for k, v in orig_acc.items():
        print(f"  {k}: {v:.2f}%")

    # ---- 3. BEAIS Step 1: Boundary Expanding --------------------------------
    print("\n=== BEAIS Step 1: Boundary Expanding (Sec. 3.2.1) ===")
    theta_prime = boundary_expanding(copy.deepcopy(model0), train_ds, forget_idx, device,
                                      args.boundary_epochs, args.boundary_lr,
                                      args.momentum, args.batch_size)

    # ---- 4. BEAIS Step 2: Impair Step ----------------------------------------
    print("\n=== BEAIS Step 2: Impair Step (Sec. 3.2.2) ===")
    input_shape = (3, 32, 32)
    t0 = time.perf_counter()
    noisy_dataset = make_error_maximizing_noise(
        model0, input_shape, args.forget_class, device,
        args.batch_size, args.noise_repetitions, args.lambda_noise,
        args.noise_steps, args.noise_lr,
    )
    theta_u = impair_step(theta_prime, noisy_dataset, train_ds, retain_idx, train_labels,
                           device, args.impair_epochs, args.impair_lr, args.momentum,
                           args.batch_size, args.retain_sub_per_class)
    beais_time = time.perf_counter() - t0

    # ---- 5. Evaluate + compare to the paper ----------------------------------
    print("\n=== BEAIS (unlearned model) accuracy ===")
    beais_acc = {k: evaluate(theta_u, v, device) for k, v in eval_loaders.items()}
    for k, v in beais_acc.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  unlearning time: {beais_time:.2f}s")

    paper = PAPER_RESULTS[args.dataset]
    print(f"\n=== Comparison vs. paper's 'Our' row ({args.dataset.upper()}, Table 1/2) ===")
    print(f"{'Metric':<12}{'Ours':>10}{'Paper':>10}{'Diff':>10}")
    for k in ["train_dr", "train_df", "test_dr", "test_df"]:
        diff = beais_acc[k] - paper[k]
        print(f"{k:<12}{beais_acc[k]:>10.2f}{paper[k]:>10.2f}{diff:>10.2f}")


main()
