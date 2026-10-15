"""
beais_rvlcdip_final.py — BEAIS on the COMPLETE RVL-CDIP dataset.

Standalone script (no dependency on other files) - reuses the proven
transfer-learning pipeline. Uses the FULL dataset (~320K train, ~40K
test) - no subsetting, per supervisor's explicit request.

KAGGLE SETUP: Add Input -> search "rvlcdip" by ducktard -> Add.
Attaching this dataset to a new notebook takes a while (400,004
individual files) - that's normal, not a bug. GPU on. Paste, run.
"""

import copy
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, TensorDataset
from torchvision import transforms
from PIL import Image


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. Model: frozen pretrained ResNet18 backbone + trainable classifier head
# ============================================================================

class TransferModel(nn.Module):
    def __init__(self, num_classes, features_module=None, freeze_backbone=True):
        super().__init__()
        if features_module is None:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            for p in self.features.parameters():
                p.requires_grad_(not freeze_backbone)
        else:
            self.features = features_module
        self.classifier = nn.Linear(512, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.classifier(feat)

    def freeze_backbone_now(self):
        """Called after base-model training finishes, right before BEAIS
        starts - locks the backbone back down so Boundary Expanding and
        the Impair Step operate exactly as proven (only the small
        classifier head moves during those stages), just on top of a
        stronger, fully fine-tuned starting model."""
        for p in self.features.parameters():
            p.requires_grad_(False)
        return self

    def expand_classes(self, extra=1):
        new_model = TransferModel(self.num_classes + extra, features_module=self.features)
        with torch.no_grad():
            new_model.classifier.weight[: self.num_classes] = self.classifier.weight
            new_model.classifier.bias[: self.num_classes] = self.classifier.bias
        return new_model

    def shrink_classes(self, num_classes):
        new_model = TransferModel(num_classes, features_module=self.features)
        with torch.no_grad():
            new_model.classifier.weight[:] = self.classifier.weight[:num_classes]
            new_model.classifier.bias[:] = self.classifier.bias[:num_classes]
        return new_model


# ============================================================================
# 2. Data: RVL-CDIP (COMPLETE - official label-file format, no subsetting)
# ============================================================================

RVLCDIP_CLASS_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific report", "scientific publication", "specification",
    "file folder", "news article", "budget", "invoice", "presentation",
    "questionnaire", "resume", "memo",
]


class RVLCDIPLabelFileDataset(Dataset):
    """Parses an official label file (train.txt/test.txt) and lazily
    loads images from disk one at a time - memory-safe regardless of
    dataset size."""
    def __init__(self, images_root, samples, transform):
        self.images_root = images_root
        self.samples = samples  # list of (relative_path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # FIX: real RVL-CDIP data has a handful of corrupt/unreadable files
        # scattered across 360K images (known issue with this legacy
        # dataset). One bad file used to crash the entire multi-hour run.
        # Now: skip it and try the next sample instead.
        for attempt in range(5):
            rel_path, label = self.samples[i]
            full_path = os.path.join(self.images_root, rel_path)
            try:
                img = Image.open(full_path).convert("L")
                return self.transform(img), label
            except Exception as e:
                if attempt == 0:
                    print(f"    [skipping corrupt image] {full_path} ({e})")
                i = (i + 1) % len(self.samples)
        raise RuntimeError(f"5 corrupt images in a row near index {i} - something's more seriously wrong.")


def _parse_label_file(label_file_path):
    samples = []
    with open(label_file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(maxsplit=1)
            samples.append((rel_path, int(label)))
    return samples


def get_rvlcdip_dataset(images_root, train_label_file, test_label_file, seed=42):
    """No subsetting - COMPLETE dataset, per supervisor's explicit request."""
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print(f"  Parsing {train_label_file} ...")
    train_samples = _parse_label_file(train_label_file)
    print(f"  Parsing {test_label_file} ...")
    test_samples = _parse_label_file(test_label_file)

    class_names = RVLCDIP_CLASS_NAMES
    train_labels = np.array([lab for _, lab in train_samples])
    test_labels = np.array([lab for _, lab in test_samples])
    print(f"  Classes (16): {class_names}")
    print(f"  Train: {len(train_samples)}  Test: {len(test_samples)}  (complete dataset - no subsetting)")

    train_ds = RVLCDIPLabelFileDataset(images_root, train_samples, tfm)
    test_ds = RVLCDIPLabelFileDataset(images_root, test_samples, tfm)
    return train_ds, test_ds, train_labels, test_labels, class_names


def split_forget_retain(labels, forget_class):
    forget_idx = np.where(labels == forget_class)[0].tolist()
    retain_idx = np.where(labels != forget_class)[0].tolist()
    return forget_idx, retain_idx


# ============================================================================
# 3. Train / evaluate, with progress printing every 100 batches
# ============================================================================

def train(model, loader, epochs, lr, momentum, device, tag=""):
    model.to(device).train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        t_epoch_start = time.perf_counter()
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.perf_counter() - t_epoch_start
                rate = (batch_idx + 1) / elapsed
                print(f"    {tag}batch {batch_idx+1}  ({total} images seen, "
                      f"{elapsed:.1f}s elapsed, {rate:.2f} batches/sec, "
                      f"running acc={100*correct/total:.2f}%)")
        print(f"  {tag}epoch {ep+1}/{epochs}  loss={loss_sum/total:.4f}  acc={100*correct/total:.2f}%")
    return model


@torch.no_grad()
def evaluate(model, loader, device, tag=""):
    model.to(device).eval()
    correct, total = 0, 0
    t_start = time.perf_counter()
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
        # FIX: this had zero progress output - on a 300K-image loader that
        # looks identical to a hang. Print every 200 batches instead.
        if (batch_idx + 1) % 200 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"    {tag}eval batch {batch_idx+1}  ({total} images checked, {elapsed:.1f}s elapsed)")
    return 100.0 * correct / max(total, 1)


# ============================================================================
# 4. BEAIS Step 1: Boundary Expanding
# ============================================================================

def boundary_expanding(model, train_ds, forget_idx, device, epochs, lr, momentum, batch_size):
    num_classes = model.num_classes
    y_pseudo = num_classes

    class Relabeled(Dataset):
        def __init__(self, base, idx, label):
            self.base, self.idx, self.label = base, idx, label
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            x, _ = self.base[self.idx[i]]
            return x, self.label

    expanded = model.expand_classes(extra=1)
    loader = DataLoader(Relabeled(train_ds, forget_idx, y_pseudo), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("  [Boundary Expanding] fine-tuning on relabeled forget set...")
    expanded = train(expanded, loader, epochs, lr, momentum, device, tag="  BE ")
    return expanded.shrink_classes(num_classes)


# ============================================================================
# 5. BEAIS Step 2: Impair Step
# ============================================================================

def make_error_maximizing_noise(original_model, input_shape, forget_label, device,
                                 batch_size, repetitions, lam, noise_steps, noise_lr):
    original_model.to(device).eval()
    for p in original_model.parameters():
        p.requires_grad_(False)
    crit = nn.CrossEntropyLoss()
    noise_batches = []
    print(f"  [Impair Step] generating {repetitions} noise batches...")
    for rep in range(repetitions):
        z = torch.randn(batch_size, *input_shape, device=device, requires_grad=True)
        y_f = torch.full((batch_size,), forget_label, dtype=torch.long, device=device)
        z_opt = optim.Adam([z], lr=noise_lr)
        for _ in range(noise_steps):
            z_opt.zero_grad()
            logits = original_model(z)
            loss = -crit(logits, y_f) + lam * z.norm()
            loss.backward()
            z_opt.step()
        noise_batches.append(z.detach().cpu())
        if (rep + 1) % 20 == 0:
            print(f"    ...noise batch {rep+1}/{repetitions}")
    for p in original_model.parameters():
        p.requires_grad_(True)
    Z = torch.cat(noise_batches, dim=0)
    labels = torch.full((Z.size(0),), forget_label, dtype=torch.long)
    return TensorDataset(Z, labels)


def impair_step(theta_prime, noisy_dataset, train_ds, retain_idx, labels, device,
                 epochs, lr, momentum, batch_size, retain_sub_per_class):
    rng = np.random.RandomState(42)
    retain_idx = np.array(retain_idx)
    retain_labels = labels[retain_idx]
    retain_sub_idx = []
    for c in np.unique(retain_labels):
        idx = retain_idx[retain_labels == c]
        rng.shuffle(idx)
        retain_sub_idx.extend(idx[:retain_sub_per_class].tolist())

    class _IntLabel(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            return x, int(y)

    combined = ConcatDataset([Subset(train_ds, retain_sub_idx), _IntLabel(noisy_dataset)])
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("  [Impair Step] fine-tuning on retain-subset + noise...")
    return train(theta_prime, loader, epochs, lr, momentum, device, tag="  IS ")


# ============================================================================
# 6. Main
# ============================================================================

def main():
    cfg = SimpleNamespace(
        base_epochs=2,           # LOW ON PURPOSE - ~320K images in 1 pass already exceeds 15 epochs on a small set
        base_lr=0.001,           # LOWERED from 0.01 - full backbone fine-tuning needs a gentler LR than head-only training
        momentum=0.9,
        batch_size=32,
        boundary_epochs=1,
        boundary_lr=0.01,
        impair_epochs=3,
        impair_lr=0.01,
        retain_sub_per_class=1000,  # matches the original paper's own value - finally have enough data for it
        noise_repetitions=60,
        noise_steps=25,
        noise_lr=0.05,
        lambda_noise=0.05,
        seed=42,
        forget_class=0,
        num_workers=4,           # parallel image loading - the 12+ hour runtime was I/O-bound (eval was as slow as
                                  # training despite having no backward pass), not GPU-bound. This is the real fix for that.
        # point images_root and the label files at your RVL-CDIP copy
        images_root="./data/rvl-cdip/images",
        train_label_file="./data/rvl-cdip/labels/train.txt",
        test_label_file="./data/rvl-cdip/labels/test.txt",
        output_dir="./outputs",
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("\n=== Loading RVL-CDIP (complete dataset) ===")
    train_ds, test_ds, train_labels, test_labels, class_names = get_rvlcdip_dataset(
        cfg.images_root, cfg.train_label_file, cfg.test_label_file, seed=cfg.seed
    )
    forget_idx, retain_idx = split_forget_retain(train_labels, cfg.forget_class)
    test_forget_idx, test_retain_idx = split_forget_retain(test_labels, cfg.forget_class)
    print(f"Forget class: '{class_names[cfg.forget_class]}'")
    print(f"D_f (train): {len(forget_idx)} | D_r (train): {len(retain_idx)}")

    eval_loaders = {
        "train_dr": DataLoader(Subset(train_ds, retain_idx), batch_size=64, num_workers=cfg.num_workers, pin_memory=True),
        "train_df": DataLoader(Subset(train_ds, forget_idx), batch_size=64, num_workers=cfg.num_workers, pin_memory=True),
        "test_dr": DataLoader(Subset(test_ds, test_retain_idx), batch_size=64, num_workers=cfg.num_workers, pin_memory=True),
        "test_df": DataLoader(Subset(test_ds, test_forget_idx), batch_size=64, num_workers=cfg.num_workers, pin_memory=True),
    }

    # NEW checkpoint name - the old one was trained frozen-backbone (head-only),
    # a fundamentally different, weaker model. Force a fresh full fine-tune.
    ckpt_path = os.path.join(cfg.output_dir, "rvlcdip_finetuned_model0.pt")
    if os.path.exists(ckpt_path):
        print(f"\n=== Loading cached FULLY FINE-TUNED base model from {ckpt_path} ===")
        model0 = TransferModel(num_classes=len(class_names), freeze_backbone=False)
        model0.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        print(f"\n=== Fully fine-tuning base model (backbone UNFROZEN), {cfg.base_epochs} epochs ===")
        model0 = TransferModel(num_classes=len(class_names), freeze_backbone=False)
        full_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)
        model0 = train(model0, full_loader, cfg.base_epochs, cfg.base_lr, cfg.momentum, device, tag="base ")
        torch.save(model0.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # Lock the backbone back down before BEAIS - Boundary Expanding + Impair
    # Step run exactly as proven (only the head moves), just starting from
    # a much stronger, fully fine-tuned model instead of a frozen one.
    model0.freeze_backbone_now()

    print("\n--- Original model accuracy ---")
    for k, v in eval_loaders.items():
        print(f"  {k}: {evaluate(model0, v, device):.2f}%")

    print("\n=== BEAIS Step 1: Boundary Expanding ===")
    theta_prime = boundary_expanding(copy.deepcopy(model0), train_ds, forget_idx, device,
                                      cfg.boundary_epochs, cfg.boundary_lr, cfg.momentum, cfg.batch_size)

    print("\n=== BEAIS Step 2: Impair Step ===")
    t0 = time.perf_counter()
    noisy_dataset = make_error_maximizing_noise(
        model0, (3, 224, 224), cfg.forget_class, device,
        cfg.batch_size, cfg.noise_repetitions, cfg.lambda_noise,
        cfg.noise_steps, cfg.noise_lr,
    )
    theta_u = impair_step(theta_prime, noisy_dataset, train_ds, retain_idx, train_labels,
                           device, cfg.impair_epochs, cfg.impair_lr, cfg.momentum,
                           cfg.batch_size, cfg.retain_sub_per_class)
    elapsed = time.perf_counter() - t0

    print("\n=== BEAIS (unlearned model) accuracy ===")
    for k, v in eval_loaders.items():
        print(f"  {k}: {evaluate(theta_u, v, device):.2f}%")
    print(f"  unlearning time: {elapsed:.2f}s")


main()
