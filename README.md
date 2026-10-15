# BEAIS: Boundary Expanding and Impair Step for Machine Unlearning

Official code for **"Rethinking Image Classification with Machine Unlearning"**, submitted to *Neurocomputing*.

BEAIS removes an entire class from a trained image classifier without retraining it from scratch. It runs in two stages:

1. **Boundary Expanding** — the forget samples are relabelled to a temporary pseudo-class and the model is briefly fine-tuned, pushing their activations into a separate region of the decision space. The extra output neuron is then discarded.
2. **Impair Step** — error-maximising noise is generated against the original model and used, together with a small subset of the retain data, to further displace the forgotten class from the decision boundary.

The result is a model that classifies the forgotten class at 0% accuracy while keeping its accuracy on the remaining classes, at a fraction of the cost of retraining.

---

## Results

### Natural image benchmarks

Accuracy on the retain set (D*r*) and forget set (D*f*), AllCNN backbone.

| Dataset | Method | Train D*r* | Train D*f* | Test D*r* | Test D*f* |
|---|---|---|---|---|---|
| CIFAR-10 | Original | 97.30 | 95.78 | 82.18 | 81.30 |
| CIFAR-10 | Retrain | 96.89 | 0.00 | 83.50 | 0.00 |
| CIFAR-10 | UNSIR | 52.72 | 0.00 | 52.31 | 0.00 |
| CIFAR-10 | Boundary Shrink | 88.58 | 8.14 | 74.34 | 6.40 |
| CIFAR-10 | Boundary Expanding | 91.51 | 10.68 | 77.04 | 7.90 |
| CIFAR-10 | **BEAIS (ours)** | 88.56 | 0.02 | **78.70** | **0.00** |
| SVHN | Original | 88.82 | 90.46 | 87.71 | 89.85 |
| SVHN | Retrain | 92.19 | 0.00 | 89.51 | 0.00 |
| SVHN | UNSIR | 75.40 | 0.79 | 76.57 | 0.80 |
| SVHN | Boundary Shrink | 77.55 | 5.98 | 76.27 | 4.36 |
| SVHN | Boundary Expanding | 78.21 | 10.23 | 76.60 | 7.63 |
| SVHN | **BEAIS (ours)** | **82.80** | 0.85 | **82.74** | **0.52** |

BEAIS is approximately **18× faster** than retraining on CIFAR-10 and **22× faster** on SVHN.

### Comparison with recent state-of-the-art methods

Base models were retrained independently for this comparison; their test retain accuracies agree to within 0.5%.

| Dataset | Method | Train D*r* | Train D*f* | Test D*r* | Test D*f* |
|---|---|---|---|---|---|
| CIFAR-10 | Original | 89.17 | 90.70 | 81.41 | 83.10 |
| CIFAR-10 | SalUn | 90.80 | 0.00 | 82.79 | 0.00 |
| CIFAR-10 | SSD | 89.20 | 0.00 | 82.12 | 0.00 |
| CIFAR-10 | AMUN | 85.90 | 0.00 | 78.71 | 0.00 |
| SVHN | Original | 97.89 | 98.88 | 94.73 | 95.36 |
| SVHN | SalUn | 98.36 | 0.00 | 95.15 | 0.00 |
| SVHN | SSD | 97.96 | 0.07 | 94.87 | 0.00 |
| SVHN | AMUN | 97.29 | 12.55 | 94.43 | 7.51 |

### Document image benchmarks

ResNet-18 pre-trained on ImageNet, used as a fixed feature extractor. Both datasets are used in full, without subsampling.

| Dataset | Method | Train D*r* | Train D*f* | Test D*r* | Test D*f* |
|---|---|---|---|---|---|
| Tobacco3482 | Original | 79.26 | 92.93 | 63.89 | 69.57 |
| Tobacco3482 | **BEAIS** | 73.92 | 1.09 | **67.13** | **0.00** |
| RVL-CDIP | Original | 59.49 | 65.36 | 58.83 | 64.04 |
| RVL-CDIP | **BEAIS** | 60.66 | **0.00** | **60.39** | **0.00** |

---

## Repository structure

```
.
├── beais/
│   ├── run_cifar10_svhn.py      # BEAIS on CIFAR-10 and SVHN (Tables 1-2)
│   ├── run_tobacco3482.py       # BEAIS on Tobacco3482 (Table 6)
│   └── run_rvlcdip.py           # BEAIS on full RVL-CDIP (Table 6)
├── baselines/
│   ├── run_salun.py             # SalUn        (Fan et al., ICLR 2024)
│   ├── run_ssd.py               # SSD          (Foster et al., AAAI 2024)
│   └── run_amun.py              # AMUN         (Ebrahimpour-Boroojeny et al., ICML 2025)
├── figures/
│   └── plot_document_results.py # Regenerates Figure 6
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/i-anasop/mu.git
cd beais-machine-unlearning
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. All experiments were run on a single NVIDIA T4.

---

## Usage

Each script is self-contained and configured through the `cfg` block at the top of its `main()` function.

### BEAIS on CIFAR-10 / SVHN

```bash
python beais/run_cifar10_svhn.py
```

Set `dataset` to `"cifar10"` or `"svhn"`. Both datasets download automatically through `torchvision`.

### BEAIS on document images

```bash
python beais/run_tobacco3482.py     # downloads automatically via Hugging Face
python beais/run_rvlcdip.py         # expects RVL-CDIP on local disk, see below
```

### Baselines

```bash
python baselines/run_salun.py
python baselines/run_ssd.py
python baselines/run_amun.py
```

Each takes the same `dataset` flag. All three load the cached base model produced by the BEAIS runs, so that every method starts from an equivalent trained network.

### Figure

```bash
python figures/plot_document_results.py
```

---

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| CIFAR-10 | `torchvision` | Downloads automatically |
| SVHN | `torchvision` | Downloads automatically; a 40,000-image stratified subset is used to match the scale of CIFAR-10 |
| Tobacco3482 | Hugging Face `maveriq/tobacco3482` | Downloads automatically; 80/20 stratified split |
| RVL-CDIP | [Official release](https://adamharley.com/rvl-cdip/) | ~45 GB. Set `images_root` and the label-file paths in `run_rvlcdip.py` |

RVL-CDIP ships in the original layout, with images spread across scan-ID folders and the class labels held in `train.txt` / `val.txt` / `test.txt`. The loader parses those label files directly and skips the small number of corrupt files present in the archive.

---

## Reproducibility notes

- All runs use seed `42`.
- Class `0` is the forget class throughout.
- For CIFAR-10 and SVHN the backbone is AllCNN trained from scratch for 20 epochs.
- For the document datasets the backbone is an ImageNet-pre-trained ResNet-18 used as a fixed feature extractor, with only the linear classification head trained. Training document classifiers from scratch at this data scale does not converge beyond chance level, which is consistent with the document image classification literature.
- Base model checkpoints are cached, so re-running a script reuses the same starting model rather than retraining it.

---

## Citation

```bibtex
@article{tariq2026rethinking,
  title={Rethinking Image Classification with Machine Unlearning},
  author={Tariq, Javeria and Moetesum, Momina and Anas, Muhammad},
  journal={Neurocomputing},
  year={2026},
  note={Under review}
}
```

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
