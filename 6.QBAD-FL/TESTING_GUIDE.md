# QBAD-FL Testing Guide

Complete guide to running, interpreting, and exporting results from the
QBAD-FL testing and benchmarking suite.

---

## Prerequisites

### 1. Install dependencies

```bash
cd 6.QBAD-FL/
pip install -r requirements_test.txt
```

### 2. Ensure MNIST data is available

The scripts expect MNIST to be in `../data/MNIST/` relative to this directory
(i.e., `<repo-root>/data/MNIST/`).  CIFAR-10 is downloaded automatically.

```
data/
├── MNIST/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
└── CIFAR_10/          # auto-downloaded on first use
```

If the MNIST files are missing, download them from
<http://yann.lecun.com/exdb/mnist/> and place the `.gz` files in `data/MNIST/`.

### 3. All tests run on CPU

No GPU is required.  Quantum circuits are simulated via PennyLane's
`default.qubit` device.

---

## Testing Levels

### Level 1 — Quick Validation (~5 minutes)

Runs a single attack type with the smallest possible configuration.

```bash
python test_qbad_fl.py --quick
# 10 clients, 5 byzantine, 2 rounds, MPAF attack
```

Expected output example:
```
─── Round 1/2 ───
  accuracy=87.30%  detection_rate=100.00%  FPR=0.00%  F1=1.0000  time=42.3s
  Detected malicious: [5, 6, 7, 8, 9]
  Actual  malicious : [5, 6, 7, 8, 9]
...
══════════════════════════════════════════════════════════════════
             QBAD-FL — Attack 5 (Quick Validation)
══════════════════════════════════════════════════════════════════
  accuracy:             mean=0.8730  std=0.0050  final=0.8730
  detection_rate:       mean=1.0000  std=0.0000  final=1.0000
  ✅ PASS (accuracy >= 85%)
```

**Expected accuracy**: 85–90%

---

### Level 2 — Multi-Attack Test (~15 minutes)

Test four representative attack types.

```bash
python test_qbad_fl.py --attacks 0 1 2 5
# Gaussian, Sign-Flip, Zero-Gradient, MPAF
# 20 clients, 5 byzantine, 3 rounds each
```

Or with more control:
```bash
python test_qbad_fl.py \
  --dataset mnist \
  --rounds 3 \
  --clients 20 \
  --byzantine 5 \
  --attacks 0 1 2 5 \
  --output results/level2_test.json
```

**Expected accuracy**: 80–92% depending on attack type.

---

### Level 3 — QBAD-FL vs FLAD Comparison (~30 minutes)

Directly compares quantum detection (VQC) with classical detection (LinearNet).

```bash
# Compare for 4 attack types
python benchmark_qbad_vs_flad.py --attacks 0 1 2 5 --rounds 5

# Full comparison (all 7 attacks)
python benchmark_qbad_vs_flad.py --full --rounds 5
```

The output includes a comparison table:

```
Comparison Table — Attack: MPAF
──────────────────────────────────────────────────────────────────────
Metric                      FLAD      QBAD-FL   Difference  Advantage
──────────────────────────────────────────────────────────────────────
accuracy                  0.8900       0.9100      +0.0200  QBAD-FL
detection_rate            0.8000       1.0000      +0.2000  QBAD-FL
false_positive_rate       0.0200       0.0000      -0.0200  QBAD-FL
precision                 0.9000       1.0000      +0.1000  QBAD-FL
recall                    0.8000       1.0000      +0.2000  QBAD-FL
f1                        0.8462       1.0000      +0.1538  QBAD-FL
──────────────────────────────────────────────────────────────────────
```

CSV and JSON outputs are saved to `results/`.

---

### Level 4 — Full Evaluation (~2 hours on CPU)

Production-ready evaluation for all 7 attack types, IID and non-IID.

```bash
# Full MNIST evaluation with plots
python run_full_experiment.py --dataset mnist --mode full --output results/

# CIFAR-10 full run
python run_full_experiment.py --dataset cifar_10 --mode full --output results/

# Faster variant (5 rounds instead of 20)
python run_full_experiment.py --mode full --rounds 5 --output results/
```

---

## Output Files

All scripts write outputs to the `results/` directory (configurable with
`--output`).

| File | Description |
|------|-------------|
| `results/accuracy_results.csv` | Round-by-round metrics for every experiment |
| `results/metrics.json` | Complete structured results with all per-round data |
| `results/benchmark_comparison.csv` | FLAD vs QBAD-FL summary table (benchmark script) |
| `results/benchmark_results.json` | Full benchmark data |
| `results/plots/accuracy_over_rounds.png` | Accuracy curves per attack/distribution |
| `results/plots/detection_rate_per_attack.png` | Bar chart: DR & accuracy per attack |

---

## Key Metrics Explained

| Metric | Formula | Ideal | Description |
|--------|---------|-------|-------------|
| **Accuracy** | correct / total | 1.0 | Model classification accuracy on the test set |
| **Detection Rate** | TP / total Byzantine | 1.0 | Fraction of malicious clients correctly identified |
| **False Positive Rate** | FP / total honest | 0.0 | Fraction of honest clients incorrectly flagged |
| **Precision** | TP / (TP + FP) | 1.0 | Of flagged clients, fraction that are truly malicious |
| **Recall** | TP / (TP + FN) | 1.0 | Same as Detection Rate |
| **F1** | 2·P·R/(P+R) | 1.0 | Harmonic mean of precision and recall |

### Success Criteria

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 85% → ✅ PASS |
| Detection Rate | ≥ 80% |
| False Positive Rate | ≤ 10% |
| F1 | ≥ 0.80 |

---

## Attack Patterns

| Code | Name | Type | Notes |
|------|------|------|-------|
| 0 | Gaussian | Untargeted | Random Gaussian noise on gradients |
| 1 | Sign-Flip | Untargeted | Negates and scales gradient directions |
| 2 | Zero-Gradient | Untargeted | Cancels out honest updates |
| 3 | Backdoor | Targeted | Data poisoning with pixel trigger |
| 4 | Model-Replacement | Targeted | Gradient expansion attack |
| 5 | MPAF | Untargeted | Large random perturbation |
| 6 | AGR-Agnostic | Untargeted | Adaptive attack exploiting aggregation |

---

## Expected Accuracy Ranges

Based on typical results with default settings (50 clients, 10 Byzantine, 20 rounds, MNIST):

| Attack | Expected Accuracy |
|--------|-----------------|
| Gaussian | 88–93% |
| Sign-Flip | 86–91% |
| Zero-Gradient | 85–90% |
| Backdoor | 82–88% |
| Model-Replacement | 80–87% |
| MPAF | 88–92% |
| AGR-Agnostic | 78–85% |

*Note: Actual results depend on random seed, data distribution, and number of rounds.*

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pennylane'`
```bash
pip install pennylane>=0.24.0
```

### `ModuleNotFoundError: No module named 'sklearn'`
```bash
pip install scikit-learn>=0.22.1
```

### `FileNotFoundError: ../data/MNIST/train-images-idx3-ubyte.gz`
Download MNIST data files to `<repo-root>/data/MNIST/`.

### `ValueError: Invalid magic number` (MNIST)
The `.gz` files may be corrupted. Re-download them.

### Slow VQC training
VQC training is CPU-bound and slower than classical LinearNet.  Use `--rounds 2`
for quick tests.  Each VQC forward pass takes ~0.5–2s depending on hardware.

### DBSCAN finds only one cluster
This can happen with very few rounds or specific attack combinations.  Increase
`--clients` or `--rounds`, or try a different `--byzantine` ratio.

### Memory errors with CIFAR-10
CIFAR-10 uses ResNet18 which is larger.  Reduce `--clients` to 20 if memory is
limited.

---

## Running from a Different Directory

All scripts use absolute paths internally, but must be run from within
`6.QBAD-FL/` for local imports to work:

```bash
cd /path/to/FLAD/6.QBAD-FL
python test_qbad_fl.py --quick
```

---

## Complete Usage Reference

### `test_qbad_fl.py`

```
usage: test_qbad_fl.py [-h] [--dataset {mnist,cifar_10}] [--rounds ROUNDS]
                        [--clients CLIENTS] [--byzantine BYZANTINE]
                        [--attacks ATTACKS [ATTACKS ...]] [--quick]
                        [--output OUTPUT]

options:
  --quick              10 clients, 5 byzantine, 2 rounds, MPAF attack
  --attacks 0 1 2 5    Attack patterns to test
  --rounds N           Communication rounds (default: 3)
  --clients N          Total number of clients (default: 20)
  --byzantine N        Byzantine clients (default: 5)
  --output PATH        Save JSON results to PATH
```

### `benchmark_qbad_vs_flad.py`

```
usage: benchmark_qbad_vs_flad.py [-h] [--dataset {mnist,cifar_10}]
                                   [--rounds ROUNDS] [--clients CLIENTS]
                                   [--byzantine BYZANTINE]
                                   [--attacks ATTACKS [ATTACKS ...]]
                                   [--full] [--output OUTPUT]

options:
  --full               Test all 7 attack types
  --attacks 0 1 5      Specific attack patterns
  --output DIR         Output directory (default: results)
```

### `run_full_experiment.py`

```
usage: run_full_experiment.py [-h] [--dataset {mnist,cifar_10}]
                               [--mode {full,single}]
                               [--attacks ATTACKS [ATTACKS ...]]
                               [--rounds ROUNDS] [--clients CLIENTS]
                               [--byzantine BYZANTINE] [--iid-only]
                               [--non-iid-only] [--output OUTPUT]
                               [--no-plots]

options:
  --mode full          All 7 attacks (default)
  --mode single        Use --attacks to specify
  --iid-only           Skip non-IID experiments
  --non-iid-only       Skip IID experiments
  --no-plots           Disable matplotlib plots
  --output DIR         Output directory (default: results)
```
