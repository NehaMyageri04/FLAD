# QBAD-FL — Quantum Byzantine Anomaly Detection in Federated Learning

QBAD-FL extends [FLAD](https://github.com/NehaMyageri04/FLAD/tree/main/1.FLAD)
by replacing its classical **LinearNet** gradient anomaly detector with a
**Variational Quantum Circuit (VQC)**.  All other components (client training,
attack methods, DBSCAN clustering, FedAvg aggregation) are unchanged, enabling
a direct comparison between classical and quantum Byzantine detection.

---

## Directory Structure

```
6.QBAD-FL/
├── vqc_circuit.py          # Quantum circuit core (PennyLane)
├── Models.py               # Mnist_CNN, ResNet18, QuantumByzantineDetector
├── main.py                 # Main execution script (quantum-enabled)
├── clients.py              # Client training (copied from 1.FLAD)
├── Attack.py               # Attack methods (copied from 1.FLAD)
├── getData.py              # Data loading (copied from 1.FLAD)
├── config.yaml             # Experiment configuration
├── VQC_ARCHITECTURE.md     # Circuit documentation
├── checkpoints/            # Saved models (auto-created)
└── README.md               # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision pennylane scikit-learn numpy
# Optional Qiskit backend
pip install pennylane-qiskit qiskit
```

### 2. Ensure data is available

MNIST data is expected at `../data/MNIST/` (relative to this directory).  
CIFAR-10 will be downloaded automatically on first run.

### 3. Run with default settings (MNIST, MPAF attack)

```bash
cd 6.QBAD-FL
python main.py
```

### 4. Run with custom parameters

```bash
python main.py \
  -data mnist \
  -p 5 \
  -nc 50 \
  -by 10 \
  -ncomm 20 \
  -E 5 \
  -B 64 \
  -lr 0.1 \
  -cen 300
```

---

## Command-Line Arguments

| Argument              | Default | Description                                             |
|-----------------------|---------|---------------------------------------------------------|
| `-data`               | `mnist` | Dataset: `mnist` or `cifar_10`                          |
| `-p`                  | `5`     | Attack pattern (0–6, see below)                         |
| `-nc`                 | `50`    | Total number of clients                                 |
| `-by`                 | `10`    | Number of Byzantine (malicious) clients                 |
| `-ncomm`              | `20`    | Number of communication rounds                          |
| `-E`                  | `5`     | Local training epochs per round                         |
| `-B`                  | `64`    | Local batch size                                        |
| `-lr`                 | `0.1`   | SGD learning rate for the main FL model                 |
| `-cen`                | `300`   | Number of server-side central data samples              |
| `-iid`                | `True`  | IID data distribution (`True`/`False`)                  |
| `-alpha`              | `0.5`   | Weight for cosine vs. length score in DBSCAN clustering |
| `-sp`                 | `./checkpoints` | Checkpoint save path                            |
| `-g`                  | `0`     | GPU id (`0` = first GPU; CPU used if none available)    |

### Attack patterns

| Code | Attack                  |
|------|-------------------------|
| 0    | Gaussian noise          |
| 1    | Sign-flipping           |
| 2    | Zero-gradient           |
| 3    | Backdoor data poisoning |
| 4    | Model replacement       |
| 5    | MPAF                    |
| 6    | AGR-agnostic            |

---

## How It Works

```
Each communication round:

  Server central data
       │
       ▼
  centralTrain()  →  weight snapshots
       │
       ▼
  train_vqc()     →  QuantumByzantineDetector (VQC trained on honest weights)
       │
       ▼
  Clients train locally  →  Upload_Parameters
       │
       ▼
  neural_network_feature_extraction()
       │  VQC scores each client's weights → 2-D feature vector
       ▼
  DBSCAN clustering  →  identify malicious cluster
       │
       ▼
  FedAvg(clean updates)  →  new global model
```

The **VQC replaces LinearNet** as the feature extractor.  Everything else is
identical to FLAD, making results directly comparable.

---

## Quantum Circuit Summary

- **Qubits**: 4  
- **Layers**: 3 × StronglyEntanglingLayers (RY, RZ, CNOT)  
- **Encoding**: Angle embedding of 4 compressed gradient features  
- **Readout**: ⟨Z⟩ expectation on qubit 3, mapped to [0, 1]  
- **Training**: Parameter-shift rule via PennyLane + Adam optimizer  

See `VQC_ARCHITECTURE.md` for a full description and comparison table.

---

## Dependencies

| Package       | Minimum version | Purpose                          |
|---------------|-----------------|----------------------------------|
| `torch`       | 1.9.0           | Main FL model training           |
| `torchvision` | 0.10.0          | CIFAR-10 download & transforms   |
| `pennylane`   | 0.24.0          | Quantum circuit simulation       |
| `scikit-learn`| 0.24.0          | DBSCAN clustering                |
| `numpy`       | 1.19.0          | Numerical operations             |
| `qiskit`      | 0.37.0          | *(optional)* Qiskit backend      |
