# VQC Architecture — QBAD-FL

## Overview

QBAD-FL replaces the classical **LinearNet** gradient anomaly detector used in
FLAD with a **Variational Quantum Circuit (VQC)**.  All other components of the
FLAD pipeline — client training, attack generation, DBSCAN clustering, and
FedAvg aggregation — remain unchanged.

---

## Circuit Diagram (ASCII)

```
Input gradient vector  (dimen,)
         │
         ▼
  Feature compression              chunk means → 4 values
         │
         ▼
  Min-max normalisation            values ∈ [0, π]
         │
         ▼
╔════════════════════════════════════════╗
║  Quantum Circuit  (4 qubits)           ║
║                                        ║
║  q0 ─H─[Ry(θ0)]─[Rz(φ0)]─╮─────────  ║
║  q1 ─H─[Ry(θ1)]─[Rz(φ1)]─⊕─╮──────  ║
║  q2 ─H─[Ry(θ2)]─[Rz(φ2)]───⊕─╮────  ║
║  q3 ─H─[Ry(θ3)]─[Rz(φ3)]─────⊕────  ║
║                                        ║
║  (× 3 layers of the above pattern)     ║
║                                        ║
║  Measure ⟨Z⟩ on q3                    ║
╚════════════════════════════════════════╝
         │
         ▼
  (1 + ⟨Z⟩) / 2   →   [0, 1]    (Byzantine probability)
         │
         ▼
  output ** 2                    (match LinearNet scaling)
```

---

## Specifications

| Property            | Value                                         |
|---------------------|-----------------------------------------------|
| Number of qubits    | 4                                             |
| Number of layers    | 3                                             |
| Encoding method     | Angle embedding (`qml.AngleEmbedding`)        |
| Variational ansatz  | `qml.StronglyEntanglingLayers` (RY, RZ, CNOT) |
| Readout             | Expectation value ⟨Z⟩ on the output qubit    |
| Differentiability   | Parameter-shift rule                          |
| Optimizer           | Adam (lr = 0.001)                             |
| Training epochs     | 20 per communication round                    |
| Quantum backend     | PennyLane `default.qubit` (CPU simulator)     |

---

## Feature Encoding

The gradient weight vectors extracted from client model updates can be
thousands of dimensions long (e.g., 3 200 for the fc layer of Mnist_CNN).
To embed them into a 4-qubit circuit via angle embedding, each vector is
compressed to 4 values:

1. **Chunking** — Split the flattened vector into 4 equal-sized segments.
2. **Aggregation** — Compute the mean of each segment (one scalar per qubit).
3. **Normalisation** — Min-max scale the 4 values to `[0, π]` per sample.

These 4 angles are fed to `qml.AngleEmbedding` which applies `Ry(angle_i)` on
qubit `i`.

---

## Variational Ansatz

Each layer of `StronglyEntanglingLayers` applies:
- `Ry(θ)` and `Rz(φ)` rotations on every qubit (trainable parameters).
- CNOT entanglement gates connecting adjacent qubits.

With 3 layers and 4 qubits, the weight tensor has shape `(3, 4, 3)` — that is
`num_layers × num_wires × 3` rotations per wire — giving **36** trainable
parameters in total (see
`qml.StronglyEntanglingLayers.shape(n_layers=3, n_wires=4)`).

---

## Training Method

The VQC is trained using the **parameter-shift rule**, which provides exact
gradients for quantum gate parameters without finite-difference approximations.
PennyLane's `diff_method="parameter-shift"` handles this automatically when the
circuit is used inside a `TorchLayer`.

Training signal: MSE against a target label of **1.0** (same as LinearNet),
meaning the detector is trained to output values close to 1 for "honest"
(server central) weight updates.

---

## Comparison with Classical LinearNet

| Aspect              | LinearNet (FLAD)                        | VQC (QBAD-FL)                            |
|---------------------|-----------------------------------------|------------------------------------------|
| Model type          | 3-layer MLP                             | Variational Quantum Circuit              |
| Parameters          | O(dimen²) classical weights             | 36 quantum rotation angles               |
| Feature space       | Full-dimensional                        | Compressed to 4 qubit angles             |
| Activation          | tanh → sigmoid → x²                    | Quantum interference → ⟨Z⟩ → (·)²       |
| Training            | Backpropagation                         | Parameter-shift rule                     |
| Hardware            | CPU / GPU                               | Quantum simulator (CPU) or QPU           |
| Adaptive robustness | Susceptible to adaptive attacks         | Potential quantum advantage hypothesis   |

---

## Design Rationale

1. **4 qubits** — Sufficient to capture variance in two compressed gradient
   features (conv1 and fc) while keeping simulation tractable on CPU.

2. **3 layers** — Provides enough circuit depth for expressive entanglement
   without excessive simulation overhead.

3. **Angle embedding** — A natural choice for continuous real-valued features;
   preserves relative magnitude information.

4. **Parameter-shift differentiation** — Hardware-compatible gradient method,
   allowing the same circuit to run on real quantum hardware in future work.

5. **Same detection pipeline** — Preserving DBSCAN clustering and cosine/length
   scoring from FLAD ensures a fair apples-to-apples comparison where only the
   feature extractor changes.

---

## References

- Cerezo et al., "Variational Quantum Algorithms" (2021)
- PennyLane documentation: https://pennylane.ai/
- Original FLAD paper: IEEE TDSC 2025
