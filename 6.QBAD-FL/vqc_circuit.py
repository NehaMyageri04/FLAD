"""
vqc_circuit.py — Variational Quantum Circuit for QBAD-FL

Implements a 6-qubit VQC using PennyLane that serves as a drop-in quantum
replacement for the classical LinearNet detector in FLAD.

Architecture
------------
- Encoding  : AngleEmbedding of 6 statistical features onto 6 qubits
              Features: tanh(mean), tanh(log-std), tanh(norm/sqrt(d)),
                        tanh(log-max-abs), tanh(skewness/3), tanh(log-norm)
- Ansatz    : 5 StronglyEntanglingLayers (RY + RZ rotations + CNOT entanglement)
- Readout   : Expectation value <Z> on the final qubit → mapped to [0, 1]
- Training  : Parameter-shift rule via PennyLane's PyTorch interface
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# ── Global circuit constants ──────────────────────────────────────────────────
NUM_QUBITS = 6
NUM_LAYERS = 5

# Shared quantum device (CPU simulator, thread-safe at module level)
_qml_device = qml.device("default.qubit", wires=NUM_QUBITS)


@qml.qnode(_qml_device, interface="torch", diff_method="parameter-shift")
def _vqc_qnode(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """PennyLane QNode that encodes *inputs* and applies *weights*.

    Parameters
    ----------
    inputs  : 1-D tensor of shape (NUM_QUBITS,) — one encoded angle per qubit.
    weights : tensor of shape returned by
              ``qml.StronglyEntanglingLayers.shape(NUM_LAYERS, NUM_QUBITS)``

    Returns
    -------
    Scalar expectation value ⟨Z⟩ on the last qubit in [-1, 1].
    """
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return qml.expval(qml.PauliZ(NUM_QUBITS - 1))


# Shape of the variational weight tensor
VQC_WEIGHT_SHAPE = qml.StronglyEntanglingLayers.shape(
    n_layers=NUM_LAYERS, n_wires=NUM_QUBITS
)


class VQCCircuit:
    """Stateless helper that exposes the VQC interface described in the spec.

    Most users should prefer :class:`QuantumByzantineDetector` in ``Models.py``
    which wraps this helper as a ``torch.nn.Module`` compatible with the FLAD
    training loop.  This class is provided for direct quantum-circuit access and
    for documentation/testing purposes.

    Parameters
    ----------
    num_qubits : int   Number of qubits (default 4).
    num_layers : int   Number of variational layers (default 3).
    """

    def __init__(self, num_qubits: int = NUM_QUBITS, num_layers: int = NUM_LAYERS):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=num_layers, n_wires=num_qubits
        )
        # Randomly initialised parameters managed as plain numpy arrays.
        # This class is intended for direct circuit inspection and testing;
        # it does NOT use PyTorch autograd.  For training with PyTorch
        # optimizers use QuantumByzantineDetector (Models.py) instead, which
        # wraps the circuit as a torch.nn.Module with tracked nn.Parameters.
        self._params = np.random.uniform(
            0, 2 * np.pi, self._weight_shape
        ).astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def encode_features(self, weight_vector) -> np.ndarray:
        """Extract statistical features from a gradient vector for quantum encoding.

        Computes 6 scale-aware statistics from the weight vector and maps each
        through tanh so the result lies in (-1, 1), then rescales to [0, π]
        for use as AngleEmbedding rotation angles.

        The 6 features are:
          0 - tanh(mean)                  → captures sign/bias attacks
          1 - tanh(log(1 + std))          → captures variance attacks (Gaussian, MPAF)
          2 - tanh(norm / sqrt(d))        → captures scale attacks (MPAF)
          3 - tanh(log(1 + max_abs))      → captures outlier magnitude
          4 - tanh(skewness / 3)          → captures asymmetric distributions
          5 - tanh(log(1 + norm))         → captures absolute scale (MPAF)

        Parameters
        ----------
        weight_vector : array-like or torch.Tensor of any shape.

        Returns
        -------
        np.ndarray of shape (num_qubits,) with values in [0, π].
        """
        if isinstance(weight_vector, torch.Tensor):
            arr = weight_vector.detach().cpu().numpy().flatten()
        else:
            arr = np.array(weight_vector, dtype=np.float32).flatten()

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

        d = len(arr)
        mean_val = arr.mean()
        std_val = float(arr.std()) + 1e-9
        norm_val = float(np.linalg.norm(arr))
        max_abs = float(np.abs(arr).max()) if d > 0 else 0.0

        # Skewness: E[(x - mean)^3] / std^3
        skewness = float(np.mean(((arr - mean_val) / std_val) ** 3))

        features = np.array([
            np.tanh(mean_val),                           # feature 0
            np.tanh(np.log1p(std_val)),                  # feature 1
            np.tanh(norm_val / (d ** 0.5 + 1e-9)),      # feature 2
            np.tanh(np.log1p(max_abs)),                  # feature 3
            np.tanh(skewness / 3.0),                     # feature 4
            np.tanh(np.log1p(norm_val)),                 # feature 5
        ], dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Map from [-1, 1] to [0, π]
        features = (features + 1.0) / 2.0 * float(np.pi)
        features = np.clip(features, 0.0, float(np.pi))

        return features

    def forward(self, inputs, params=None) -> float:
        """Run the quantum circuit and return a Byzantine probability in [0, 1].

        Parameters
        ----------
        inputs : array-like of shape (num_qubits,) — pre-encoded angles.
        params : optional weight array; uses stored parameters if None.

        Returns
        -------
        float in [0, 1] — probability that the update is Byzantine.
        """
        if params is None:
            params = self._params

        inputs_arr = np.nan_to_num(np.asarray(inputs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        inputs_t = torch.tensor(inputs_arr, dtype=torch.float32)
        weights_t = torch.tensor(params, dtype=torch.float32)

        exp_val = _vqc_qnode(inputs_t, weights_t)
        result = float((1.0 + exp_val) / 2.0)
        if np.isnan(result) or np.isinf(result):
            result = 0.5
        return result

    def get_parameters(self) -> np.ndarray:
        """Return a copy of the current variational parameters."""
        return self._params.copy()

    def set_parameters(self, params) -> None:
        """Overwrite the variational parameters.

        Parameters
        ----------
        params : array-like matching ``VQC_WEIGHT_SHAPE``.
        """
        self._params = np.array(params, dtype=np.float32).reshape(self._weight_shape)
