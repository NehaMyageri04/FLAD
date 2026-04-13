"""
vqc_circuit.py — Variational Quantum Circuit for QBAD-FL

Implements a 4-qubit VQC using PennyLane that serves as a drop-in quantum
replacement for the classical LinearNet detector in FLAD.

Architecture
------------
- Encoding  : AngleEmbedding of 4 compressed gradient features onto 4 qubits
- Ansatz    : 3 StronglyEntanglingLayers (RY + RZ rotations + CNOT entanglement)
- Readout   : Expectation value <Z> on the final qubit → mapped to [0, 1]
- Training  : Parameter-shift rule via PennyLane's PyTorch interface
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# ── Global circuit constants ──────────────────────────────────────────────────
NUM_QUBITS = 4
NUM_LAYERS = 3

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
        """Compress a high-dimensional gradient vector to ``num_qubits`` angles.

        The vector is split into ``num_qubits`` equal-sized chunks; the mean of
        each chunk becomes one angle.  Values are then min-max normalised to
        [0, π] so they are valid rotation angles for :func:`qml.AngleEmbedding`.

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

        # Replace NaN/inf before chunking
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        n = len(arr)
        features = np.zeros(self.num_qubits, dtype=np.float32)
        for i in range(self.num_qubits):
            start = i * (n // self.num_qubits)
            end = (i + 1) * (n // self.num_qubits) if i < self.num_qubits - 1 else n
            features[i] = arr[start:end].mean() if end > start else 0.0

        # Replace any NaN that may have arisen from empty slices
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp to valid range before normalisation
        features = np.clip(features, -1.0, 1.0)

        # Normalise to [0, π]
        f_min, f_max = features.min(), features.max()
        if f_max - f_min > 1e-9:
            features = (features - f_min) / (f_max - f_min) * np.pi
        else:
            features = np.zeros(self.num_qubits, dtype=np.float32)

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
