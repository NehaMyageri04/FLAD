import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml

from vqc_circuit import NUM_QUBITS, NUM_LAYERS, VQC_WEIGHT_SHAPE, _vqc_qnode


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # kernel1(10,1,5,5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # kernel2(20,10,5,5)
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)  # weight(320,10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        batch_size = tensor.size(0)
        tensor = self.pooling(F.relu(self.conv1(tensor)))
        tensor = self.pooling(F.relu(self.conv2(tensor)))
        tensor = tensor.view(batch_size, -1)  # Flatten
        return self.fc(tensor)


class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        stride = 1
        _features = out_features
        if self.in_features != self.out_features:
            if self.out_features / self.in_features == 2.0:
                stride = 2
            else:
                raise ValueError(
                    "The number of output features is at most two times the number of input features!"
                )

        self.conv1 = nn.Conv2d(
            in_features, _features, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            _features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            _features, _features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            _features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.downsample = (
            None
            if self.in_features == self.out_features
            else nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(
                    out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                ),
            )
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsampling layer is used when the number of features in the input and output is different
        if self.in_features != self.out_features:
            identity = self.downsample(x)

        # Summation of residuals
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512), BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)       # output: Tensor (64,512,1,1)
        x = torch.flatten(x, 1)   # output: Tensor (64,512)
        x = self.fc(x)            # output: Tensor (64,10)
        return x


class QuantumByzantineDetector(nn.Module):
    """Quantum Byzantine Detector — wraps a VQC as a ``torch.nn.Module``.

    This class is a drop-in replacement for ``LinearNet`` in the FLAD pipeline.
    It accepts the same high-dimensional weight vector as input and produces the
    same scalar output format, but uses a Variational Quantum Circuit internally.

    Architecture
    ------------
    1. **Feature compression** : Splits the input into ``num_qubits`` equal
       chunks and takes the mean of each chunk, yielding ``num_qubits`` values.
    2. **Normalisation**        : Min-max scales values to [0, π] per sample.
    3. **VQC forward pass**     : AngleEmbedding + StronglyEntanglingLayers.
    4. **Readout**              : ⟨Z⟩ on the last qubit mapped from [-1,1] to
                                  [0,1] and then squared (matching the
                                  ``LinearNet`` output format of ``x**2``).

    Parameters
    ----------
    dimen      : int  Flattened dimension of the input weight vector.
    num_qubits : int  Number of qubits (default 4).
    num_layers : int  Number of variational layers (default 3).
    """

    def __init__(
        self,
        dimen: int,
        num_qubits: int = NUM_QUBITS,
        num_layers: int = NUM_LAYERS,
    ):
        super(QuantumByzantineDetector, self).__init__()
        self.dimen = dimen
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Build a per-instance QNode so that instances are independent
        _dev = qml.device("default.qubit", wires=num_qubits)
        _weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=num_layers, n_wires=num_qubits
        )

        @qml.qnode(_dev, interface="torch", diff_method="parameter-shift")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return qml.expval(qml.PauliZ(num_qubits - 1))

        # TorchLayer registers variational weights as nn.Parameters
        self.qlayer = qml.qnn.TorchLayer(_circuit, {"weights": _weight_shape})

    # ── Feature encoding ──────────────────────────────────────────────────────

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compress *(batch, dimen)* to *(batch, num_qubits)* angles in [0, π]."""
        # Replace NaN/inf before encoding
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        dimen = x.size(1)

        chunk_size = max(1, dimen // self.num_qubits)
        chunks = []
        for i in range(self.num_qubits):
            start = i * chunk_size
            end = (start + chunk_size) if i < self.num_qubits - 1 else dimen
            end = min(end, dimen)
            chunks.append(x[:, start:end].mean(dim=1, keepdim=True))

        features = torch.cat(chunks, dim=1)  # (batch, num_qubits)

        # Replace any NaN from empty chunks, then clamp to valid range
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = torch.clamp(features, -1.0, 1.0)

        # Per-sample min-max normalisation to [0, π]
        f_min = features.min(dim=1, keepdim=True)[0]
        f_max = features.max(dim=1, keepdim=True)[0]
        features = (features - f_min) / (f_max - f_min + 1e-9) * float(np.pi)

        return features

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run each sample through the VQC and return a *(batch, 1)* tensor.

        Quantum circuits execute on CPU regardless of the caller's device; the
        method handles device transfers transparently.
        """
        # Detach from any existing computation graph; quantum simulation manages
        # its own gradients via the parameter-shift rule and does not propagate
        # gradients through the input encoding.
        x = x.detach().cpu()
        x = x.view(x.size(0), -1)
        encoded = self._encode_features(x)  # (batch, num_qubits)

        # Process one sample at a time (quantum circuit is inherently sequential)
        outputs = []
        for i in range(encoded.size(0)):
            out = self.qlayer(encoded[i])  # scalar tensor
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0)          # (batch,)
        outputs = torch.nan_to_num(outputs, nan=0.5)   # replace NaN with neutral value
        outputs = ((1.0 + outputs) / 2.0).view(-1, 1)  # map [-1,1]→[0,1], reshape

        return outputs ** 2  # match LinearNet's x**2 output scaling
