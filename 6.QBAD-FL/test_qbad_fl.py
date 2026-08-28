"""
test_qbad_fl.py — Validation test for QBAD-FL.

The experiment evaluates the QBAD-FL Byzantine detection pipeline using:
    1. Label-independent norm filtering
    2. Fixed VQC feature extraction
    3. Isolation Forest anomaly detection
    4. Label-independent sign-flip detection
    5. Ensemble detection
    6. Label-independent clipping
    7. Federated averaging

IMPORTANT EXPERIMENTAL RULE
---------------------------
Ground-truth Byzantine identities are used ONLY for:
    - constructing simulated attacks
    - post-hoc evaluation/debug output

They are NOT used for:
    - norm threshold calculation
    - VQC feature extraction
    - Isolation Forest
    - sign-flip detection
    - ensemble decisions
    - gradient clipping
    - federated aggregation

The test set is used ONLY for post-round evaluation.

There is intentionally NO checkpoint recovery based on test accuracy.
This prevents the evaluation set from influencing future FL rounds.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.ensemble import IsolationForest


# ── Import setup ──────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


from Models import Mnist_CNN, ResNet18, QuantumByzantineDetector
from clients import ClientsGroup
import Attack

from metrics import (
    calculate_detection_rate,
    calculate_false_positive_rate,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    aggregate_round_metrics,
    generate_report,
    save_results_json,
    ATTACK_NAMES,
)


# ── Detector constants ────────────────────────────────────────────────────────

# Fixed Isolation Forest contamination assumption.
# IMPORTANT:
# This is NOT dynamically obtained from ground-truth labels at runtime.
_IFOREST_CONTAMINATION = 0.25

# Fixed number of Isolation Forest trees.
_IFOREST_N_ESTIMATORS = 100

# Midpoint used if a VQC feature becomes NaN.
_FEATURE_NAN_FALLBACK = 0.5

# Sign-flip decision threshold.
# A client whose direction has cosine similarity below -0.5
# with the robust current-round reference is flagged.
_SIGN_FLIP_COSINE_THRESHOLD = -0.5

# Norm threshold multiplier.
_NORM_OUTLIER_MULTIPLIER = 10


# ── VQC feature extraction ────────────────────────────────────────────────────

def feature_extraction_model(Central_par, cfg, dev):
    """
    Initialize two separate fixed QuantumByzantineDetectors.

    One detector processes conv1 weights.
    One detector processes fc weights.

    The VQC parameters are fixed and are NOT trained using
    Byzantine labels or test-set performance.
    """

    if cfg["data_name"] == "mnist":

        detector_conv1 = QuantumByzantineDetector(
            dimen=10 * 1 * 5 * 5,
            num_qubits=8,
            num_layers=5,
        )

        detector_fc = QuantumByzantineDetector(
            dimen=10 * 320,
            num_qubits=8,
            num_layers=5,
        )

    else:

        detector_conv1 = QuantumByzantineDetector(
            dimen=64 * 3 * 3 * 3,
            num_qubits=8,
            num_layers=5,
        )

        detector_fc = QuantumByzantineDetector(
            dimen=10 * 512,
            num_qubits=8,
            num_layers=5,
        )

    if dev.type != "cpu":
        detector_conv1 = detector_conv1.to(dev)
        detector_fc = detector_fc.to(dev)

    return detector_conv1, detector_fc


# ── Sign-flip detector ────────────────────────────────────────────────────────

def detect_sign_flip_attacks(Upload_Parameters, nc):
    """
    Detect sign-flip anomalies using ONLY the current submitted updates.

    No:
        - honest-client identities
        - Byzantine-client identities
        - Byzantine labels
        - trusted honest history
        - known Byzantine count

    are used by the detector.

    The reference direction is calculated from the current population
    using a coordinate-wise median of normalized client directions.

    Parameters
    ----------
    Upload_Parameters : list
        Submitted client model updates.

    nc : int
        Total number of clients.

    Returns
    -------
    list
        Indices detected as sign-flip anomalies.
    """

    if not Upload_Parameters:
        return []

    directions = []

    # Normalize every submitted update.
    for update in Upload_Parameters:

        flat = torch.cat(
            [
                update[k].flatten()
                for k in sorted(update.keys())
            ]
        ).detach()

        norm = torch.norm(flat).item()

        if norm <= 1e-12:

            directions.append(
                np.zeros_like(
                    flat.cpu().numpy(),
                    dtype=np.float64,
                )
            )

        else:

            directions.append(
                (flat / norm)
                .cpu()
                .numpy()
                .astype(np.float64)
            )

    directions = np.stack(directions, axis=0)

    # Coordinate-wise median over the COMPLETE submitted population.
    #
    # No honest/Byzantine identities are used.
    reference = np.median(
        directions,
        axis=0,
    )

    reference_norm = float(
        np.linalg.norm(reference)
    )

    if reference_norm <= 1e-12:
        return []

    reference = reference / reference_norm

    detected = []

    # nc is ONLY the total-client loop bound.
    for c in range(
        min(nc, len(directions))
    ):

        cosine_sim = float(
            np.dot(
                reference,
                directions[c],
            )
        )

        if cosine_sim < _SIGN_FLIP_COSINE_THRESHOLD:

            detected.append(c)

            print(
                "  [Sign-Flip Detection] "
                "Client {} cosine_sim={:.4f} → FLAGGED".format(
                    c,
                    cosine_sim,
                )
            )

    if detected:

        print(
            "  [Sign-Flip Detection] "
            "Detected {} flip anomalies: {}".format(
                len(detected),
                detected,
            )
        )

    return detected


# ── VQC + Isolation Forest ────────────────────────────────────────────────────

def vqc_feature_extraction(
    Upload_Parameters,
    detector_conv1,
    detector_fc,
    cfg,
    dev,
):
    """
    Extract fixed VQC features from submitted updates and perform
    Isolation Forest anomaly detection.

    Sign-flip detection is then performed using the current submitted
    population only.
    """

    nc = cfg["num_of_clients"]

    # ── Extract relevant model tensors ────────────────────────────────────────

    if cfg["data_name"] == "mnist":

        k1 = torch.zeros(
            nc,
            10,
            1,
            5,
            5,
        ).to(dev)

        w3 = torch.zeros(
            nc,
            10,
            320,
        ).to(dev)

    else:

        k1 = torch.zeros(
            nc,
            64,
            3,
            3,
            3,
        ).to(dev)

        w3 = torch.zeros(
            nc,
            10,
            512,
        ).to(dev)

    for i, W in enumerate(Upload_Parameters):

        if cfg["data_name"] == "mnist":

            k1[i] = W["conv1.weight"].data
            w3[i] = W["fc.weight"].data

        else:

            k1[i] = W["module.conv1.weight"].data
            w3[i] = W["module.fc.weight"].data

    # ── Fixed VQC feature extraction ──────────────────────────────────────────

    print(
        "  [VQC] Extracting quantum features "
        "(27-D per detector, 54-D total)..."
    )

    with torch.no_grad():

        features_conv1 = (
            detector_conv1(
                k1.view(nc, -1)
            )
            .cpu()
            .numpy()
        )

        features_fc = (
            detector_fc(
                w3.view(nc, -1)
            )
            .cpu()
            .numpy()
        )

    expected_conv1_dim = (
        detector_conv1.num_qubits * 3 + 3
    )

    expected_fc_dim = (
        detector_fc.num_qubits * 3 + 3
    )

    if features_conv1.shape != (
        nc,
        expected_conv1_dim,
    ):

        raise ValueError(
            "Unexpected conv1 feature shape: {} "
            "(expected ({}, {}))".format(
                features_conv1.shape,
                nc,
                expected_conv1_dim,
            )
        )

    if features_fc.shape != (
        nc,
        expected_fc_dim,
    ):

        raise ValueError(
            "Unexpected fc feature shape: {} "
            "(expected ({}, {}))".format(
                features_fc.shape,
                nc,
                expected_fc_dim,
            )
        )

    feature = np.concatenate(
        [
            features_conv1,
            features_fc,
        ],
        axis=1,
    )

    # ── NaN protection ────────────────────────────────────────────────────────

    if np.isnan(feature).any():

        print(
            "  [Warning] NaN in features, "
            "replacing with column means"
        )

        col_mean = np.nan_to_num(
            np.nanmean(
                feature,
                axis=0,
            ),
            nan=_FEATURE_NAN_FALLBACK,
        )

        feature = np.where(
            np.isnan(feature),
            col_mean,
            feature,
        )

    print(
        "  [VQC] Feature shape: {}".format(
            feature.shape
        )
    )

    print(
        "    Feature ranges: "
        "min={:.4f}  max={:.4f}  mean={:.4f}".format(
            feature.min(),
            feature.max(),
            feature.mean(),
        )
    )

    # ── Isolation Forest ──────────────────────────────────────────────────────

    malicious = []

    try:

        clf = IsolationForest(
            contamination=_IFOREST_CONTAMINATION,
            random_state=42,
            n_estimators=_IFOREST_N_ESTIMATORS,
        )

        predictions = clf.fit_predict(
            feature
        )

        malicious = [
            c
            for c in range(nc)
            if predictions[c] == -1
        ]

        print(
            "  [Isolation Forest] Predictions: {}".format(
                predictions.tolist()
            )
        )

        print(
            "  [VQC Detection] Detected {} anomalies: {}".format(
                len(malicious),
                malicious,
            )
        )

    except Exception as e:

        print(
            "  [Warning] Isolation Forest failed: {}".format(
                e
            )
        )

        malicious = []

    # ── Sign-flip detector ────────────────────────────────────────────────────

    flip_detected = detect_sign_flip_attacks(
        Upload_Parameters,
        nc,
    )

    # ── Ensemble ──────────────────────────────────────────────────────────────

    vqc_detected = list(malicious)

    malicious = list(
        set(vqc_detected)
        |
        set(flip_detected)
    )

    if flip_detected:

        print(
            "  [Ensemble] Combined "
            "VQC={} + SignFlip={} → Final={}".format(
                vqc_detected,
                flip_detected,
                malicious,
            )
        )

    return malicious


# ── Norm filtering ────────────────────────────────────────────────────────────

def detect_norm_outliers(
    Upload_Parameters,
    cfg,
    dev,
):
    """
    Detect magnitude outliers using the complete submitted population.

    The threshold is NOT based on:
        - honest clients
        - Byzantine clients
        - Byzantine count
        - attack labels
        - trusted history
    """

    norms = [

        torch.norm(
            torch.cat(
                [
                    u[k].flatten()
                    for k in sorted(u.keys())
                ]
            )
        ).item()

        for u in Upload_Parameters
    ]

    if not norms:
        return []

    # Label-independent population reference.
    median_norm = float(
        np.median(norms)
    )

    threshold = (
        median_norm
        *
        _NORM_OUTLIER_MULTIPLIER
    )

    norm_rejected = [
        i
        for i, norm in enumerate(norms)
        if norm > threshold
    ]

    if norm_rejected:

        print(
            "  [Norm Filter] "
            "Population median norm: {:.4f}, "
            "Threshold: {:.4f}".format(
                median_norm,
                threshold,
            )
        )

    return norm_rejected


# ── Label-independent clipping ────────────────────────────────────────────────

def clip_gradient_norms(
    Upload_Parameters,
    cfg,
):
    """
    Clip oversized updates using a threshold calculated from
    the complete submitted population.

    No ground-truth client information is used.
    """

    norms = [

        torch.norm(
            torch.cat(
                [
                    u[k].flatten()
                    for k in sorted(u.keys())
                ]
            )
        ).item()

        for u in Upload_Parameters
    ]

    if not norms:
        return

    median_norm = float(
        np.median(norms)
    )

    max_allowed = (
        median_norm
        *
        _NORM_OUTLIER_MULTIPLIER
    )

    clipped_count = 0

    for i, update in enumerate(
        Upload_Parameters
    ):

        if norms[i] > max_allowed:

            scale = (
                max_allowed
                /
                norms[i]
            )

            for k in update:

                update[k] = (
                    update[k]
                    *
                    scale
                )

            clipped_count += 1

    if clipped_count > 0:

        print(
            "  [Gradient Clipping] "
            "Clipped {} updates to max norm {:.4f}".format(
                clipped_count,
                max_allowed,
            )
        )


# ── Federated averaging ───────────────────────────────────────────────────────

def fed_avg(
    Upload_Parameters,
    malicious,
):
    """
    Aggregate only updates that were identified by the detector.

    The malicious list comes exclusively from detector outputs.
    """

    params = list(
        Upload_Parameters
    )

    for j, idx in enumerate(
        sorted(malicious)
    ):

        del params[
            idx - j
        ]

    if not params:
        raise ValueError(
            "All client updates were rejected."
        )

    total = len(params)

    agg = None

    for p in params:

        if agg is None:

            agg = {
                k: v.clone()
                for k, v in p.items()
            }

        else:

            for k in p:

                agg[k] += p[k]

    for k in agg:

        agg[k] /= total

    return agg


# ── Attack construction ───────────────────────────────────────────────────────

def collect_attack_updates(
    byzantine_clients,
    myClients,
    pattern,
    honest_all_weight,
    num_clients,
    epoch,
    batchsize,
    net,
    loss_func,
    opti,
    global_parameters,
    data_name="mnist",
):
    """
    Construct simulated Byzantine updates.

    Ground-truth Byzantine identities are required here because this
    function is constructing the simulated attack population.

    This information does NOT enter the detector.
    """

    uploads = []

    for client in byzantine_clients:

        lp = {}

        if pattern == 0:

            for key in honest_all_weight:

                lp[key] = Attack.Gaussian_attack(
                    honest_all_weight[key]
                )

        elif pattern == 1:

            for key in honest_all_weight:

                lp[key] = Attack.Sign_flipping_attack(
                    honest_all_weight[key]
                )

        elif pattern == 2:

            byz = sum(
                1
                for _ in byzantine_clients
            )

            for key in honest_all_weight:

                lp[key] = Attack.ZeroGradient_attack(
                    honest_all_weight[key],
                    byz,
                )

        elif pattern == 3:

            pc = Attack.backdoor_poisoning_data(
                myClients.clients_set[client],
                data_name,
            )

            lp = pc.localTrain(
                epoch,
                batchsize,
                net,
                loss_func,
                opti,
                global_parameters,
            )

        elif pattern == 4:

            pc = Attack.model_replacement_attack_data(
                myClients.clients_set[client],
                data_name,
            )

            lp = pc.localTrain(
                epoch,
                batchsize,
                net,
                loss_func,
                opti,
                global_parameters,
            )

            for key in lp:

                lp[key] = (
                    lp[key]
                    *
                    num_clients
                )

        elif pattern == 5:

            for key in honest_all_weight:

                lp[key] = Attack.MPAF(
                    honest_all_weight[key]
                )

        elif pattern == 6:

            for key in honest_all_weight:

                lp[key] = Attack.AGR_agnostic(
                    honest_all_weight[key]
                )

        uploads.append(lp)

    return uploads


# ── Server-side structural checkpoint recovery ────────────────────────────────

# Recovery is deliberately conservative: it is intended to catch catastrophic
# structural corruption, not ordinary accuracy fluctuations.
_RECOVERY_ENABLED = True
_RECOVERY_PARAM_NORM_RATIO = 10.0
_RECOVERY_DRIFT_RATIO = 5.0
_RECOVERY_EPS = 1e-12


def _clone_state_dict(state_dict):
    """Deep-clone a model state_dict for an immutable server checkpoint."""
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _state_dict_parameter_norm(state_dict):
    """Return the global L2 norm of all tensor parameters/buffers."""
    total_sq = 0.0
    for value in state_dict.values():
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().float()
        if not torch.isfinite(tensor).all():
            return float("inf")
        total_sq += float(torch.sum(tensor * tensor).item())
    return float(np.sqrt(max(total_sq, 0.0)))


def _state_dict_drift(current_state, previous_state):
    """Return the global L2 distance between two server model states."""
    total_sq = 0.0
    if current_state.keys() != previous_state.keys():
        raise ValueError("Checkpoint/model state keys do not match.")

    for key in current_state:
        current = current_state[key].detach().float()
        previous = previous_state[key].detach().float()
        if current.shape != previous.shape:
            raise ValueError("Checkpoint shape mismatch for key: {}".format(key))
        if not torch.isfinite(current).all():
            return float("inf")
        total_sq += float(torch.sum((current - previous) ** 2).item())

    return float(np.sqrt(max(total_sq, 0.0)))


def assess_server_model_health(
    current_state,
    previous_state,
    previous_stable_state,
):
    """Check server-observable structural properties of a candidate model.

    This function deliberately does NOT inspect:
        - test data
        - validation data
        - labels
        - accuracy
        - predictions
        - Byzantine identities
        - attack labels
        - detection metrics

    It compares the candidate only with previously accepted server models.
    """

    current_norm = _state_dict_parameter_norm(current_state)
    previous_norm = _state_dict_parameter_norm(previous_state)
    stable_norm = _state_dict_parameter_norm(previous_stable_state)

    drift = _state_dict_drift(
        current_state,
        previous_stable_state,
    )

    base_norm = max(
        previous_norm,
        stable_norm,
        _RECOVERY_EPS,
    )

    norm_threshold = base_norm * _RECOVERY_NORM_RATIO
    drift_threshold = base_norm * _RECOVERY_DRIFT_RATIO

    # Absolute safety ceiling relative to the last stable model.
    stable_ceiling = (
        max(stable_norm, _RECOVERY_EPS)
        * _RECOVERY_MAX_NORM_MULTIPLIER
    )

    nonfinite = (
        not np.isfinite(current_norm)
        or not np.isfinite(drift)
    )

    norm_explosion = (
        current_norm > norm_threshold
        or current_norm > stable_ceiling
    )

    excessive_drift = drift > drift_threshold

    healthy = not (
        nonfinite
        or norm_explosion
        or excessive_drift
    )

    reasons = []

    if nonfinite:
        reasons.append("non-finite parameters")

    if norm_explosion:
        reasons.append("structural parameter-norm explosion")

    if excessive_drift:
        reasons.append("excessive round-to-round parameter drift")

    return {
        "healthy": healthy,
        "parameter_norm": current_norm,
        "previous_norm": previous_norm,
        "stable_norm": stable_norm,
        "parameter_norm_threshold": norm_threshold,
        "absolute_norm_ceiling": stable_ceiling,
        "parameter_drift": drift,
        "drift_threshold": drift_threshold,
        "recovery_reasons": reasons,
    }

# ── Main experiment runner ─────────────────────────────────────────────────────

def run_experiment(
    cfg,
    verbose=True,
):
    """
    Run one QBAD-FL experiment.

    IMPORTANT:
    Test accuracy is NEVER used to modify the model,
    select checkpoints, trigger recovery, or affect future rounds.

    Checkpoint recovery, when enabled, is driven ONLY by server-side
    structural health checks on the aggregated global model. No test data,
    validation data, Byzantine labels, detection metrics, or attack identities
    are used by the recovery mechanism.
    """

    dev = torch.device(
        "cpu"
    )

    # ── Model ─────────────────────────────────────────────────────────────────

    if cfg["data_name"] == "mnist":

        net = Mnist_CNN()

    else:

        net = ResNet18()

    net = net.to(dev)

    loss_func = F.cross_entropy

    opti = optim.SGD(
        net.parameters(),
        lr=cfg["learning_rate"],
    )

    # ── Clients ───────────────────────────────────────────────────────────────

    myClients = ClientsGroup(
        cfg["data_name"],
        cfg["IID"],
        cfg["num_of_clients"],
        dev,
    )

    myClients.get_central_data(
        cfg["central_data_size"],
        cfg["central_data_pro"],
    )

    testDataLoader = (
        myClients.test_data_loader
    )

    # ── Experiment identities ────────────────────────────────────────────────
    #
    # These are used for:
    #   - attack construction
    #   - evaluation
    #
    # They are NOT passed to detectors.

    nc = cfg["num_of_clients"]

    byz = cfg["byzantine_size"]

    honest_clients = [
        "client{}".format(i)
        for i in range(nc - byz)
    ]

    byzantine_clients = [
        "client{}".format(i)
        for i in range(nc - byz, nc)
    ]

    actual_malicious_indices = list(
        range(nc - byz, nc)
    )

    # ── Initial global model ──────────────────────────────────────────────────

    global_parameters = {
        k: v.clone()
        for k, v in net.state_dict().items()
    }

    # ── Server-side checkpoint baseline ──────────────────────────────────────
    # The initial model is only the first stable checkpoint.
    # Recovery thereafter uses the most recently accepted server model.
    # No test/validation information is involved.
    initial_model_state = _clone_state_dict(global_parameters)
    previous_stable_state = _clone_state_dict(global_parameters)
    previous_server_state = _clone_state_dict(global_parameters)

    recovery_count = 0

    round_results = []

    start = time.time()

    # ── Federated rounds ──────────────────────────────────────────────────────

    for rnd in range(
        cfg["num_comm"]
    ):

        rnd_start = time.time()

        if verbose:

            print(
                "\n─── Round {}/{} ───".format(
                    rnd + 1,
                    cfg["num_comm"],
                )
            )

        # ── Central training ──────────────────────────────────────────────────

        Central_par = myClients.centralTrain(
            cfg["epoch"],
            cfg["batchsize"],
            net,
            loss_func,
            opti,
            global_parameters,
        )

        # VQC detectors are fixed feature extractors.
        detector_conv1, detector_fc = (
            feature_extraction_model(
                Central_par,
                cfg,
                dev,
            )
        )

        # ── Honest client updates ─────────────────────────────────────────────

        Upload_Parameters = []

        honest_all_weight = None

        for cl in honest_clients:

            lp = myClients.clients_set[
                cl
            ].localTrain(
                cfg["epoch"],
                cfg["batchsize"],
                net,
                loss_func,
                opti,
                global_parameters,
            )

            Upload_Parameters.append(
                lp
            )

            # Used ONLY to construct simulated attacks.
            if (
                cfg["pattern"] <= 2
                or
                cfg["pattern"] >= 5
            ):

                if honest_all_weight is None:

                    honest_all_weight = {
                        k: v.clone().unsqueeze(0)
                        for k, v in lp.items()
                    }

                else:

                    for k in lp:

                        honest_all_weight[k] = torch.cat(
                            [
                                honest_all_weight[k],
                                lp[k].unsqueeze(0),
                            ],
                            dim=0,
                        )

        # ── Byzantine attack construction ────────────────────────────────────

        byz_uploads = collect_attack_updates(
            byzantine_clients,
            myClients,
            cfg["pattern"],
            honest_all_weight,
            nc,
            cfg["epoch"],
            cfg["batchsize"],
            net,
            loss_func,
            opti,
            global_parameters,
            data_name=cfg["data_name"],
        )

        # IMPORTANT:
        #
        # Only AFTER all honest and Byzantine updates have been constructed
        # do we pass the complete population to the detectors.
        #
        # No honest-only history is constructed for detection.

        Upload_Parameters.extend(
            byz_uploads
        )

        # ── Stage 1: Norm filtering ───────────────────────────────────────────

        norm_rejected = detect_norm_outliers(
            Upload_Parameters,
            cfg,
            dev,
        )

        print(
            "  [Norm Filter] Rejected: {}".format(
                norm_rejected
            )
        )

        # ── Stage 2: VQC + sign-flip detection ───────────────────────────────

        vqc_detected = vqc_feature_extraction(
            Upload_Parameters,
            detector_conv1,
            detector_fc,
            cfg,
            dev,
        )

        # ── Stage 3: Ensemble ─────────────────────────────────────────────────

        detected = list(
            set(norm_rejected)
            |
            set(vqc_detected)
        )

        print(
            "  [Ensemble] "
            "Norm filter={} + VQC={} → Final={}: {}".format(
                len(norm_rejected),
                len(vqc_detected),
                len(detected),
                sorted(detected),
            )
        )

        # ── DEBUG / EVALUATION ONLY ──────────────────────────────────────────

        print(
            "\n  [DEBUG] Gradient Norms Analysis:"
        )

        norms = [

            torch.norm(
                torch.cat(
                    [
                        u[k].flatten()
                        for k in sorted(u.keys())
                    ]
                )
            ).item()

            for u in Upload_Parameters
        ]

        honest_end = nc - byz

        for i, n in enumerate(norms):

            is_byz = (
                " ← BYZANTINE"
                if i >= honest_end
                else ""
            )

            print(
                "    Client {:2d}: norm={:.4f}{}".format(
                    i,
                    n,
                    is_byz,
                )
            )

        if honest_end > 0:

            print(
                "    Honest norms - "
                "Median: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
                    np.median(
                        norms[:honest_end]
                    ),
                    np.mean(
                        norms[:honest_end]
                    ),
                    np.std(
                        norms[:honest_end]
                    ),
                )
            )

        else:

            print(
                "    Honest norms - N/A "
                "(no honest clients)"
            )

        if byz > 0:

            print(
                "    Byz norms   - "
                "Median: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
                    np.median(
                        norms[honest_end:]
                    ),
                    np.mean(
                        norms[honest_end:]
                    ),
                    np.std(
                        norms[honest_end:]
                    ),
                )
            )

        else:

            print(
                "    Byz norms   - N/A "
                "(no Byzantine clients)"
            )

        # ── Stage 4: Label-independent clipping ──────────────────────────────

        clip_gradient_norms(
            Upload_Parameters,
            cfg,
        )

        # ── Stage 5: Federated aggregation ───────────────────────────────────

        candidate_parameters = fed_avg(
            list(Upload_Parameters),
            detected,
        )

        # ── Stage 6: Server-side structural health / checkpoint recovery ─────
        #
        # IMPORTANT:
        # This stage sees ONLY the candidate global model and previously
        # accepted server models. It does NOT see test/validation data,
        # accuracy, labels, Byzantine identities, attack labels, or metrics.
        recovery_triggered = False
        recovery_reasons = []

        if _RECOVERY_ENABLED:
            health = assess_server_model_health(
                candidate_parameters,
                previous_server_state,
                previous_stable_state,
            )

            print(
                "  [Model Health] "
                "param_norm={:.6f} | "
                "previous_norm={:.6f} | "
                "drift={:.6f} | "
                "norm_threshold={:.6f} | "
                "drift_threshold={:.6f}".format(
                    health["parameter_norm"],
                    health["previous_norm"],
                    health["parameter_drift"],
                    health["parameter_norm_threshold"],
                    health["drift_threshold"],
                )
            )

            if not health["healthy"]:
                recovery_triggered = True
                recovery_reasons = list(
                    health["recovery_reasons"]
                )
                recovery_count += 1

                print(
                    "  [Recovery] Candidate rejected; restoring "
                    "previous structurally stable server checkpoint."
                )
                print(
                    "  [Recovery] Reason(s): {}".format(
                        ", ".join(recovery_reasons)
                    )
                )

                global_parameters = _clone_state_dict(
                    previous_stable_state
                )
                net.load_state_dict(
                    global_parameters,
                    strict=True,
                )

                # The rejected candidate is never used as history.
                previous_server_state = _clone_state_dict(
                    previous_stable_state
                )

            else:
                global_parameters = _clone_state_dict(
                    candidate_parameters
                )

                previous_server_state = _clone_state_dict(
                    global_parameters
                )
                previous_stable_state = _clone_state_dict(
                    global_parameters
                )

                print("  [Recovery] Candidate accepted.")

        else:
            global_parameters = _clone_state_dict(
                candidate_parameters
            )
            previous_server_state = _clone_state_dict(
                global_parameters
            )
            previous_stable_state = _clone_state_dict(
                global_parameters
            )

        # ── Test evaluation ONLY ──────────────────────────────────────────────
        #
        # IMPORTANT:
        # The test result does NOT:
        #   - select a checkpoint
        #   - restore a model
        #   - change the learning process
        #   - change detection
        #   - change aggregation
        #
        # It is purely reported evaluation.

        net.load_state_dict(
            global_parameters,
            strict=True,
        )

        sum_accu = 0.0
        num_batches = 0

        with torch.no_grad():

            for data, label in testDataLoader:

                data = data.to(dev)
                label = label.to(dev)

                preds = torch.argmax(
                    net(data),
                    dim=1,
                )

                sum_accu += (
                    (preds == label)
                    .float()
                    .mean()
                    .item()
                )

                num_batches += 1

        if num_batches > 0:

            accuracy = (
                sum_accu
                /
                num_batches
            )

        else:

            accuracy = 0.0

        # ── Detection metrics ────────────────────────────────────────────────
        #
        # Ground truth enters ONLY here for evaluation.

        dr = calculate_detection_rate(
            detected,
            actual_malicious_indices,
        )

        fpr = calculate_false_positive_rate(
            detected,
            actual_malicious_indices,
            nc,
        )

        prec = calculate_precision(
            detected,
            actual_malicious_indices,
        )

        rec = calculate_recall(
            detected,
            actual_malicious_indices,
        )

        f1 = calculate_f1(
            detected,
            actual_malicious_indices,
        )

        rnd_time = (
            time.time()
            -
            rnd_start
        )

        rnd_result = {
            "round": rnd + 1,
            "accuracy": accuracy,
            "detection_rate": dr,
            "false_positive_rate": fpr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "recovery_enabled": _RECOVERY_ENABLED,
            "recovery_triggered": recovery_triggered,
            "recovery_reasons": recovery_reasons,
            "detected_malicious": sorted(detected),
            "actual_malicious": actual_malicious_indices,
            "round_time_seconds": rnd_time,
        }

        round_results.append(
            rnd_result
        )

        if verbose:

            print(
                "  accuracy={:.2%}  "
                "detection_rate={:.2%}  "
                "FPR={:.2%}  "
                "F1={:.4f}  "
                "time={:.1f}s".format(
                    accuracy,
                    dr,
                    fpr,
                    f1,
                    rnd_time,
                )
            )

            print(
                "  Detected malicious: {}".format(
                    sorted(detected)
                )
            )

            print(
                "  Actual  malicious : {}".format(
                    actual_malicious_indices
                )
            )

    # ── Experiment summary ────────────────────────────────────────────────────

    total_time = (
        time.time()
        -
        start
    )

    summary = aggregate_round_metrics(
        round_results
    )

    return {
        "config": {
            "dataset": cfg["data_name"],
            "num_clients": nc,
            "byzantine_size": byz,
            "attack_pattern": cfg["pattern"],
            "attack_name": ATTACK_NAMES.get(
                cfg["pattern"],
                "Unknown",
            ),
            "iid": cfg["IID"],
            "num_comm": cfg["num_comm"],
            "checkpoint_recovery": _RECOVERY_ENABLED,
            "recovery_norm_ratio": _RECOVERY_NORM_RATIO,
            "recovery_drift_ratio": _RECOVERY_DRIFT_RATIO,
            "recovery_max_norm_multiplier": _RECOVERY_MAX_NORM_MULTIPLIER,
        },

        "round_results": round_results,

        "summary": summary,

        "total_runtime_seconds": total_time,
    }


# ── Default configuration ─────────────────────────────────────────────────────

def _default_config():

    return {

        "data_name": "mnist",

        "num_of_clients": 20,

        "byzantine_size": 5,

        "pattern": 5,

        "epoch": 5,

        "batchsize": 64,

        "learning_rate": 0.1,

        "num_comm": 3,

        "IID": True,

        "central_data_size": 300,

        "central_data_pro": 0.1,

        "alpha": 0.5,

        "iforest_contamination":
            _IFOREST_CONTAMINATION,
    }


# ── Command-line interface ────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="QBAD-FL validation test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=[
            "mnist",
            "cifar_10",
        ],
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Communication rounds",
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=20,
        help="Total clients",
    )

    parser.add_argument(
        "--byzantine",
        type=int,
        default=5,
        help="Byzantine clients",
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        type=int,
        default=[5],
        help=(
            "Attack pattern(s): "
            "0=Gaussian "
            "1=Sign-flip "
            "2=Zero-gradient "
            "3=Backdoor "
            "4=Model-replacement "
            "5=MPAF "
            "6=AGR-agnostic"
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Minimal test: "
            "10 clients, "
            "5 Byzantine, "
            "2 rounds, "
            "1 MPAF attack"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON output path"
        ),
    )

    parser.add_argument(
        "--iid",
        type=lambda x:
            x.lower()
            in (
                "true",
                "1",
                "yes",
            ),
        default=True,
        help=(
            "IID (True) or "
            "Non-IID (False)"
        ),
    )

    args = parser.parse_args()

    # ── Quick mode ────────────────────────────────────────────────────────────

    if args.quick:

        args.clients = 10

        args.byzantine = 5

        args.rounds = 2

        args.attacks = [5]

        print(
            "Quick mode: "
            "10 clients, "
            "5 Byzantine, "
            "2 rounds, "
            "MPAF attack"
        )

    # ── Configuration ─────────────────────────────────────────────────────────

    cfg = _default_config()

    cfg["data_name"] = (
        args.dataset
    )

    cfg["num_comm"] = (
        args.rounds
    )

    cfg["num_of_clients"] = (
        args.clients
    )

    cfg["byzantine_size"] = (
        args.byzantine
    )

    cfg["IID"] = (
        args.iid
    )

    all_results = []

    # ── Run requested attacks ─────────────────────────────────────────────────

    for attack in args.attacks:

        cfg["pattern"] = attack

        print(
            "\n"
            +
            "=" * 65
        )

        print(
            "Attack: {} ({})".format(
                attack,
                ATTACK_NAMES.get(
                    attack,
                    "?",
                ),
            )
        )

        print(
            "=" * 65
        )

        results = run_experiment(
            cfg,
            verbose=True,
        )

        all_results.append(
            results
        )

        print(
            generate_report(
                results,
                title=(
                    "QBAD-FL — Attack {}".format(
                        attack
                    )
                ),
            )
        )

    # ── Multi-attack summary ──────────────────────────────────────────────────

    if len(all_results) > 1:

        print(
            "\n\n"
            +
            "=" * 65
        )

        print(
            "Multi-Attack Summary".center(65)
        )

        print(
            "=" * 65
        )

        print(
            "{:<20s} {:>10s} {:>14s} {:>10s}".format(
                "Attack",
                "Accuracy",
                "DetectionRate",
                "FPR",
            )
        )

        print(
            "-" * 55
        )

        for r in all_results:

            s = r["summary"]

            print(
                "{:<20s} {:>10.2%} {:>14.2%} {:>10.2%}".format(
                    r["config"]["attack_name"],
                    s.get(
                        "accuracy",
                        {}
                    ).get(
                        "final",
                        0.0,
                    ),
                    s.get(
                        "detection_rate",
                        {}
                    ).get(
                        "final",
                        0.0,
                    ),
                    s.get(
                        "false_positive_rate",
                        {}
                    ).get(
                        "final",
                        0.0,
                    ),
                )
            )

        print(
            "=" * 65
        )

    # ── Save results ──────────────────────────────────────────────────────────

    if args.output:

        save_results_json(
            {
                "experiments":
                    all_results
            },
            args.output,
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
