"""
metrics.py — Utility functions for QBAD-FL evaluation and reporting.

Provides helpers to calculate Byzantine detection metrics (detection rate,
false positive rate, precision/recall), model accuracy, comparison against
the FLAD baseline, and formatted report generation.
"""

import json
import os
import time

import numpy as np


# ── Detection metrics ─────────────────────────────────────────────────────────


def calculate_detection_rate(detected_malicious, actual_malicious_indices):
    """Percentage of Byzantine clients correctly identified.

    Parameters
    ----------
    detected_malicious : list[int]
        Client indices flagged as malicious by the detector.
    actual_malicious_indices : list[int]
        Ground-truth Byzantine client indices.

    Returns
    -------
    float in [0, 1]  —  TP / (total actual malicious).  Returns 0.0 if there
    are no actual malicious clients.
    """
    if len(actual_malicious_indices) == 0:
        return 0.0
    actual_set = set(actual_malicious_indices)
    detected_set = set(detected_malicious)
    tp = len(actual_set & detected_set)
    return tp / len(actual_set)


def calculate_false_positive_rate(detected_malicious, actual_malicious_indices,
                                   num_clients):
    """Percentage of honest clients incorrectly flagged as malicious.

    Parameters
    ----------
    detected_malicious : list[int]
        Client indices flagged as malicious by the detector.
    actual_malicious_indices : list[int]
        Ground-truth Byzantine client indices.
    num_clients : int
        Total number of clients.

    Returns
    -------
    float in [0, 1]  —  FP / (total actual honest).  Returns 0.0 if there are
    no honest clients.
    """
    actual_set = set(actual_malicious_indices)
    detected_set = set(detected_malicious)
    num_honest = num_clients - len(actual_set)
    if num_honest <= 0:
        return 0.0
    fp = len(detected_set - actual_set)
    return fp / num_honest


def calculate_precision(detected_malicious, actual_malicious_indices):
    """Precision of Byzantine detection: TP / (TP + FP).

    Returns 0.0 when the detector makes no detections.
    """
    if len(detected_malicious) == 0:
        return 0.0
    actual_set = set(actual_malicious_indices)
    detected_set = set(detected_malicious)
    tp = len(actual_set & detected_set)
    return tp / len(detected_set)


def calculate_recall(detected_malicious, actual_malicious_indices):
    """Recall (same as detection rate): TP / (TP + FN)."""
    return calculate_detection_rate(detected_malicious, actual_malicious_indices)


def calculate_f1(detected_malicious, actual_malicious_indices):
    """F1 score for Byzantine detection."""
    p = calculate_precision(detected_malicious, actual_malicious_indices)
    r = calculate_recall(detected_malicious, actual_malicious_indices)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def calculate_accuracy(sum_accu, num_batches):
    """Model test accuracy from accumulated batch-level sum.

    Parameters
    ----------
    sum_accu    : float  Sum of per-batch accuracy values.
    num_batches : int    Number of batches evaluated.

    Returns
    -------
    float in [0, 1]
    """
    if num_batches == 0:
        return 0.0
    return float(sum_accu) / num_batches


# ── Per-round metrics aggregation ─────────────────────────────────────────────


def aggregate_round_metrics(round_results):
    """Compute summary statistics across all communication rounds.

    Parameters
    ----------
    round_results : list[dict]
        Each dict must contain keys: ``accuracy``, ``detection_rate``,
        ``false_positive_rate``, ``precision``, ``recall``, ``f1``.

    Returns
    -------
    dict with mean / std / min / max for each metric.
    """
    if not round_results:
        return {}

    metrics_keys = ["accuracy", "detection_rate", "false_positive_rate",
                    "precision", "recall", "f1"]
    summary = {}
    for key in metrics_keys:
        values = [r[key] for r in round_results if key in r]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values)),
                "min":  float(np.min(values)),
                "max":  float(np.max(values)),
                "final": float(values[-1]),
            }
    return summary


# ── Reporting ─────────────────────────────────────────────────────────────────

ATTACK_NAMES = {
    0: "Gaussian",
    1: "Sign-Flip",
    2: "Zero-Gradient",
    3: "Backdoor",
    4: "Model-Replacement",
    5: "MPAF",
    6: "AGR-Agnostic",
}


def generate_report(results, title="QBAD-FL Experiment Results"):
    """Format experiment results as a human-readable string.

    Parameters
    ----------
    results : dict  Output from an experiment run.
    title   : str   Report header text.

    Returns
    -------
    str
    """
    lines = []
    lines.append("=" * 65)
    lines.append(title.center(65))
    lines.append("=" * 65)

    cfg = results.get("config", {})
    if cfg:
        lines.append("\nConfiguration:")
        lines.append("  Dataset      : {}".format(cfg.get("dataset", "?")))
        lines.append("  Clients      : {} total, {} Byzantine".format(
            cfg.get("num_clients", "?"), cfg.get("byzantine_size", "?")))
        attack_id = cfg.get("attack_pattern", "?")
        attack_name = ATTACK_NAMES.get(attack_id, "Unknown")
        lines.append("  Attack       : {} ({})".format(attack_id, attack_name))
        lines.append("  IID          : {}".format(cfg.get("iid", "?")))
        lines.append("  Rounds       : {}".format(cfg.get("num_comm", "?")))

    summary = results.get("summary", {})
    if summary:
        lines.append("\nSummary Metrics (across all rounds):")
        for metric, stats in summary.items():
            lines.append("  {:22s}  mean={:.4f}  std={:.4f}  final={:.4f}".format(
                metric + ":", stats["mean"], stats["std"], stats["final"]))

    runtime = results.get("total_runtime_seconds")
    if runtime is not None:
        lines.append("\n  Total runtime: {:.1f}s ({:.1f} min)".format(
            runtime, runtime / 60))

    lines.append("=" * 65)

    # Pass/fail indicator
    final_acc = (summary.get("accuracy", {}).get("final", 0.0))
    status = "✅ PASS (accuracy >= 85%)" if final_acc >= 0.85 else "⚠️  BELOW TARGET (< 85%)"
    lines.append("  Final accuracy: {:.2%}  →  {}".format(final_acc, status))
    lines.append("=" * 65)

    return "\n".join(lines)


# ── FLAD comparison ───────────────────────────────────────────────────────────


def compare_with_flad(qbad_results, flad_results):
    """Build a side-by-side comparison dict between QBAD-FL and FLAD.

    Parameters
    ----------
    qbad_results : dict  Results from a QBAD-FL experiment run.
    flad_results : dict  Results from an equivalent FLAD experiment run.

    Returns
    -------
    dict with keys:
      ``metric``            — metric name
      ``flad_value``        — FLAD final value
      ``qbad_value``        — QBAD-FL final value
      ``difference``        — qbad - flad
      ``advantage``         — "QBAD-FL" | "FLAD" | "Tie"
    """
    metrics_keys = ["accuracy", "detection_rate", "false_positive_rate",
                    "precision", "recall", "f1"]
    comparison = []
    q_summary = qbad_results.get("summary", {})
    f_summary = flad_results.get("summary", {})

    for key in metrics_keys:
        q_val = q_summary.get(key, {}).get("final", None)
        f_val = f_summary.get(key, {}).get("final", None)
        if q_val is None or f_val is None:
            continue
        diff = q_val - f_val
        # For FPR, lower is better; for everything else higher is better
        if key == "false_positive_rate":
            advantage = "QBAD-FL" if diff < -0.001 else ("FLAD" if diff > 0.001 else "Tie")
        else:
            advantage = "QBAD-FL" if diff > 0.001 else ("FLAD" if diff < -0.001 else "Tie")
        comparison.append({
            "metric": key,
            "flad_value": f_val,
            "qbad_value": q_val,
            "difference": diff,
            "advantage": advantage,
        })
    return comparison


def format_comparison_table(comparison, attack_name=""):
    """Render a comparison list as a formatted ASCII table string."""
    lines = []
    header = " Attack: {}".format(attack_name) if attack_name else ""
    lines.append("\nComparison Table{}".format(header))
    lines.append("-" * 70)
    lines.append("{:<26s} {:>10s} {:>12s} {:>12s}  {}".format(
        "Metric", "FLAD", "QBAD-FL", "Difference", "Advantage"))
    lines.append("-" * 70)
    for row in comparison:
        lines.append("{:<26s} {:>10.4f} {:>12.4f} {:>+12.4f}  {}".format(
            row["metric"],
            row["flad_value"],
            row["qbad_value"],
            row["difference"],
            row["advantage"],
        ))
    lines.append("-" * 70)
    return "\n".join(lines)


# ── I/O helpers ───────────────────────────────────────────────────────────────


def save_results_json(results, filepath):
    """Serialise *results* dict to *filepath* as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("Results saved to {}".format(filepath))


def save_results_csv(rows, filepath, fieldnames=None):
    """Write a list of dicts as CSV.

    Parameters
    ----------
    rows       : list[dict]
    filepath   : str
    fieldnames : list[str] | None  — inferred from first row if None.
    """
    import csv
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("CSV saved to {}".format(filepath))


def load_results_json(filepath):
    """Load a previously saved JSON results file."""
    with open(filepath) as fh:
        return json.load(fh)
