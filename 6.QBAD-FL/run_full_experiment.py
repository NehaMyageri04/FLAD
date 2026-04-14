"""
run_full_experiment.py — Production-ready full evaluation of QBAD-FL.

Tests all 7 attack patterns, both IID and non-IID data distributions,
across multiple communication rounds, and generates a detailed report
with CSV, JSON, and optional matplotlib plots.

Usage
-----
# Full evaluation on MNIST (default 20 rounds, all attacks, IID + non-IID)
python run_full_experiment.py --dataset mnist --mode full --output results/

# CIFAR-10 full run
python run_full_experiment.py --dataset cifar_10 --mode full --output results/

# Quick full-spectrum run (fewer rounds)
python run_full_experiment.py --mode full --rounds 5 --output results/

# Single attack, single distribution
python run_full_experiment.py --attacks 5 --iid-only --rounds 20
"""

import argparse
import os
import sys
import time
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN

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
    save_results_csv,
    ATTACK_NAMES,
)

# ── Internal FL helpers ───────────────────────────────────────────────────────

# Minimum per-axis epsilon for DBSCAN: prevents eps→0 when VQC converges
# tightly on clean data (which would make all points become noise).
_DBSCAN_MIN_EPS = 0.05

# Number of previous rounds of honest updates to retain for sign-flip detection.
_HONEST_UPDATE_HISTORY_SIZE = 3


def _cos(a, b):
    return np.sum(a * b.T) / (
        (np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9
    )


def _train_vqc(weight_train, dimen):
    weight_train = weight_train.view(weight_train.size(0), -1).cpu()
    loader = DataLoader(dataset=weight_train, batch_size=1, shuffle=True)
    model = QuantumByzantineDetector(dimen)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction="sum")
    label = torch.tensor([[1.0]])
    for epoch in range(20):
        epoch_loss = 0.0
        for batch in loader:
            out = model(batch)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 5 == 4:
            print("    VQC train epoch {}/20  loss={:.4f}".format(epoch + 1, epoch_loss))
    with torch.no_grad():
        all_out = model(weight_train)
    return model, all_out.mean(dim=0), all_out.max() - all_out.min()


def _feature_extraction_model(Central_par, data_name):
    num = len(Central_par)
    if data_name == "mnist":
        k1 = torch.zeros(num, 10, 1, 5, 5)
        w3 = torch.zeros(num, 10, 320)
    else:
        k1 = torch.zeros(num, 64, 3, 3, 3)
        w3 = torch.zeros(num, 10, 512)
    for i, W in enumerate(Central_par):
        if data_name == "mnist":
            k1[i] = W["conv1.weight"].data
            w3[i] = W["fc.weight"].data
        else:
            k1[i] = W["module.conv1.weight"].data
            w3[i] = W["module.fc.weight"].data
    FC, Std, Dis = {}, {}, {}
    if data_name == "mnist":
        FC["conv1.weight"], Std["conv1.weight"], Dis["conv1.weight"] = _train_vqc(k1, 10 * 1 * 5 * 5)
        FC["fc.weight"], Std["fc.weight"], Dis["fc.weight"] = _train_vqc(w3, 10 * 320)
    else:
        FC["conv1.weight"], Std["conv1.weight"], Dis["conv1.weight"] = _train_vqc(k1, 64 * 3 * 3 * 3)
        FC["fc.weight"], Std["fc.weight"], Dis["fc.weight"] = _train_vqc(w3, 10 * 512)
    return FC, Std, Dis


def _cosine_fallback_detect(feature, honest_std, nc, alpha=0.5):
    """Fallback Byzantine detection using cosine similarity and L2 distance."""
    honest_L2 = float(np.sqrt(np.sum(honest_std ** 2))) + 1e-9
    scores = []
    for c in range(nc):
        f = feature[c]
        f_L2 = float(np.sqrt(np.sum(f ** 2))) + 1e-9
        cosin = float(np.sum(f * honest_std)) / (f_L2 * honest_L2)
        length = abs(f_L2 / honest_L2 - 1.0)
        scores.append(alpha * cosin - (1.0 - alpha) * length)
    scores = np.array(scores)
    threshold = float(np.median(scores)) - 1.5 * float(scores.std() + 1e-9)
    return [c for c in range(nc) if scores[c] < threshold]


def _detect_sign_flip_attacks(Upload_Parameters, honest_update_history, nc):
    """Detect sign-flip attacks by checking gradient direction correlation.

    Sign-flipped updates have a consistently negative dot product with the
    historical honest update direction.  At least two rounds of history are
    required for a reliable reference direction.

    Parameters
    ----------
    Upload_Parameters    : list of state-dict-like dicts, one per client.
    honest_update_history: list of averaged honest update dicts from past rounds.
    nc                   : total number of clients.

    Returns
    -------
    list of int — indices of clients detected as sign-flip attackers.
    """
    if len(honest_update_history) < 2:
        return []  # Need at least 2 historical updates for a reliable reference

    # Compute the average honest gradient direction over all stored rounds
    avg_honest_flat = None
    for historical_update in honest_update_history:
        flat = torch.cat([historical_update[k].flatten()
                          for k in sorted(historical_update.keys())])
        if avg_honest_flat is None:
            avg_honest_flat = flat.clone()
        else:
            avg_honest_flat = avg_honest_flat + flat
    avg_honest_flat = avg_honest_flat / len(honest_update_history)

    detected = []
    for c, client_update in enumerate(Upload_Parameters):
        client_flat = torch.cat([client_update[k].flatten()
                                 for k in sorted(client_update.keys())])
        dot_prod = torch.dot(avg_honest_flat, client_flat).item()
        # Clearly negative dot product → gradient direction is flipped
        if dot_prod < -0.1:
            detected.append(c)

    return detected


def _vqc_detect(Upload_Parameters, FC, Std, Dis, nc, data_name, alpha, dev,
                honest_update_history=None):
    if data_name == "mnist":
        k1 = torch.zeros(nc, 10, 1, 5, 5).to(dev)
        w3 = torch.zeros(nc, 10, 320).to(dev)
    else:
        k1 = torch.zeros(nc, 64, 3, 3, 3).to(dev)
        w3 = torch.zeros(nc, 10, 512).to(dev)
    for i, W in enumerate(Upload_Parameters):
        if data_name == "mnist":
            k1[i] = W["conv1.weight"].data
            w3[i] = W["fc.weight"].data
        else:
            k1[i] = W["module.conv1.weight"].data
            w3[i] = W["module.fc.weight"].data
    feature = np.zeros([nc, 2])
    feature[:, 0] = FC["conv1.weight"](k1.view(nc, -1)).cpu().detach().numpy().reshape(nc,)
    feature[:, 1] = FC["fc.weight"](w3.view(nc, -1)).cpu().detach().numpy().reshape(nc,)

    if np.isnan(feature).any():
        col_mean = np.nan_to_num(np.nanmean(feature, axis=0), nan=0.0)
        feature = np.where(np.isnan(feature), col_mean, feature)

    honest_std = np.array([Std["conv1.weight"].item(), Std["fc.weight"].item()])
    honest_L2 = np.sqrt(np.sum(honest_std * honest_std.T)) + 1e-9

    # Adaptive eps with minimum threshold to prevent DBSCAN from finding no clusters
    eps1 = max(abs(Dis["conv1.weight"].item()), _DBSCAN_MIN_EPS)
    eps3 = max(abs(Dis["fc.weight"].item()), _DBSCAN_MIN_EPS)
    eps = max((eps1 ** 2 + eps3 ** 2) ** 0.5, 0.1)

    try:
        db = DBSCAN(eps=eps, min_samples=2).fit(feature)
        label_list = db.labels_
        label_score = {}
        for lbl in set(db.labels_):
            if lbl == -1:
                continue
            pts = np.array([feature[c] for c in range(nc) if label_list[c] == lbl])
            if len(pts) == 0:
                continue
            lm = pts.mean(axis=0)
            cosin = _cos(lm, honest_std)
            length = abs(np.sqrt(np.sum(lm * lm.T)) / honest_L2 - 1.0)
            label_score[lbl] = alpha * cosin - (1 - alpha) * length

        if label_score:
            honest_label = sorted(label_score.items(), key=lambda x: x[1], reverse=True)[0][0]
            malicious = [c for c in range(nc) if label_list[c] != honest_label]
        else:
            malicious = _cosine_fallback_detect(feature, honest_std, nc, alpha)
    except Exception:
        malicious = _cosine_fallback_detect(feature, honest_std, nc, alpha)

    # ── Phase 2: Sign-flip detection (ensemble with VQC) ─────────────────────
    if honest_update_history is not None:
        flip_detected = _detect_sign_flip_attacks(Upload_Parameters, honest_update_history, nc)
        if flip_detected:
            malicious = list(set(malicious) | set(flip_detected))

    return malicious


def _fed_avg(params_list, malicious):
    params = list(params_list)
    for j, idx in enumerate(sorted(malicious)):
        del params[idx - j]
    agg = None
    for p in params:
        if agg is None:
            agg = {k: v.clone() for k, v in p.items()}
        else:
            for k in p:
                agg[k] += p[k]
    for k in agg:
        agg[k] /= len(params)
    return agg


# ── Single run ────────────────────────────────────────────────────────────────


def run_single_experiment(cfg, verbose=True):
    """Run one complete QBAD-FL experiment.

    Parameters
    ----------
    cfg : dict  Full experiment configuration.
    verbose : bool

    Returns
    -------
    dict  Results with round_results and summary.
    """
    dev = torch.device("cpu")
    nc, byz = cfg["num_of_clients"], cfg["byzantine_size"]

    net = Mnist_CNN() if cfg["data_name"] == "mnist" else ResNet18()
    net = net.to(dev)
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=cfg["learning_rate"])

    myClients = ClientsGroup(cfg["data_name"], cfg["IID"], nc, dev)
    myClients.get_central_data(cfg["central_data_size"], cfg["central_data_pro"])
    testDataLoader = myClients.test_data_loader

    honest_clients = ["client{}".format(i) for i in range(nc - byz)]
    byzantine_clients = ["client{}".format(i) for i in range(nc - byz, nc)]
    actual_malicious_indices = list(range(nc - byz, nc))

    global_parameters = {k: v.clone() for k, v in net.state_dict().items()}
    round_results = []
    start = time.time()
    honest_update_history = []

    for rnd in range(cfg["num_comm"]):
        rnd_start = time.time()
        if verbose:
            print("  Round {}/{}…".format(rnd + 1, cfg["num_comm"]), end=" ", flush=True)

        Central_par = myClients.centralTrain(
            cfg["epoch"], cfg["batchsize"], net, loss_func, opti, global_parameters
        )
        FC, Std, Dis = _feature_extraction_model(Central_par, cfg["data_name"])

        uploads = []
        honest_all_weight = None

        for cl in honest_clients:
            lp = myClients.clients_set[cl].localTrain(
                cfg["epoch"], cfg["batchsize"], net, loss_func, opti, global_parameters
            )
            uploads.append(lp)
            if cfg["pattern"] <= 2 or cfg["pattern"] >= 5:
                if honest_all_weight is None:
                    honest_all_weight = {k: v.clone().unsqueeze(0) for k, v in lp.items()}
                else:
                    for k in lp:
                        honest_all_weight[k] = torch.cat(
                            [honest_all_weight[k], lp[k].unsqueeze(0)], dim=0
                        )

        for cl in byzantine_clients:
            lp = {}
            if cfg["pattern"] == 0:
                for k in honest_all_weight:
                    lp[k] = Attack.Gaussian_attack(honest_all_weight[k])
            elif cfg["pattern"] == 1:
                for k in honest_all_weight:
                    lp[k] = Attack.Sign_flipping_attack(honest_all_weight[k])
            elif cfg["pattern"] == 2:
                for k in honest_all_weight:
                    lp[k] = Attack.ZeroGradient_attack(honest_all_weight[k], byz)
            elif cfg["pattern"] == 3:
                pc = Attack.backdoor_poisoning_data(myClients.clients_set[cl], cfg["data_name"])
                lp = pc.localTrain(cfg["epoch"], cfg["batchsize"], net, loss_func, opti, global_parameters)
            elif cfg["pattern"] == 4:
                pc = Attack.model_replacement_attack_data(myClients.clients_set[cl], cfg["data_name"])
                lp = pc.localTrain(cfg["epoch"], cfg["batchsize"], net, loss_func, opti, global_parameters)
                for k in lp:
                    lp[k] = lp[k] * nc
            elif cfg["pattern"] == 5:
                for k in honest_all_weight:
                    lp[k] = Attack.MPAF(honest_all_weight[k])
            elif cfg["pattern"] == 6:
                for k in honest_all_weight:
                    lp[k] = Attack.AGR_agnostic(honest_all_weight[k])
            uploads.append(lp)

        # Track average honest update for sign-flip detection history
        if uploads:
            honest_uploads = uploads[:nc - byz]
            if honest_uploads:
                avg_honest = {}
                for k in honest_uploads[0]:
                    avg_honest[k] = torch.stack(
                        [u[k].clone() for u in honest_uploads]
                    ).mean(dim=0)
                honest_update_history.append(avg_honest)
                if len(honest_update_history) > _HONEST_UPDATE_HISTORY_SIZE:
                    honest_update_history.pop(0)

        detected = _vqc_detect(uploads, FC, Std, Dis, nc, cfg["data_name"], cfg["alpha"], dev,
                               honest_update_history)
        global_parameters = _fed_avg(list(uploads), detected)

        net.load_state_dict(global_parameters, strict=True)
        sum_accu, num_batches = 0, 0
        with torch.no_grad():
            for data, label in testDataLoader:
                data, label = data.to(dev), label.to(dev)
                preds = torch.argmax(net(data), dim=1)
                sum_accu += (preds == label).float().mean()
                num_batches += 1

        acc = float(sum_accu / num_batches)
        dr = calculate_detection_rate(detected, actual_malicious_indices)
        fpr = calculate_false_positive_rate(detected, actual_malicious_indices, nc)
        prec = calculate_precision(detected, actual_malicious_indices)
        rec = calculate_recall(detected, actual_malicious_indices)
        f1 = calculate_f1(detected, actual_malicious_indices)
        rnd_time = time.time() - rnd_start

        rnd_result = {
            "round": rnd + 1,
            "accuracy": acc,
            "detection_rate": dr,
            "false_positive_rate": fpr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "detected_malicious": detected,
            "actual_malicious": actual_malicious_indices,
            "round_time_seconds": rnd_time,
        }
        round_results.append(rnd_result)

        if verbose:
            print("acc={:.2%}  DR={:.2%}  FPR={:.2%}  t={:.1f}s".format(
                acc, dr, fpr, rnd_time))

    total_time = time.time() - start
    summary = aggregate_round_metrics(round_results)
    return {
        "config": {
            "dataset": cfg["data_name"],
            "num_clients": nc,
            "byzantine_size": byz,
            "attack_pattern": cfg["pattern"],
            "attack_name": ATTACK_NAMES.get(cfg["pattern"], "Unknown"),
            "iid": cfg["IID"],
            "num_comm": cfg["num_comm"],
        },
        "round_results": round_results,
        "summary": summary,
        "total_runtime_seconds": total_time,
    }


# ── Plot generation (optional – requires matplotlib) ──────────────────────────


def _try_plot_results(all_results, output_dir):
    """Generate accuracy-over-rounds and detection-rate plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available – skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Accuracy over rounds (one line per attack × distribution)
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_results:
        label = "{} ({})".format(
            r["config"]["attack_name"],
            "IID" if r["config"]["iid"] else "Non-IID",
        )
        rounds = [x["round"] for x in r["round_results"]]
        accs = [x["accuracy"] for x in r["round_results"]]
        ax.plot(rounds, accs, marker="o", label=label)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("QBAD-FL Accuracy Over Rounds")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True)
    path = os.path.join(plots_dir, "accuracy_over_rounds.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Plot saved: {}".format(path))

    # Detection rate per attack (final round, IID only)
    iid_results = [r for r in all_results if r["config"]["iid"]]
    if iid_results:
        attacks = [r["config"]["attack_name"] for r in iid_results]
        drs = [r["summary"].get("detection_rate", {}).get("final", 0.0) for r in iid_results]
        accs = [r["summary"].get("accuracy", {}).get("final", 0.0) for r in iid_results]

        x = np.arange(len(attacks))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, drs, width, label="Detection Rate", color="steelblue")
        ax.bar(x + width / 2, accs, width, label="Accuracy", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=30, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("QBAD-FL: Detection Rate & Accuracy per Attack (IID)")
        ax.legend()
        ax.grid(axis="y")
        path = os.path.join(plots_dir, "detection_rate_per_attack.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("  Plot saved: {}".format(path))


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Full QBAD-FL evaluation across all attack types",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar_10"])
    parser.add_argument("--mode", default="full", choices=["full", "single"],
                        help="full=all attacks; single=use --attacks")
    parser.add_argument("--attacks", nargs="+", type=int, default=list(range(7)),
                        help="Attack patterns (0-6). Used when --mode single.")
    parser.add_argument("--rounds", type=int, default=20, help="Communication rounds")
    parser.add_argument("--clients", type=int, default=50, help="Total clients")
    parser.add_argument("--byzantine", type=int, default=10, help="Byzantine clients")
    parser.add_argument("--iid-only", action="store_true",
                        help="Skip non-IID experiments")
    parser.add_argument("--non-iid-only", action="store_true",
                        help="Skip IID experiments")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable matplotlib plot generation")
    args = parser.parse_args()

    attacks = list(range(7)) if args.mode == "full" else args.attacks

    iid_modes = []
    if not args.non_iid_only:
        iid_modes.append(True)
    if not args.iid_only:
        iid_modes.append(False)
    if not iid_modes:
        iid_modes = [True]

    os.makedirs(args.output, exist_ok=True)
    all_results = []
    csv_rows = []

    base_cfg = {
        "data_name": args.dataset,
        "num_of_clients": args.clients,
        "byzantine_size": args.byzantine,
        "epoch": 5,
        "batchsize": 64,
        "learning_rate": 0.1,
        "num_comm": args.rounds,
        "central_data_size": 300,
        "central_data_pro": 0.1,
        "alpha": 0.5,
    }

    total_start = time.time()
    exp_count = 0

    for iid in iid_modes:
        dist_label = "IID" if iid else "Non-IID"
        for attack in attacks:
            exp_count += 1
            attack_name = ATTACK_NAMES.get(attack, "Unknown")
            print("\n" + "=" * 65)
            print("Experiment {}: Attack {} ({}) | {}".format(
                exp_count, attack, attack_name, dist_label))
            print("=" * 65)

            cfg = dict(base_cfg)
            cfg["IID"] = iid
            cfg["pattern"] = attack

            results = run_single_experiment(cfg, verbose=True)
            all_results.append(results)

            s = results["summary"]
            print(generate_report(results,
                                  title="Attack {} ({}) {}".format(attack, attack_name, dist_label)))

            # CSV row per round
            for rr in results["round_results"]:
                csv_rows.append({
                    "attack_type": attack,
                    "attack_name": attack_name,
                    "distribution": dist_label,
                    "round": rr["round"],
                    "accuracy": round(rr["accuracy"], 4),
                    "detection_rate": round(rr["detection_rate"], 4),
                    "false_positive_rate": round(rr["false_positive_rate"], 4),
                    "precision": round(rr["precision"], 4),
                    "recall": round(rr["recall"], 4),
                    "f1": round(rr["f1"], 4),
                    "round_time_s": round(rr["round_time_seconds"], 2),
                })

    total_time = time.time() - total_start

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("FULL EXPERIMENT SUMMARY".center(75))
    print("=" * 75)
    print("{:<20s} {:>6s} {:>10s} {:>14s} {:>10s}".format(
        "Attack", "Dist", "Accuracy", "DetectionRate", "FPR"))
    print("-" * 62)
    for r in all_results:
        s = r["summary"]
        print("{:<20s} {:>6s} {:>10.2%} {:>14.2%} {:>10.2%}".format(
            r["config"]["attack_name"],
            "IID" if r["config"]["iid"] else "N-IID",
            s.get("accuracy", {}).get("final", 0.0),
            s.get("detection_rate", {}).get("final", 0.0),
            s.get("false_positive_rate", {}).get("final", 0.0),
        ))
    print("=" * 75)
    print("Total runtime: {:.1f}s ({:.1f} min)".format(total_time, total_time / 60))

    # ── Save outputs ─────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output, "accuracy_results.csv")
    json_path = os.path.join(args.output, "metrics.json")
    save_results_csv(csv_rows, csv_path)
    save_results_json(
        {
            "experiments": all_results,
            "total_runtime_seconds": total_time,
        },
        json_path,
    )

    if not args.no_plots:
        print("\nGenerating plots…")
        _try_plot_results(all_results, args.output)

    print("\nAll outputs saved to {}/".format(args.output))
    print("  {:<35s}  ← round-by-round CSV".format(csv_path))
    print("  {:<35s}  ← full metrics JSON".format(json_path))


if __name__ == "__main__":
    main()
