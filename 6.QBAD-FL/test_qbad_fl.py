"""
test_qbad_fl.py — Quick validation test for QBAD-FL.

Verifies that the Variational Quantum Circuit (VQC) Byzantine detector works
correctly against one or more attack types with a small-scale experiment.

Usage
-----
# Level 1: Quickest possible check (~5 min, 1 attack, 10 clients, 2 rounds)
python test_qbad_fl.py --quick

# Level 2: Test specific attack patterns (~15 min)
python test_qbad_fl.py --attacks 0 1 2 5

# Level 3: Custom experiment
python test_qbad_fl.py --dataset mnist --rounds 3 --clients 20 --byzantine 5
"""

import argparse
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN

# Ensure we can import from the 6.QBAD-FL directory itself
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

# ── Helpers (mirror 6.QBAD-FL/main.py without global args) ───────────────────


def cos(a, b):
    return np.sum(a * b.T) / (
        (np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9
    )


def train_vqc(weight_train, dimen, dev):
    """Train QuantumByzantineDetector on server central weights."""
    weight_train = weight_train.view(weight_train.size(0), -1).cpu()
    loader = DataLoader(dataset=weight_train, batch_size=1, shuffle=True)
    model = QuantumByzantineDetector(dimen)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction="sum")
    label = torch.tensor([[1.0]])
    for _ in range(20):
        for batch in loader:
            output = model(batch)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        all_output = model(weight_train)
    std = all_output.mean(dim=0)
    dis = all_output.max() - all_output.min()
    return model, std, dis


def feature_extraction_model(Central_par, cfg, dev):
    num = len(Central_par)
    if cfg["data_name"] == "mnist":
        k1 = torch.zeros(num, 10, 1, 5, 5)
        w3 = torch.zeros(num, 10, 320)
    else:
        k1 = torch.zeros(num, 64, 3, 3, 3)
        w3 = torch.zeros(num, 10, 512)

    for i, W in enumerate(Central_par):
        if cfg["data_name"] == "cifar_10":
            k1[i] = W["module.conv1.weight"].data
            w3[i] = W["module.fc.weight"].data
        else:
            k1[i] = W["conv1.weight"].data
            w3[i] = W["fc.weight"].data

    FC, Std, Dis = {}, {}, {}
    if cfg["data_name"] == "mnist":
        FC["conv1.weight"], Std["conv1.weight"], Dis["conv1.weight"] = train_vqc(k1, 10 * 1 * 5 * 5, dev)
        FC["fc.weight"], Std["fc.weight"], Dis["fc.weight"] = train_vqc(w3, 10 * 320, dev)
    else:
        FC["conv1.weight"], Std["conv1.weight"], Dis["conv1.weight"] = train_vqc(k1, 64 * 3 * 3 * 3, dev)
        FC["fc.weight"], Std["fc.weight"], Dis["fc.weight"] = train_vqc(w3, 10 * 512, dev)
    return FC, Std, Dis


def vqc_feature_extraction(Upload_Parameters, FC, Std, Dis, cfg, dev):
    nc = cfg["num_of_clients"]
    if cfg["data_name"] == "mnist":
        k1 = torch.zeros(nc, 10, 1, 5, 5).to(dev)
        w3 = torch.zeros(nc, 10, 320).to(dev)
    else:
        k1 = torch.zeros(nc, 64, 3, 3, 3).to(dev)
        w3 = torch.zeros(nc, 10, 512).to(dev)

    for i, W in enumerate(Upload_Parameters):
        if cfg["data_name"] == "mnist":
            k1[i] = W["conv1.weight"].data
            w3[i] = W["fc.weight"].data
        else:
            k1[i] = W["module.conv1.weight"].data
            w3[i] = W["module.fc.weight"].data

    feature = np.zeros([nc, 2])
    feature[:, 0] = (
        FC["conv1.weight"](k1.view(nc, -1)).cpu().detach().numpy().reshape(nc,)
    )
    feature[:, 1] = (
        FC["fc.weight"](w3.view(nc, -1)).cpu().detach().numpy().reshape(nc,)
    )

    if np.isnan(feature).any():
        print("  [Warning] NaN values detected in VQC features - replacing with column means")
        col_mean = np.nan_to_num(np.nanmean(feature, axis=0), nan=0.0)
        feature = np.where(np.isnan(feature), col_mean, feature)

    honest_std = np.array([Std["conv1.weight"].item(), Std["fc.weight"].item()])
    honest_L2 = np.sqrt(np.sum(honest_std * honest_std.T)) + 1e-9

    eps1 = Dis["conv1.weight"].item()
    eps3 = Dis["fc.weight"].item()
    eps = (eps1 ** 2 + eps3 ** 2) ** 0.5

    db = DBSCAN(eps=eps, min_samples=3).fit(feature)
    label_list = db.labels_

    temp = np.zeros([1, 2])
    label_mean, label_score = {}, {}
    for lbl in set(db.labels_):
        if lbl != -1:
            for c in range(nc):
                if label_list[c] == lbl:
                    temp = np.concatenate((temp, feature[c, :].reshape(1, 2)), axis=0)
            label_mean[lbl] = temp.mean(axis=0) * temp.shape[0] / (temp.shape[0] - 1)
            cosin = cos(label_mean[lbl], honest_std)
            length = abs(
                np.sqrt(np.sum(label_mean[lbl] * label_mean[lbl].T)) / honest_L2 - 1.0
            )
            label_score[lbl] = cfg["alpha"] * cosin - (1 - cfg["alpha"]) * length

    label_score = sorted(label_score.items(), key=lambda x: x[1], reverse=True)
    honest_label = label_score[0][0]
    malicious = [c for c in range(nc) if label_list[c] != honest_label]
    return malicious


def fed_avg(Upload_Parameters, malicious):
    params = list(Upload_Parameters)
    for j, idx in enumerate(sorted(malicious)):
        del params[idx - j]
    total = len(params)
    agg = None
    for p in params:
        if agg is None:
            agg = {k: v.clone() for k, v in p.items()}
        else:
            for k in p:
                agg[k] += p[k]
    for k in agg:
        agg[k] /= total
    return agg


def collect_attack_updates(byzantine_clients, myClients, pattern, honest_all_weight,
                            num_clients, epoch, batchsize, net, loss_func, opti,
                            global_parameters, data_name="mnist"):
    uploads = []
    for client in byzantine_clients:
        lp = {}
        if pattern == 0:
            for key in honest_all_weight:
                lp[key] = Attack.Gaussian_attack(honest_all_weight[key])
        elif pattern == 1:
            for key in honest_all_weight:
                lp[key] = Attack.Sign_flipping_attack(honest_all_weight[key])
        elif pattern == 2:
            byz = sum(1 for _ in byzantine_clients)
            for key in honest_all_weight:
                lp[key] = Attack.ZeroGradient_attack(honest_all_weight[key], byz)
        elif pattern == 3:
            pc = Attack.backdoor_poisoning_data(myClients.clients_set[client],
                                                data_name)
            lp = pc.localTrain(epoch, batchsize, net, loss_func, opti, global_parameters)
        elif pattern == 4:
            pc = Attack.model_replacement_attack_data(myClients.clients_set[client],
                                                      data_name)
            lp = pc.localTrain(epoch, batchsize, net, loss_func, opti, global_parameters)
            for key in lp:
                lp[key] = lp[key] * num_clients
        elif pattern == 5:
            for key in honest_all_weight:
                lp[key] = Attack.MPAF(honest_all_weight[key])
        elif pattern == 6:
            for key in honest_all_weight:
                lp[key] = Attack.AGR_agnostic(honest_all_weight[key])
        uploads.append(lp)
    return uploads


# ── Core experiment runner ─────────────────────────────────────────────────────


def run_experiment(cfg, verbose=True):
    """Run one QBAD-FL experiment and return round-by-round results.

    Parameters
    ----------
    cfg : dict  Experiment configuration (see _default_config()).
    verbose : bool  Print progress to stdout.

    Returns
    -------
    dict with keys: config, round_results, summary, total_runtime_seconds.
    """
    dev = torch.device("cpu")  # CPU-only for test scripts

    net = Mnist_CNN() if cfg["data_name"] == "mnist" else ResNet18()
    net = net.to(dev)
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=cfg["learning_rate"])

    myClients = ClientsGroup(cfg["data_name"], cfg["IID"],
                             cfg["num_of_clients"], dev)
    myClients.get_central_data(cfg["central_data_size"], cfg["central_data_pro"])
    testDataLoader = myClients.test_data_loader

    nc = cfg["num_of_clients"]
    byz = cfg["byzantine_size"]
    honest_clients = ["client{}".format(i) for i in range(nc - byz)]
    byzantine_clients = ["client{}".format(i) for i in range(nc - byz, nc)]
    actual_malicious_indices = list(range(nc - byz, nc))

    global_parameters = {k: v.clone() for k, v in net.state_dict().items()}

    round_results = []
    start = time.time()

    for rnd in range(cfg["num_comm"]):
        rnd_start = time.time()
        if verbose:
            print("\n─── Round {}/{} ───".format(rnd + 1, cfg["num_comm"]))

        Central_par = myClients.centralTrain(
            cfg["epoch"], cfg["batchsize"], net, loss_func, opti, global_parameters
        )
        FC, Std, Dis = feature_extraction_model(Central_par, cfg, dev)

        Upload_Parameters = []
        honest_all_weight = None

        for cl in honest_clients:
            lp = myClients.clients_set[cl].localTrain(
                cfg["epoch"], cfg["batchsize"], net, loss_func, opti, global_parameters
            )
            Upload_Parameters.append(lp)
            if cfg["pattern"] <= 2 or cfg["pattern"] >= 5:
                if honest_all_weight is None:
                    honest_all_weight = {k: v.clone().unsqueeze(0) for k, v in lp.items()}
                else:
                    for k in lp:
                        honest_all_weight[k] = torch.cat(
                            [honest_all_weight[k], lp[k].unsqueeze(0)], dim=0
                        )

        byz_uploads = collect_attack_updates(
            byzantine_clients, myClients, cfg["pattern"], honest_all_weight,
            nc, cfg["epoch"], cfg["batchsize"], net, loss_func, opti, global_parameters,
            data_name=cfg["data_name"]
        )
        Upload_Parameters.extend(byz_uploads)

        detected = vqc_feature_extraction(Upload_Parameters, FC, Std, Dis, cfg, dev)

        global_parameters = fed_avg(list(Upload_Parameters), detected)

        # Evaluate model
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
            print(
                "  accuracy={:.2%}  detection_rate={:.2%}  "
                "FPR={:.2%}  F1={:.4f}  time={:.1f}s".format(
                    acc, dr, fpr, f1, rnd_time
                )
            )
            print("  Detected malicious: {}".format(detected))
            print("  Actual  malicious : {}".format(actual_malicious_indices))

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


# ── Default configuration factory ────────────────────────────────────────────


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
    }


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="QBAD-FL quick validation test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar_10"])
    parser.add_argument("--rounds", type=int, default=3, help="Communication rounds")
    parser.add_argument("--clients", type=int, default=20, help="Total clients")
    parser.add_argument("--byzantine", type=int, default=5, help="Byzantine clients")
    parser.add_argument(
        "--attacks",
        nargs="+",
        type=int,
        default=[5],
        help="Attack pattern(s) to test: 0=Gaussian 1=Sign-flip 2=Zero-gradient "
             "3=Backdoor 4=Model-replacement 5=MPAF 6=AGR-agnostic",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Minimal test: 10 clients, 5 byzantine, 2 rounds, 1 attack (MPAF)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results (e.g. results/quick_test.json)",
    )
    args = parser.parse_args()

    if args.quick:
        args.clients = 10
        args.byzantine = 5
        args.rounds = 2
        args.attacks = [5]
        print("Quick mode: 10 clients, 5 byzantine, 2 rounds, MPAF attack")

    cfg = _default_config()
    cfg["data_name"] = args.dataset
    cfg["num_comm"] = args.rounds
    cfg["num_of_clients"] = args.clients
    cfg["byzantine_size"] = args.byzantine

    all_results = []

    for attack in args.attacks:
        cfg["pattern"] = attack
        print("\n" + "=" * 65)
        print("Attack: {} ({})".format(attack, ATTACK_NAMES.get(attack, "?")))
        print("=" * 65)
        results = run_experiment(cfg, verbose=True)
        all_results.append(results)
        print(generate_report(results, title="QBAD-FL — Attack {}".format(attack)))

    # Multi-attack summary
    if len(all_results) > 1:
        print("\n\n" + "=" * 65)
        print("Multi-Attack Summary".center(65))
        print("=" * 65)
        print(
            "{:<20s} {:>10s} {:>14s} {:>10s}".format(
                "Attack", "Accuracy", "DetectionRate", "FPR"
            )
        )
        print("-" * 55)
        for r in all_results:
            s = r["summary"]
            print(
                "{:<20s} {:>10.2%} {:>14.2%} {:>10.2%}".format(
                    r["config"]["attack_name"],
                    s.get("accuracy", {}).get("final", 0.0),
                    s.get("detection_rate", {}).get("final", 0.0),
                    s.get("false_positive_rate", {}).get("final", 0.0),
                )
            )
        print("=" * 65)

    if args.output:
        save_results_json(
            {"experiments": all_results},
            args.output,
        )


if __name__ == "__main__":
    main()
