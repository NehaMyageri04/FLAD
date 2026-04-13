"""
benchmark_qbad_vs_flad.py — Side-by-side comparison of QBAD-FL and FLAD.

Runs both the classical FLAD detector (LinearNet) and the quantum QBAD-FL
detector (VQC) on identical data and attack scenarios, then generates a
comparison table and exports results to CSV/JSON.

Usage
-----
# Compare for attacks 0, 1, 2, 5 with 5 communication rounds
python benchmark_qbad_vs_flad.py --attacks 0 1 2 5 --rounds 5

# Full side-by-side comparison (all 7 attacks)
python benchmark_qbad_vs_flad.py --full

# Custom experiment
python benchmark_qbad_vs_flad.py --attacks 5 --rounds 3 --clients 20 --byzantine 5
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FLAD_DIR = os.path.join(_THIS_DIR, "..", "1.FLAD")

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from metrics import (
    calculate_detection_rate,
    calculate_false_positive_rate,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    aggregate_round_metrics,
    compare_with_flad,
    format_comparison_table,
    save_results_json,
    save_results_csv,
    ATTACK_NAMES,
)

# ── Shared FL utilities ───────────────────────────────────────────────────────


def _cos(a, b):
    return np.sum(a * b.T) / (
        (np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9
    )


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


def _evaluate(net, testDataLoader, global_parameters, dev):
    net.load_state_dict(global_parameters, strict=True)
    sum_accu, num = 0, 0
    with torch.no_grad():
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = torch.argmax(net(data), dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
    return float(sum_accu / num)


def _collect_honest_uploads(honest_clients, myClients, cfg, net, loss_func, opti,
                             global_parameters):
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
    return uploads, honest_all_weight


def _collect_byzantine_uploads(byzantine_clients, myClients, pattern, honest_all_weight,
                                nc, cfg, net, loss_func, opti, global_parameters):
    import Attack as A
    uploads = []
    for cl in byzantine_clients:
        lp = {}
        if pattern == 0:
            for k in honest_all_weight:
                lp[k] = A.Gaussian_attack(honest_all_weight[k])
        elif pattern == 1:
            for k in honest_all_weight:
                lp[k] = A.Sign_flipping_attack(honest_all_weight[k])
        elif pattern == 2:
            byz = len(byzantine_clients)
            for k in honest_all_weight:
                lp[k] = A.ZeroGradient_attack(honest_all_weight[k], byz)
        elif pattern == 3:
            pc = A.backdoor_poisoning_data(myClients.clients_set[cl], cfg["data_name"])
            lp = pc.localTrain(cfg["epoch"], cfg["batchsize"], net, loss_func, opti,
                               global_parameters)
        elif pattern == 4:
            pc = A.model_replacement_attack_data(myClients.clients_set[cl], cfg["data_name"])
            lp = pc.localTrain(cfg["epoch"], cfg["batchsize"], net, loss_func, opti,
                               global_parameters)
            for k in lp:
                lp[k] = lp[k] * nc
        elif pattern == 5:
            for k in honest_all_weight:
                lp[k] = A.MPAF(honest_all_weight[k])
        elif pattern == 6:
            for k in honest_all_weight:
                lp[k] = A.AGR_agnostic(honest_all_weight[k])
        uploads.append(lp)
    return uploads


# ── FLAD detector (classical LinearNet) ───────────────────────────────────────


def _train_linear(weight_train, dimen, dev):
    """FLAD's original LinearNet-based detector."""
    # Import LinearNet from 1.FLAD if available, else define inline
    try:
        if _FLAD_DIR not in sys.path:
            sys.path.insert(0, _FLAD_DIR)
        from Models import LinearNet  # type: ignore
    except ImportError:
        import torch.nn as nn

        class LinearNet(nn.Module):
            def __init__(self, dimen):
                super().__init__()
                self.linear1 = nn.Linear(dimen, dimen // 10)
                self.linear2 = nn.Linear(dimen // 10, dimen // 100)
                self.linear3 = nn.Linear(dimen // 100, 1)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.tanh(self.linear1(x))
                x = torch.tanh(self.linear2(x))
                x = torch.sigmoid(self.linear3(x))
                return x ** 2

    weight_train = weight_train.view(weight_train.size(0), -1).to(dev)
    loader = DataLoader(dataset=weight_train, batch_size=1, shuffle=True)
    model = LinearNet(dimen).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction="sum")
    label = torch.tensor([[1.0]]).to(dev)
    for _ in range(20):
        for batch in loader:
            out = model(batch)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        all_out = model(weight_train)
    return model, all_out.mean(dim=0), all_out.max() - all_out.min()


def _flad_feature_extraction(Upload_Parameters, FC, Std, Dis, cfg, dev):
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
    feature[:, 0] = FC["conv1.weight"](k1.view(nc, -1)).cpu().detach().numpy().reshape(nc,)
    feature[:, 1] = FC["fc.weight"](w3.view(nc, -1)).cpu().detach().numpy().reshape(nc,)

    honest_std = np.array([Std["conv1.weight"].item(), Std["fc.weight"].item()])
    honest_L2 = np.sqrt(np.sum(honest_std * honest_std.T)) + 1e-9

    eps = ((Dis["conv1.weight"].item() ** 2 + Dis["fc.weight"].item() ** 2) ** 0.5)
    db = DBSCAN(eps=eps, min_samples=3).fit(feature)
    label_list = db.labels_

    temp = np.zeros([1, 2])
    label_score = {}
    for lbl in set(db.labels_):
        if lbl != -1:
            for c in range(nc):
                if label_list[c] == lbl:
                    temp = np.concatenate((temp, feature[c, :].reshape(1, 2)), axis=0)
            lm = temp.mean(axis=0) * temp.shape[0] / (temp.shape[0] - 1)
            cosin = _cos(lm, honest_std)
            length = abs(np.sqrt(np.sum(lm * lm.T)) / honest_L2 - 1.0)
            label_score[lbl] = cfg["alpha"] * cosin - (1 - cfg["alpha"]) * length

    honest_label = sorted(label_score.items(), key=lambda x: x[1], reverse=True)[0][0]
    return [c for c in range(nc) if label_list[c] != honest_label]


def _flad_feature_extraction_model(Central_par, cfg, dev):
    num = len(Central_par)
    if cfg["data_name"] == "mnist":
        k1 = torch.zeros(num, 10, 1, 5, 5)
        w3 = torch.zeros(num, 10, 320)
    else:
        k1 = torch.zeros(num, 64, 3, 3, 3)
        w3 = torch.zeros(num, 10, 512)
    for i, W in enumerate(Central_par):
        if cfg["data_name"] == "mnist":
            k1[i] = W["conv1.weight"].data
            w3[i] = W["fc.weight"].data
        else:
            k1[i] = W["module.conv1.weight"].data
            w3[i] = W["module.fc.weight"].data
    FC, Std, Dis = {}, {}, {}
    if cfg["data_name"] == "mnist":
        FC["conv1.weight"], Std["conv1.weight"], Dis["conv1.weight"] = _train_linear(k1, 10 * 1 * 5 * 5, dev)
        FC["fc.weight"], Std["fc.weight"], Dis["fc.weight"] = _train_linear(w3, 10 * 320, dev)
    else:
        FC["conv1.weight"], Std["conv1.weight"], Dis["conv1.weight"] = _train_linear(k1, 64 * 3 * 3 * 3, dev)
        FC["fc.weight"], Std["fc.weight"], Dis["fc.weight"] = _train_linear(w3, 10 * 512, dev)
    return FC, Std, Dis


# ── QBAD-FL detector (VQC) ────────────────────────────────────────────────────


def _train_vqc(weight_train, dimen, dev):
    from Models import QuantumByzantineDetector
    weight_train = weight_train.view(weight_train.size(0), -1).cpu()
    loader = DataLoader(dataset=weight_train, batch_size=1, shuffle=True)
    model = QuantumByzantineDetector(dimen)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction="sum")
    label = torch.tensor([[1.0]])
    for _ in range(20):
        for batch in loader:
            out = model(batch)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        all_out = model(weight_train)
    return model, all_out.mean(dim=0), all_out.max() - all_out.min()


def _qbad_feature_extraction_model(Central_par, cfg, dev):
    num = len(Central_par)
    if cfg["data_name"] == "mnist":
        k1 = torch.zeros(num, 10, 1, 5, 5)
        w3 = torch.zeros(num, 10, 320)
    else:
        k1 = torch.zeros(num, 64, 3, 3, 3)
        w3 = torch.zeros(num, 10, 512)
    for i, W in enumerate(Central_par):
        if cfg["data_name"] == "mnist":
            k1[i] = W["conv1.weight"].data
            w3[i] = W["fc.weight"].data
        else:
            k1[i] = W["module.conv1.weight"].data
            w3[i] = W["module.fc.weight"].data
    FC, Std, Dis = {}, {}, {}
    if cfg["data_name"] == "mnist":
        FC["conv1.weight"], Std["conv1.weight"], Dis["conv1.weight"] = _train_vqc(k1, 10 * 1 * 5 * 5, dev)
        FC["fc.weight"], Std["fc.weight"], Dis["fc.weight"] = _train_vqc(w3, 10 * 320, dev)
    else:
        FC["conv1.weight"], Std["conv1.weight"], Dis["conv1.weight"] = _train_vqc(k1, 64 * 3 * 3 * 3, dev)
        FC["fc.weight"], Std["fc.weight"], Dis["fc.weight"] = _train_vqc(w3, 10 * 512, dev)
    return FC, Std, Dis


def _qbad_feature_extraction(Upload_Parameters, FC, Std, Dis, cfg, dev):
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
    feature[:, 0] = FC["conv1.weight"](k1.view(nc, -1)).cpu().detach().numpy().reshape(nc,)
    feature[:, 1] = FC["fc.weight"](w3.view(nc, -1)).cpu().detach().numpy().reshape(nc,)
    honest_std = np.array([Std["conv1.weight"].item(), Std["fc.weight"].item()])
    honest_L2 = np.sqrt(np.sum(honest_std * honest_std.T)) + 1e-9
    eps = ((Dis["conv1.weight"].item() ** 2 + Dis["fc.weight"].item() ** 2) ** 0.5)
    db = DBSCAN(eps=eps, min_samples=3).fit(feature)
    label_list = db.labels_
    temp = np.zeros([1, 2])
    label_score = {}
    for lbl in set(db.labels_):
        if lbl != -1:
            for c in range(nc):
                if label_list[c] == lbl:
                    temp = np.concatenate((temp, feature[c, :].reshape(1, 2)), axis=0)
            lm = temp.mean(axis=0) * temp.shape[0] / (temp.shape[0] - 1)
            cosin = _cos(lm, honest_std)
            length = abs(np.sqrt(np.sum(lm * lm.T)) / honest_L2 - 1.0)
            label_score[lbl] = cfg["alpha"] * cosin - (1 - cfg["alpha"]) * length
    honest_label = sorted(label_score.items(), key=lambda x: x[1], reverse=True)[0][0]
    return [c for c in range(nc) if label_list[c] != honest_label]


# ── Single experiment runner ──────────────────────────────────────────────────


def _run_one(cfg, mode, myClients, net, loss_func, opti, testDataLoader,
             actual_malicious_indices, verbose):
    """Run FL experiment with specified detector (flad or qbad).

    Returns list of per-round metric dicts.
    """
    from Models import Mnist_CNN, ResNet18

    dev = torch.device("cpu")
    nc = cfg["num_of_clients"]
    byz = cfg["byzantine_size"]
    honest_clients = ["client{}".format(i) for i in range(nc - byz)]
    byzantine_clients = ["client{}".format(i) for i in range(nc - byz, nc)]

    global_parameters = {k: v.clone() for k, v in net.state_dict().items()}
    round_results = []

    feature_model_fn = _qbad_feature_extraction_model if mode == "qbad" else _flad_feature_extraction_model
    feature_extract_fn = _qbad_feature_extraction if mode == "qbad" else _flad_feature_extraction

    for rnd in range(cfg["num_comm"]):
        rnd_start = time.time()

        Central_par = myClients.centralTrain(
            cfg["epoch"], cfg["batchsize"], net, loss_func, opti, global_parameters
        )
        FC, Std, Dis = feature_model_fn(Central_par, cfg, dev)

        uploads, honest_all_weight = _collect_honest_uploads(
            honest_clients, myClients, cfg, net, loss_func, opti, global_parameters
        )
        byz_ups = _collect_byzantine_uploads(
            byzantine_clients, myClients, cfg["pattern"], honest_all_weight,
            nc, cfg, net, loss_func, opti, global_parameters
        )
        uploads.extend(byz_ups)

        detected = feature_extract_fn(uploads, FC, Std, Dis, cfg, dev)
        global_parameters = _fed_avg(list(uploads), detected)

        acc = _evaluate(net, testDataLoader, global_parameters, dev)
        dr = calculate_detection_rate(detected, actual_malicious_indices)
        fpr = calculate_false_positive_rate(detected, actual_malicious_indices, nc)
        prec = calculate_precision(detected, actual_malicious_indices)
        rec = calculate_recall(detected, actual_malicious_indices)
        f1 = calculate_f1(detected, actual_malicious_indices)
        rnd_time = time.time() - rnd_start

        round_results.append({
            "round": rnd + 1,
            "accuracy": acc,
            "detection_rate": dr,
            "false_positive_rate": fpr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "round_time_seconds": rnd_time,
        })
        if verbose:
            print("    [{}] R{}: acc={:.2%} DR={:.2%} FPR={:.2%} t={:.1f}s".format(
                mode.upper(), rnd + 1, acc, dr, fpr, rnd_time))

    return round_results


# ── Benchmark runner ──────────────────────────────────────────────────────────


def run_benchmark(cfg, verbose=True):
    """Run side-by-side FLAD vs QBAD-FL experiment for one attack pattern.

    Returns dict with keys flad_results, qbad_results, comparison, attack_name.
    """
    from Models import Mnist_CNN, ResNet18
    from clients import ClientsGroup

    dev = torch.device("cpu")
    nc, byz = cfg["num_of_clients"], cfg["byzantine_size"]

    if cfg["data_name"] == "mnist":
        net = Mnist_CNN()
    else:
        net = ResNet18()
    net = net.to(dev)
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=cfg["learning_rate"])

    myClients = ClientsGroup(cfg["data_name"], cfg["IID"], nc, dev)
    myClients.get_central_data(cfg["central_data_size"], cfg["central_data_pro"])
    testDataLoader = myClients.test_data_loader
    actual_malicious_indices = list(range(nc - byz, nc))

    attack_name = ATTACK_NAMES.get(cfg["pattern"], "Unknown")
    if verbose:
        print("\n  Running FLAD (classical)  …")
    t0 = time.time()
    flad_rounds = _run_one(cfg, "flad", myClients, net, loss_func, opti,
                            testDataLoader, actual_malicious_indices, verbose)
    flad_time = time.time() - t0

    if verbose:
        print("  Running QBAD-FL (quantum) …")
    t0 = time.time()
    qbad_rounds = _run_one(cfg, "qbad", myClients, net, loss_func, opti,
                            testDataLoader, actual_malicious_indices, verbose)
    qbad_time = time.time() - t0

    flad_summary = aggregate_round_metrics(flad_rounds)
    qbad_summary = aggregate_round_metrics(qbad_rounds)

    flad_results = {
        "config": dict(cfg),
        "round_results": flad_rounds,
        "summary": flad_summary,
        "total_runtime_seconds": flad_time,
    }
    qbad_results = {
        "config": dict(cfg),
        "round_results": qbad_rounds,
        "summary": qbad_summary,
        "total_runtime_seconds": qbad_time,
    }
    comparison = compare_with_flad(qbad_results, flad_results)

    return {
        "attack_name": attack_name,
        "attack_pattern": cfg["pattern"],
        "flad_results": flad_results,
        "qbad_results": qbad_results,
        "comparison": comparison,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark QBAD-FL vs FLAD side-by-side",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar_10"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--byzantine", type=int, default=5)
    parser.add_argument("--attacks", nargs="+", type=int, default=[5],
                        help="Attack patterns to test (0-6)")
    parser.add_argument("--full", action="store_true",
                        help="Test all 7 attack types")
    parser.add_argument("--output", default="results",
                        help="Output directory for CSV/JSON files")
    args = parser.parse_args()

    if args.full:
        args.attacks = list(range(7))

    cfg = {
        "data_name": args.dataset,
        "num_of_clients": args.clients,
        "byzantine_size": args.byzantine,
        "epoch": 5,
        "batchsize": 64,
        "learning_rate": 0.1,
        "num_comm": args.rounds,
        "IID": True,
        "central_data_size": 300,
        "central_data_pro": 0.1,
        "alpha": 0.5,
    }

    os.makedirs(args.output, exist_ok=True)
    all_benchmarks = []
    csv_rows = []

    for attack in args.attacks:
        cfg["pattern"] = attack
        attack_name = ATTACK_NAMES.get(attack, "?")
        print("\n" + "=" * 65)
        print("Attack {}: {}".format(attack, attack_name))
        print("=" * 65)

        result = run_benchmark(cfg, verbose=True)
        all_benchmarks.append(result)

        print(format_comparison_table(result["comparison"], attack_name=attack_name))

        f_acc = result["flad_results"]["summary"].get("accuracy", {}).get("final", 0.0)
        q_acc = result["qbad_results"]["summary"].get("accuracy", {}).get("final", 0.0)
        f_dr = result["flad_results"]["summary"].get("detection_rate", {}).get("final", 0.0)
        q_dr = result["qbad_results"]["summary"].get("detection_rate", {}).get("final", 0.0)
        f_fpr = result["flad_results"]["summary"].get("false_positive_rate", {}).get("final", 0.0)
        q_fpr = result["qbad_results"]["summary"].get("false_positive_rate", {}).get("final", 0.0)

        csv_rows.append({
            "attack_type": attack,
            "attack_name": attack_name,
            "flad_accuracy": round(f_acc, 4),
            "qbad_accuracy": round(q_acc, 4),
            "accuracy_diff": round(q_acc - f_acc, 4),
            "flad_detection_rate": round(f_dr, 4),
            "qbad_detection_rate": round(q_dr, 4),
            "flad_fpr": round(f_fpr, 4),
            "qbad_fpr": round(q_fpr, 4),
            "flad_runtime_s": round(result["flad_results"]["total_runtime_seconds"], 1),
            "qbad_runtime_s": round(result["qbad_results"]["total_runtime_seconds"], 1),
        })

    # Summary table
    print("\n\n" + "=" * 75)
    print("BENCHMARK SUMMARY: FLAD vs QBAD-FL".center(75))
    print("=" * 75)
    print("{:<18s} {:>8s} {:>10s} {:>8s} {:>10s} {:>10s}".format(
        "Attack", "FLAD Acc", "QBAD Acc", "Diff", "FLAD DR", "QBAD DR"))
    print("-" * 68)
    for row in csv_rows:
        diff_str = "{:+.2%}".format(row["accuracy_diff"])
        print("{:<18s} {:>8.2%} {:>10.2%} {:>8s} {:>10.2%} {:>10.2%}".format(
            row["attack_name"],
            row["flad_accuracy"],
            row["qbad_accuracy"],
            diff_str,
            row["flad_detection_rate"],
            row["qbad_detection_rate"],
        ))
    print("=" * 75)

    # Save outputs
    csv_path = os.path.join(args.output, "benchmark_comparison.csv")
    json_path = os.path.join(args.output, "benchmark_results.json")
    save_results_csv(csv_rows, csv_path)
    save_results_json({"benchmarks": all_benchmarks}, json_path)
    print("\nOutputs saved to {}/".format(args.output))


if __name__ == "__main__":
    main()
