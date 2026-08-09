#!/usr/bin/env python
"""CAID (Critical Assessment of Intrinsic Disorder) benchmark for the BFN disorder head.

Evaluates per-residue disorder prediction against the CAID2/CAID3 reference
sets (FASTA sequence + per-residue binary/continuous disorder labels) and
reports AUC-ROC / AUC-PR, alongside reference methods (IUPred3, AF2-pLDDT,
ESMFold) where their per-residue outputs are available.

Pipeline per target:
  sequence  --[structure backend]-->  PDB  --predict_disorder()-->  per-residue scores
  compare scores vs labels → AUC-ROC / AUC-PR / MCC.

Because `predict_disorder` (modules/idp_disorder_analysis.py) takes a PDB and
runs the BFN disorder head on the backbone, a pure-sequence CAID target needs
a structure first. Configure via --structure-backend {af2, esmfold, skip}:
  - af2     : JAX-AF2 single-chain (reuses modules/af2_jax_runner.py)
  - esmfold : ESMFold (requires esm package)
  - skip    : only score targets for which a precomputed PDB is supplied
              (folded-domain targets); used to isolate the disorder-head AUC
              from the structure-prediction step.

Usage:
  python scripts/benchmark_caid.py --caid-dir data/caid2 --out results/ablation/caid.json
  python scripts/benchmark_caid.py --self-test          # tiny embedded case, no GPU
  python scripts/benchmark_caid.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_caid_reference(caid_dir, cluster_manifest=None):
    """Load CAID reference set.

    Expected layout:
      caid_dir/sequences.fasta   — '>target_id\\nSEQUENCE\\n' pairs
      caid_dir/labels/<id>.label — per-residue disorder labels:
                                    '1'/'0' (binary) or float in [0,1].

    Returns: list of dicts {id, sequence, labels: list[float]}.
    """
    seq_path = os.path.join(caid_dir, "sequences.fasta")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"CAID sequences.fasta not found at {seq_path}")

    targets = []
    cur_id, cur_seq = None, []
    with open(seq_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_id is not None:
                    targets.append({"id": cur_id, "sequence": "".join(cur_seq), "labels": None})
                cur_id = line[1:].split()[0]
                cur_seq = []
            elif line:
                cur_seq.append(line)
    if cur_id is not None:
        targets.append({"id": cur_id, "sequence": "".join(cur_seq), "labels": None})

    labels_dir = os.path.join(caid_dir, "labels")
    for t in targets:
        for ext in (".label", ".txt"):
            lp = os.path.join(labels_dir, t["id"] + ext)
            if os.path.exists(lp):
                with open(lp) as f:
                    vals = []
                    for tok in f.read().split():
                        try:
                            vals.append(float(tok))
                        except ValueError:
                            vals.append(float("nan"))
                t["labels"] = vals
                break
    # Drop targets with missing/mismatched labels.
    kept = [
        t for t in targets if t["labels"] is not None and len(t["labels"]) == len(t["sequence"])
    ]
    if cluster_manifest is not None:
        with open(cluster_manifest, encoding="utf-8") as handle:
            clusters = json.load(handle)
        missing = sorted(target["id"] for target in kept if target["id"] not in clusters)
        if missing:
            raise ValueError(f"CAID cluster manifest missing target IDs: {missing[:10]}")
        for target in kept:
            target["cluster_id"] = str(clusters[target["id"]])
    return kept


def assert_external_cluster_isolation(targets, training_clusters):
    """Reject external targets sharing a supplied homology cluster with training."""
    train = {str(cluster) for cluster in training_clusters}
    missing = [target["id"] for target in targets if not target.get("cluster_id")]
    if missing:
        raise ValueError(f"External targets lack cluster IDs: {missing[:10]}")
    overlap = sorted({target["cluster_id"] for target in targets} & train)
    if overlap:
        raise ValueError(f"Training/external homology-cluster leakage: {overlap[:10]}")


def assert_scored_clusters_resolved(targets, work_dir, structure_backend):
    """Reject cached targets whose homology cluster could not be resolved."""
    if structure_backend != "skip":
        return
    unresolved = [
        target["id"]
        for target in targets
        if str(target.get("cluster_id", "")).startswith("unresolved:")
        and os.path.exists(os.path.join(work_dir, target["id"] + ".pdb"))
    ]
    if unresolved:
        raise ValueError(f"Scored CAID targets have unresolved clusters: {unresolved[:10]}")


# ── AUC (self-contained, no sklearn dependency) ──


def _binary_labels(labels, threshold=0.5):
    return [1 if float(l) >= threshold else 0 for l in labels]


def auc_roc(scores, labels):
    """Mann-Whitney U formulation of AUC-ROC."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc_pr(scores, labels):
    """Step-wise AUC-PR (average precision)."""
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels, strict=True), key=lambda x: -x[0])
    tp, fp = 0, 0
    ap = 0.0
    prev_recall = 0.0
    for _, l in pairs:
        if l == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def binary_classification_metrics(scores, labels, threshold=0.5, calibration_bins=10):
    """Threshold and calibration metrics for finite probabilities."""
    predictions = [1 if float(score) >= threshold else 0 for score in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels, strict=True))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels, strict=True))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels, strict=True))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    brier = sum(
        (float(score) - label) ** 2 for score, label in zip(scores, labels, strict=True)
    ) / len(labels)
    ece = 0.0
    for index in range(calibration_bins):
        low, high = index / calibration_bins, (index + 1) / calibration_bins
        members = [
            i
            for i, score in enumerate(scores)
            if low <= float(score) < high or (index == calibration_bins - 1 and float(score) == 1)
        ]
        if members:
            confidence = sum(float(scores[i]) for i in members) / len(members)
            frequency = sum(labels[i] for i in members) / len(members)
            ece += len(members) / len(labels) * abs(confidence - frequency)
    return {
        "f1": f1,
        "mcc": mcc,
        "brier": brier,
        "ece": ece,
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(scores, labels, label_threshold=0.5, prediction_threshold=0.5):
    """Return discrimination, threshold, and calibration metrics for one target."""
    import math

    if len(scores) != len(labels):
        raise ValueError(f"Score/label length mismatch: {len(scores)} != {len(labels)}")
    defined = [
        (score, label)
        for score, label in zip(scores, labels, strict=True)
        if math.isfinite(float(label))
    ]
    filtered_scores = [score for score, _ in defined]
    bin_labels = _binary_labels([label for _, label in defined], label_threshold)
    if any(not 0 <= float(score) <= 1 for score in filtered_scores):
        raise ValueError("Disorder scores must be probabilities in [0, 1]")
    return {
        "auc_roc": auc_roc(filtered_scores, bin_labels),
        "auc_pr": auc_pr(filtered_scores, bin_labels),
        **binary_classification_metrics(
            filtered_scores, bin_labels, threshold=prediction_threshold
        ),
        "n_residues": len(defined),
        "n_undefined": len(labels) - len(defined),
    }


def _structure_for_sequence(seq, target_id, backend, work_dir, device):
    """Produce a PDB for a pure-sequence CAID target. Returns path or None."""
    sys.path.insert(0, os.path.join(ROOT, "modules"))
    out_pdb = os.path.join(work_dir, target_id + ".pdb")
    if os.path.exists(out_pdb):
        return out_pdb
    if backend == "skip":
        return None
    if backend == "esmfold":
        try:
            from esm.pretrained import esmfold_v1  # noqa: F401

            # NOTE: full ESMFold wiring left as a thin shim; install esm to enable.
            return _esmfold_to_pdb(seq, out_pdb)
        except Exception as e:
            print(f"  [esmfold] unavailable for {target_id}: {e}")
            return None
    if backend == "af2":
        try:
            from af2_jax_runner import run_multimer_prediction

            res = run_multimer_prediction(seq, "", num_recycle=0)
            if res.get("success") and res.get("pdb"):
                with open(out_pdb, "w") as f:
                    f.write(res["pdb"])
                return out_pdb
        except Exception as e:
            print(f"  [af2] unavailable for {target_id}: {e}")
    return None


def _esmfold_to_pdb(seq, out_pdb):
    """ESMFold single-chain prediction → PDB (thin shim; needs esm package)."""
    raise NotImplementedError("ESMFold backend requires the esm package and model weights.")


def load_checkpoint(path, device):
    """Strictly load the authoritative EMA checkpoint used by the benchmark."""
    import torch

    sys.path.insert(0, ROOT)
    from disorderflow.models import get_model

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("weights_kind") != "ema":
        raise RuntimeError("Authoritative CAID evaluation requires an EMA best checkpoint")
    model = get_model(checkpoint["config"].model).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint["config"]


def predict_scores_for_pdb(model, config, pdb_path, chain, device, calibration=None):
    """Run the BFN disorder head on a structure → per-residue scores."""
    sys.path.insert(0, os.path.join(ROOT, "modules"))
    from idp_disorder_analysis import predict_disorder

    result = predict_disorder(model, config, pdb_path, chain, device, calibration)
    return list(result["disorder_scores"])


def run_benchmark(
    caid_dir,
    out_path,
    structure_backend,
    max_targets,
    device,
    work_dir,
    checkpoint,
    cluster_manifest=None,
    training_clusters=None,
    calibration=None,
    eligibility_audit=None,
):
    targets = load_caid_reference(caid_dir, cluster_manifest)
    if eligibility_audit is not None:
        with open(eligibility_audit, encoding="utf-8") as handle:
            eligibility = json.load(handle)
            allowed = set(eligibility.get("eligible_ids", eligibility["independent_ids"]))
        targets = [target for target in targets if target["id"] in allowed]
    if training_clusters is not None:
        assert_external_cluster_isolation(targets, training_clusters)
        assert_scored_clusters_resolved(targets, work_dir, structure_backend)
    print(f"Loaded {len(targets)} CAID targets from {caid_dir}")
    if max_targets:
        targets = targets[:max_targets]

    os.makedirs(work_dir, exist_ok=True)
    results = []
    aucs_roc, aucs_pr, f1s, mccs, briers, eces = [], [], [], [], [], []
    pooled_scores, pooled_labels = [], []
    loaded = None
    prediction_threshold = 0.5
    calibration_metadata = None
    if calibration is not None:
        with open(calibration, encoding="utf-8") as handle:
            calibration_metadata = json.load(handle)
        prediction_threshold = float(calibration_metadata["threshold"])

    for i, t in enumerate(targets):
        print(f"[{i + 1}/{len(targets)}] {t['id']} (L={len(t['sequence'])})")
        pdb = _structure_for_sequence(t["sequence"], t["id"], structure_backend, work_dir, device)
        if pdb is None:
            print("  skip (no structure available; try --structure-backend af2)")
            continue
        try:
            if loaded is None:
                loaded = load_checkpoint(checkpoint, device)
            scores = predict_scores_for_pdb(*loaded, pdb, "A", device, calibration=calibration)
        except Exception as e:  # noqa: BLE001
            print(f"  predict_disorder failed: {e}")
            continue
        ev = evaluate(scores, t["labels"], prediction_threshold=prediction_threshold)
        print(f"  AUC-ROC={ev['auc_roc']:.3f}  AUC-PR={ev['auc_pr']:.3f}")
        serialized = {
            key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in ev.items()
        }
        results.append({"id": t["id"], "cluster_id": t.get("cluster_id"), **serialized})
        for score, label in zip(scores, t["labels"], strict=True):
            if math.isfinite(float(label)):
                pooled_scores.append(float(score))
                pooled_labels.append(1 if float(label) >= 0.5 else 0)
        if ev["auc_roc"] == ev["auc_roc"]:  # not nan
            aucs_roc.append(ev["auc_roc"])
        if ev["auc_pr"] == ev["auc_pr"]:
            aucs_pr.append(ev["auc_pr"])
        f1s.append(ev["f1"])
        mccs.append(ev["mcc"])
        briers.append(ev["brier"])
        eces.append(ev["ece"])

    summary = {
        "n_targets": len(results),
        "mean_auc_roc": sum(aucs_roc) / len(aucs_roc) if aucs_roc else None,
        "mean_auc_pr": sum(aucs_pr) / len(aucs_pr) if aucs_pr else None,
        "mean_f1": sum(f1s) / len(f1s) if f1s else None,
        "mean_mcc": sum(mccs) / len(mccs) if mccs else None,
        "mean_brier": sum(briers) / len(briers) if briers else None,
        "mean_ece": sum(eces) / len(eces) if eces else None,
        "pooled_auc_roc": auc_roc(pooled_scores, pooled_labels) if pooled_scores else None,
        "pooled_auc_pr": auc_pr(pooled_scores, pooled_labels) if pooled_scores else None,
        "pooled_metrics": (
            binary_classification_metrics(
                pooled_scores, pooled_labels, threshold=prediction_threshold
            )
            if pooled_scores
            else None
        ),
        "n_auc_roc_targets": len(aucs_roc),
        "n_auc_pr_targets": len(aucs_pr),
        "pooled_residues": len(pooled_scores),
        "pooled_positive_rate": (
            sum(pooled_labels) / len(pooled_labels) if pooled_labels else None
        ),
        "structure_backend": structure_backend,
        "checkpoint": checkpoint,
        "cluster_manifest": cluster_manifest,
        "eligibility_audit": eligibility_audit,
        "calibration": calibration,
        "calibration_method": (
            calibration_metadata.get("method") if calibration_metadata else None
        ),
        "prediction_threshold": prediction_threshold,
        "training_cluster_isolation_checked": training_clusters is not None,
        "per_target": results,
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
        print(f"\nSaved benchmark summary → {out_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_target"}, indent=2))
    return summary


def self_test():
    """Embedded tiny case: validate AUC computation with a known-good example."""
    # Perfectly separated: high scores → ordered (label 0), low → disordered (1).
    # Disorder head outputs HIGH disorder for disordered regions, so disordered=1
    # should get high scores.
    scores = [0.9, 0.85, 0.1, 0.05, 0.8]  # predicted disorder
    labels = [1, 1, 0, 0, 1]  # ground truth disorder
    ev = evaluate(scores, labels)
    assert abs(ev["auc_roc"] - 1.0) < 1e-6, ev
    print("self-test OK: perfectly-separated case AUC-ROC=1.0, AUC-PR=1.0")
    print(json.dumps(ev, indent=2))
    masked = evaluate([0.1, 0.9, 0.2], [0.0, float("nan"), 1.0])
    assert masked["n_residues"] == 2 and masked["n_undefined"] == 1, masked
    print("self-test OK: undefined labels are excluded")
    # Random (worthless) predictor: AUC≈0.5
    import random

    random.seed(0)
    s2 = [random.random() for _ in range(200)]
    l2 = [1 if x > 0.5 else 0 for x in [random.random() for _ in range(200)]]
    ev2 = evaluate(s2, l2)
    assert 0.3 < ev2["auc_roc"] < 0.7, ev2
    print(f"self-test OK: random predictor AUC-ROC≈{ev2['auc_roc']:.3f} (~0.5)")


def main():
    parser = argparse.ArgumentParser(description="CAID disorder-prediction benchmark")
    parser.add_argument(
        "--caid-dir",
        default=os.path.join("data", "caid2"),
        help="Directory with sequences.fasta + labels/",
    )
    parser.add_argument("--out", default=os.path.join("results", "ablation", "caid.json"))
    parser.add_argument(
        "--structure-backend",
        default="skip",
        choices=["af2", "esmfold", "skip"],
        help="How to obtain a structure for pure-sequence targets",
    )
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", help="Authoritative corrected EMA checkpoint")
    parser.add_argument(
        "--cluster-manifest", help="JSON mapping every CAID target ID to a homology cluster"
    )
    parser.add_argument(
        "--training-clusters", help="Text file of homology clusters used in training"
    )
    parser.add_argument(
        "--eligibility-audit", help="Homology audit JSON containing independent_ids"
    )
    parser.add_argument(
        "--calibration", help="Frozen calibration artifact used for probabilities and threshold"
    )
    parser.add_argument("--work-dir", default=os.path.join("results", "caid_workdir"))
    parser.add_argument(
        "--self-test", action="store_true", help="Run an embedded AUC correctness check and exit"
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.checkpoint or not args.calibration:
        parser.error("--checkpoint and --calibration are required for an authoritative benchmark")

    if not os.path.isdir(args.caid_dir):
        print(f"ERROR: CAID dir not found: {args.caid_dir}")
        print("  Download CAID2/3 reference, or run --self-test to validate the metrics.")
        sys.exit(1)

    training_clusters = None
    if args.training_clusters:
        with open(args.training_clusters, encoding="utf-8") as handle:
            training_clusters = [line.strip() for line in handle if line.strip()]
    if bool(args.cluster_manifest) != bool(args.training_clusters):
        parser.error("--cluster-manifest and --training-clusters must be provided together")
    run_benchmark(
        args.caid_dir,
        args.out,
        args.structure_backend,
        args.max_targets,
        args.device,
        args.work_dir,
        args.checkpoint,
        args.cluster_manifest,
        training_clusters,
        args.calibration,
        args.eligibility_audit,
    )


if __name__ == "__main__":
    main()
