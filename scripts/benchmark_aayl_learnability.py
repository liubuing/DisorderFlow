"""Frozen, descriptive single-to-multiple mutation learning experiment.

Run --prepare before --run. Previously exposed labels: not a blind test.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze

OUT = ROOT / "results/aayl_learnability_v1"
COHORT = ROOT / "data/aayl_original_retrospective_v1/cohort.json"
SCORES = ROOT / "results/aayl_original_retrospective_v1/scores.json"
AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = [20260916, 20260917, 20260918, 20260919, 20260920]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mutations(sequence, parent):
    if len(sequence) != len(parent):
        raise ValueError("Substitutions only")
    return frozenset((i, a) for i, (a, b) in enumerate(zip(sequence, parent)) if a != b)


def top_weights(values, k):
    """Fractional membership at tied top-k boundary, independent of row order."""
    values = np.asarray(values)
    threshold = np.sort(values)[-k]
    out = (values > threshold).astype(float)
    tie = values == threshold
    out[tie] = (k - out.sum()) / tie.sum()
    return out


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    if n < 3 or len(np.unique(y)) < 2:
        return {"n": n, "spearman": None, "top20_recall": None, "enrichment": None, "pair_accuracy": None}
    rho = float(spearmanr(y, p).statistic) if len(np.unique(p)) > 1 else None
    k = math.ceil(.2 * n)
    recall = float(top_weights(y, k) @ top_weights(p, k) / k)
    i, j = np.triu_indices(n, 1)
    dy, dp = np.sign(y[i] - y[j]), np.sign(p[i] - p[j])
    valid = dy != 0
    pair = float(np.mean((dy[valid] == dp[valid]) + .5 * (dp[valid] == 0)))
    return {"n": n, "spearman": rho, "top20_recall": recall,
            "enrichment": recall / (k / n), "pair_accuracy": pair}


def load():
    d = json.loads(COHORT.read_text())
    rows = d["eligible"]
    assert len({(r["heavy_sequence"], r["light_sequence"]) for r in rows}) == len(rows)
    parents = {f: s["sequences"][s["heavy_chain"]] for f, s in d["structures"].items()}
    changes = [mutations(r["heavy_sequence"], parents[r["family"]]) for r in rows]
    return d, rows, changes


def prepare():
    import torch
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm2_t6_8M_UR50D.pt"
    d, rows, changes = load()
    protocol = {
        "status": "exposed_data_development_not_confirmatory", "cohort_sha256": digest(COHORT),
        "zero_shot_scores_sha256": digest(SCORES), "script_sha256": digest(Path(__file__)),
        "endpoint": "9 - median of three original log10 estimated Kd(nM), higher better",
        "split": "within each lineage: train single substitutions; test doubles and triples SEPARATELY",
        "budgets": [20, 50, "all"], "seeds": SEEDS,
        "full_budget": "one run using first seed; no fake replication of identical full training sets",
        "budget_100": "not feasible: only 89 and 55 single mutants; omitted before training",
        "primary": "full budget, all candidates, each of four lineage x mutation-degree cells; Spearman and fractional top20 recall",
        "secondary": "coverage strata and 20/50-label learning curves; exploratory, not selection of best test split",
        "coverage": "covered iff EVERY (position, mutant amino acid) appears in selected training singles",
        "zero_shot": ["ECLS", "negative_complex_NLL", "negative_apo_NLL"],
        "ECLS_direction": "apo_NLL - complex_NLL, higher better; inverse of frozen-paper cost convention",
        "models": {"position_ridge": {"alpha": [.01, .1, 1., 10., 100.]},
                   "position_rbf": {"alpha": [.01, .1, 1., 10.], "gamma": [.25, 1., 4.]},
                   "esm2_ridge": {"alpha": [.01, .1, 1., 10., 100.]}},
        "selection": "5-fold shuffled training-only CV, negative MSE; same folds for all models; y centered using training fold only",
        "onehot": "fixed standard-20-AA H3 position encoding minus parent; no fitted vocabulary or test scaling",
        "embedding": "frozen ESM2 t6 8M layer6 H3 mean from full heavy sequence; 320 dimensions; train-fold-only StandardScaler then Ridge",
        "embedding_weights": str(weights), "embedding_weights_sha256": digest(weights),
        "no_WT_anchor": "single-mutant-only training does not identify absolute parent offset; evaluate within degree only, never claim calibrated affinity or pooled-degree ranking",
        "uncertainty": "seed spread is subsampling sensitivity, not an independent-system confidence interval",
        "limitations": "one antigen, two lineages, complete-readout selection; pretrained-model exposure not excluded",
        "decision": "advance complex modeling only if ESM has strictly higher full-budget Spearman than position Ridge in all four cells and top20 recall no worse in at least three; a resource gate, not significance or publication evidence",
    }
    freeze(OUT / "protocol.json", protocol)
    splits = []
    for family in sorted(d["structures"]):
        singles = [i for i, r in enumerate(rows) if r["family"] == family and len(changes[i]) == 1]
        for budget in protocol["budgets"]:
            for seed in SEEDS[:1] if budget == "all" else SEEDS:
                train = singles if budget == "all" else sorted(np.random.default_rng(seed).choice(singles, budget, replace=False).tolist())
                seen = frozenset().union(*(changes[i] for i in train))
                test = [i for i, r in enumerate(rows) if r["family"] == family and len(changes[i]) in (2, 3)]
                assert not set(train) & set(test)
                splits.append({"family": family, "budget": budget, "seed": seed, "train": train, "test": test,
                               "covered": [i for i in test if changes[i] <= seen]})
    freeze(OUT / "splits.json", {"cohort_sha256": digest(COHORT), "rows": [r["poi"] for r in rows], "splits": splits})
    print("Protocol and 22 deterministic training splits frozen before fitting", flush=True)


def embeddings(d, rows, protocol):
    import esm
    import torch
    path = OUT / "esm2_h3_embeddings.npz"
    receipt = OUT / "embedding_receipt.json"
    if path.exists() and receipt.exists():
        meta = json.loads(receipt.read_text())
        assert meta["artifact_sha256"] == digest(path) and meta["cohort_sha256"] == digest(COHORT)
        return np.load(path)["features"]
    weights = Path(protocol["embedding_weights"])
    assert digest(weights) == protocol["embedding_weights_sha256"]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model.eval().cpu()
    converter = alphabet.get_batch_converter()
    out = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            _, _, tokens = converter([(r["poi"], r["heavy_sequence"]) for r in batch])
            rep = model(tokens, repr_layers=[6], return_contacts=False)["representations"][6]
            for j, r in enumerate(batch):
                indices = np.array(d["structures"][r["family"]]["h3_indices_zero_based_observed_heavy"]) + 1
                out.append(rep[j, indices].mean(0).numpy())
            if start % 128 == 0:
                print(f"Frozen ESM features: {min(start + 16, len(rows))}/{len(rows)}", flush=True)
    features = np.array(out)
    np.savez_compressed(path, features=features)
    freeze(receipt, {"artifact_sha256": digest(path), "cohort_sha256": digest(COHORT),
                    "weights_sha256": digest(weights), "torch_version": str(torch.__version__),
                    "device": "cpu", "shape": list(features.shape), "labels_used": False})
    return features


def run():
    from sklearn.compose import TransformedTargetRegressor
    import sklearn
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["script_sha256"] == digest(Path(__file__))
    assert protocol["cohort_sha256"] == digest(COHORT)
    assert protocol["zero_shot_scores_sha256"] == digest(SCORES)
    d, rows, changes = load()
    e = embeddings(d, rows, protocol)
    y = np.array([r["endpoint"] for r in rows])
    scores = {(r["family"], r["poi"]): r for r in json.loads(SCORES.read_text())["records"]}
    zero = {m: np.array([scores[(r["family"], r["poi"])][m] for r in rows]) for m in protocol["zero_shot"]}
    for r in rows:
        assert scores[(r["family"], r["poi"])]["endpoint"] == r["endpoint"]
    results, predictions, fits = [], [], []
    for split in json.loads((OUT / "splits.json").read_text())["splits"]:
        train, test = np.array(split["train"]), np.array(split["test"])
        family = split["family"]
        parent = d["structures"][family]["cdr_h3_sequence"]
        x = np.zeros((len(rows), len(parent) * 20))
        for i, r in enumerate(rows):
            if r["family"] == family:
                for pos, (a, b) in enumerate(zip(r["h3_sequence"], parent)):
                    x[i, pos * 20 + AA.index(a)] += 1
                    x[i, pos * 20 + AA.index(b)] -= 1
        pred = {m: v[test] for m, v in zero.items()}
        for name in protocol["models"]:
            feat = e if name == "esm2_ridge" else x
            model = make_pipeline(StandardScaler(), Ridge()) if name == "esm2_ridge" else (KernelRidge(kernel="rbf") if name == "position_rbf" else Ridge())
            # Target centering also happens inside each CV fold, especially for KRR.
            model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler(with_std=False))
            prefix = "regressor__ridge__" if name == "esm2_ridge" else "regressor__"
            grid = {prefix + k: v for k, v in protocol["models"][name].items()}
            cv = KFold(5, shuffle=True, random_state=split["seed"])
            search = GridSearchCV(model, grid, scoring="neg_mean_squared_error", cv=cv, n_jobs=1, error_score="raise")
            search.fit(feat[train], y[train])
            pred[name] = search.predict(feat[test])
            fits.append({"family": family, "budget": split["budget"], "seed": split["seed"], "model": name,
                         "n_train": len(train), "best_params": search.best_params_, "inner_cv_MSE": -search.best_score_})
        for name, p in pred.items():
            for i, value in zip(test, p):
                predictions.append({"family": family, "budget": split["budget"], "seed": split["seed"], "model": name,
                                    "poi": rows[i]["poi"], "prediction": float(value), "endpoint": float(y[i])})
            for degree in [2, 3]:
                for stratum in ["all", "covered", "uncovered"]:
                    mask = np.array([len(changes[i]) == degree and (stratum == "all" or ((i in split["covered"]) == (stratum == "covered"))) for i in test])
                    results.append({"family": family, "budget": split["budget"], "seed": split["seed"], "model": name,
                                    "degree": degree, "stratum": stratum, **metrics(y[test][mask], p[mask])})
        print(f"Completed {family} budget={split['budget']} seed={split['seed']}", flush=True)
    primary = [r for r in results if r["budget"] == "all" and r["stratum"] == "all"]
    cells = []
    for family in sorted(d["structures"]):
        for degree in [2, 3]:
            lookup = {r["model"]: r for r in primary if r["family"] == family and r["degree"] == degree}
            a, b = lookup["esm2_ridge"], lookup["position_ridge"]
            cells.append({"family": family, "degree": degree,
                          "esm_rho_gain": a["spearman"] - b["spearman"],
                          "esm_top20_gain": a["top20_recall"] - b["top20_recall"]})
    gate = all(c["esm_rho_gain"] > 0 for c in cells) and sum(c["esm_top20_gain"] >= 0 for c in cells) >= 3
    freeze(OUT / "predictions.json", {"records": predictions})
    freeze(OUT / "fits.json", {"records": fits})
    freeze(OUT / "report.json", {"protocol_sha256": digest(OUT / "protocol.json"), "splits_sha256": digest(OUT / "splits.json"),
                               "sklearn_version": sklearn.__version__, "metrics": results, "primary": primary,
                               "complex_model_gate_passed": gate, "gate_cells": cells})
    print(json.dumps({"complex_model_gate_passed": gate, "cells": cells}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run()
