"""Development-only single+double training, triple testing and label audit."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_aayl_learnability as base
from scripts.build_aayl_original_endpoint_cohort import freeze

OUT = ROOT / "results/aayl_low_order_v2"


def select_train(pool, changes, budget, seed):
    if budget == "all":
        return sorted(pool)
    singles = [i for i in pool if len(changes[i]) == 1]
    doubles = [i for i in pool if len(changes[i]) == 2]
    ns = round(budget * len(singles) / len(pool))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(singles, ns, replace=False).tolist() + rng.choice(doubles, budget-ns, replace=False).tolist())


def prepare():
    d, rows, changes = base.load()
    protocol = {
        "classification": "previously_exposed_development_data_not_independent_confirmation",
        "inputs": {str(p.resolve().relative_to(ROOT)): base.digest(p) for p in [base.COHORT, base.SCORES,
            base.OUT / "esm2_h3_embeddings.npz", base.OUT / "embedding_receipt.json",
            base.OUT / "report.json", Path(base.__file__), Path(__file__)]},
        "split": "each lineage independently; single+double train, all triples test; exact sequence disjoint",
        "budgets": [20, 50, 100, "all"], "seeds": base.SEEDS,
        "sampling": "fixed proportions of singles/doubles; round(budget*single_count/pool_count); all once; 5 random subsets for other budgets",
        "models": json.loads((base.OUT / "protocol.json").read_text())["models"],
        "selection": "same training-only shuffled 5-fold CV per split, negative MSE; fold-local target centering and ESM scaling",
        "primary": "full-budget Spearman and fractional top20 recall in both lineages, all triples",
        "secondary": "20/50/100 curves; exact-substitution coverage; comparison to prior single-only results is descriptive and full budgets differ",
        "gate": "ESM strictly improves Spearman over position Ridge in BOTH full-budget lineage cells and top20 recall decreases in NEITHER; resource gate only",
        "audit": "by lineage and degree: replicate-pair Spearman, median within-candidate log10 range; nonoverlapping empirical replicate ranges and pairwise accuracy therein; ranges are NOT confidence intervals",
        "missing": "eligible/(eligible+excluded) and observed finite replicate counts by lineage and degree; no imputation or claim missing-at-random",
        "limitations": "one antigen; labels and previous test outputs exposed; diagnostic results cannot be used for tuning this run; small ESM2 does not represent all protein models"
    }
    freeze(OUT / "protocol.json", protocol)
    splits = []
    for family in sorted(d["structures"]):
        pool = [i for i, r in enumerate(rows) if r["family"] == family and len(changes[i]) in (1, 2)]
        test = [i for i, r in enumerate(rows) if r["family"] == family and len(changes[i]) == 3]
        for budget in protocol["budgets"]:
            for seed in base.SEEDS[:1] if budget == "all" else base.SEEDS:
                train = select_train(pool, changes, budget, seed)
                seen = frozenset().union(*(changes[i] for i in train))
                splits.append({"family": family, "budget": budget, "seed": seed, "train": train, "test": test,
                               "covered": [i for i in test if changes[i] <= seen]})
    freeze(OUT / "splits.json", {"rows": [r["poi"] for r in rows], "splits": splits})
    print("Protocol and 32 splits frozen before fitting")


def label_audit(d, rows, changes):
    out = []
    for family, structure in sorted(d["structures"].items()):
        parent = structure["sequences"][structure["heavy_chain"]]
        for degree in [1, 2, 3]:
            kept = [r for i, r in enumerate(rows) if r["family"] == family and len(changes[i]) == degree]
            excluded = [r for r in d["excluded"] if r["family"] == family and len(base.mutations(r["heavy_sequence"], parent)) == degree]
            values = np.array([[next(o["original_log10_nM"] for o in r["observations"] if o["replicate"] == str(j)) for j in [1, 2, 3]] for r in kept])
            correlations = {f"{i+1}_{j+1}": float(spearmanr(values[:, i], values[:, j]).statistic) for i,j in [(0,1),(0,2),(1,2)]}
            out.append({"family": family, "degree": degree, "included": len(kept), "excluded": len(excluded),
                        "completion_fraction": len(kept)/(len(kept)+len(excluded)),
                        "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
                        "finite_readings_in_excluded": dict(Counter(sum(o["original_log10_nM"] is not None for o in r["observations"]) for r in excluded)),
                        "replicate_spearman": correlations, "median_log10_range": float(np.median(np.ptp(values, axis=1))),
                        "p90_log10_range": float(np.quantile(np.ptp(values, axis=1), .9))})
    return out


def fit_predict(x, y_train, train, test, model_name, grid, seed):
    model = make_pipeline(StandardScaler(), Ridge()) if model_name == "esm2_ridge" else (KernelRidge(kernel="rbf") if model_name == "position_rbf" else Ridge())
    model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler(with_std=False))
    prefix = "regressor__ridge__" if model_name == "esm2_ridge" else "regressor__"
    search = GridSearchCV(model, {prefix+k:v for k,v in grid.items()}, scoring="neg_mean_squared_error",
                          cv=KFold(5, shuffle=True, random_state=seed), n_jobs=1, error_score="raise")
    # No held-out labels are accepted by this function.
    search.fit(x[train], y_train)
    return search.predict(x[test]), search.best_params_, -float(search.best_score_), search


def run():
    import sklearn
    threadpool_limits(2)
    protocol = json.loads((OUT / "protocol.json").read_text())
    for relative, sha in protocol["inputs"].items():
        assert base.digest(ROOT / relative) == sha
    d, rows, changes = base.load()
    esm = np.load(base.OUT / "esm2_h3_embeddings.npz")["features"]
    y = np.array([r["endpoint"] for r in rows])
    score_rows = {(r["family"],r["poi"]):r for r in json.loads(base.SCORES.read_text())["records"]}
    zero_names = ["ECLS", "negative_complex_NLL", "negative_apo_NLL"]
    records, predictions, fits = [], [], []
    for s in json.loads((OUT / "splits.json").read_text())["splits"]:
        tr, te = np.array(s["train"]), np.array(s["test"])
        assert not set(tr) & set(te)
        parent = d["structures"][s["family"]]["cdr_h3_sequence"]
        x = np.zeros((len(rows), len(parent)*20))
        for i,r in enumerate(rows):
            if r["family"] == s["family"]:
                for j,(a,b) in enumerate(zip(r["h3_sequence"],parent)):
                    x[i,j*20+base.AA.index(a)] += 1
                    x[i,j*20+base.AA.index(b)] -= 1
        pred = {m:np.array([score_rows[(rows[i]["family"],rows[i]["poi"])][m] for i in te]) for m in zero_names}
        identity = {k:s[k] for k in ["family", "budget", "seed"]}
        for name, grid in protocol["models"].items():
            p, params, mse, _ = fit_predict(esm if name == "esm2_ridge" else x, y[tr], tr, te, name, grid, s["seed"])
            pred[name] = p
            fits.append({**identity,"model":name,"params":params,"inner_MSE":mse,"n_train":len(tr)})
        # Descriptive reliability strata; never used in training/model selection.
        raw = np.array([[o["original_log10_nM"] for o in rows[i]["observations"]] for i in te])
        lower, upper = (9-raw).min(axis=1), (9-raw).max(axis=1)
        ii,jj = np.triu_indices(len(te),1)
        separate = (lower[ii]>upper[jj]) | (lower[jj]>upper[ii])
        for name,p in pred.items():
            predictions.extend({**identity,"model":name,"index":int(i),"prediction":float(v)} for i,v in zip(te,p))
            for stratum in ["all","covered","uncovered"]:
                mask = np.array([stratum == "all" or ((i in s["covered"]) == (stratum == "covered")) for i in te])
                records.append({**identity,"model":name,"stratum":stratum,**base.metrics(y[te][mask],p[mask])})
            dy,dp = np.sign(y[te][ii]-y[te][jj]),np.sign(p[ii]-p[jj])
            records.append({**identity,"model":name,"stratum":"nonoverlapping_replicate_ranges",
                "n_pairs":int(separate.sum()),"total_pairs":len(ii),
                "pair_accuracy":float(np.mean((dy[separate]==dp[separate])+.5*(dp[separate]==0))) if separate.any() else None})
        print(f"Completed {s['family']} budget={s['budget']} seed={s['seed']}",flush=True)
    primary = [r for r in records if r["budget"] == "all" and r["stratum"] == "all"]
    cells = []
    for family in sorted(d["structures"]):
        lookup = {r["model"]:r for r in primary if r["family"] == family}
        a,b = lookup["esm2_ridge"],lookup["position_ridge"]
        cells.append({"family":family,"rho_gain":a["spearman"]-b["spearman"],"recall_gain":a["top20_recall"]-b["top20_recall"]})
    freeze(OUT / "predictions.json", {"records":predictions})
    freeze(OUT / "fits.json", {"records":fits})
    report = {"protocol_sha256":base.digest(OUT / "protocol.json"),"splits_sha256":base.digest(OUT / "splits.json"),
              "sklearn_version":sklearn.__version__,"metrics":records,"primary":primary,
              "complex_model_gate_passed":all(c["rho_gain"]>0 and c["recall_gain"]>=0 for c in cells),
              "gate_cells":cells,"label_audit":label_audit(d,rows,changes)}
    freeze(OUT / "report.json",report)
    print(json.dumps({"primary":primary,"gate":report["complex_model_gate_passed"],"audit":report["label_audit"]},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.run: run()
