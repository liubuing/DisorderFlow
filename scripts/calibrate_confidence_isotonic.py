#!/usr/bin/env python
"""方案B: Isotonic / Affine calibration for BFN confidence heads.

Collects ALL available paired (BFN_pred, AF2_true) data from oc_validation_results/
and fits:
  1. Per-model affine calibration (scale + shift)
  2. Global isotonic regression across all models
  3. PPL-augmented linear model (BFN_pLDDT + BFN_ipTM + log(PPL) → AF2 score)

Outputs:
  - calibration_functions.pkl  (sklearn calibrators, ready for inference)
  - calibration_report.json    (metrics before/after)

Usage:
  python scripts/calibrate_confidence_isotonic.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT = Path(__file__).resolve().parent.parent
OC_DIR = PROJECT / "oc_validation_results"
OUT_DIR = PROJECT / "calibration_artifacts"
OUT_DIR.mkdir(exist_ok=True)


# ─── 1. Collect paired data ───────────────────────────────────────────────────

def collect_paired_data():
    """Extract all (bfn_plddt, bfn_iptm, bfn_ppl, af2_plddt, af2_iptm) tuples."""
    records = []

    # --- V12, V14: nested under models.{version}.af2_results ---
    for fname in ["oc_v12_phase2.json", "oc_v14_phase2.json",
                  "oc_v11seqconf_phase2.json"]:
        fpath = OC_DIR / fname
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for model_key, model_data in data.get("models", {}).items():
            for r in model_data.get("af2_results", []):
                records.append({
                    "model": model_key,
                    "source": fname,
                    "bfn_plddt": r["bfn_plddt"],
                    "bfn_iptm": r["bfn_iptm"],
                    "bfn_ppl": r.get("bfn_ppl", None),
                    "bfn_pae": r.get("bfn_pae", None),
                    "bfn_entropy": r.get("bfn_entropy", None),
                    "af2_plddt": r["af2_plddt"],
                    "af2_iptm": r["af2_iptm"],
                    "af2_ptm": r.get("af2_ptm", None),
                })

    # --- V15, V15b, V15dg03, V15C, V16C: same nested format ---
    for fname in ["oc_v15_phase2_complex.json", "oc_v15b_phase2_complex.json",
                  "oc_v15dg03_phase2_complex.json", "oc_v15C_phase2.json",
                  "oc_v16C_phase2.json", "oc_v14_phase2_complex.json"]:
        fpath = OC_DIR / fname
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for model_key, model_data in data.get("models", {}).items():
            for r in model_data.get("af2_results", []):
                records.append({
                    "model": model_key,
                    "source": fname,
                    "bfn_plddt": r["bfn_plddt"],
                    "bfn_iptm": r["bfn_iptm"],
                    "bfn_ppl": r.get("bfn_ppl", None),
                    "bfn_pae": r.get("bfn_pae", None),
                    "bfn_entropy": r.get("bfn_entropy", None),
                    "af2_plddt": r["af2_plddt"],
                    "af2_iptm": r["af2_iptm"],
                    "af2_ptm": r.get("af2_ptm", None),
                })

    # --- V17i: flat list under "designs" ---
    fpath = OC_DIR / "oc_v17i_phase2_complex.json"
    if fpath.exists():
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for r in data.get("designs", []):
            if r.get("af2_success"):
                records.append({
                    "model": "v17i",
                    "source": "oc_v17i_phase2_complex.json",
                    "bfn_plddt": r["bfn_plddt"],
                    "bfn_iptm": r["bfn_iptm"],
                    "bfn_ppl": r.get("bfn_ppl", None),
                    "bfn_pae": r.get("bfn_pae", None),
                    "bfn_entropy": r.get("bfn_entropy", None),
                    "af2_plddt": r["af2_plddt"],
                    "af2_iptm": r["af2_iptm"],
                    "af2_ptm": r.get("af2_ptm", None),
                })

    return records


# ─── 2. Fit calibrators ──────────────────────────────────────────────────────

def fit_affine(x, y):
    """Fit y = a*x + b via least squares. Returns (a, b)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.column_stack([x, np.ones_like(x)])
    result = np.linalg.lstsq(A, y, rcond=None)
    a, b = result[0]
    return float(a), float(b)


def fit_calibrators(records):
    """Fit multiple calibration strategies."""
    bfn_plddt = np.array([r["bfn_plddt"] for r in records])
    bfn_iptm = np.array([r["bfn_iptm"] for r in records])
    af2_plddt = np.array([r["af2_plddt"] for r in records])
    af2_iptm = np.array([r["af2_iptm"] for r in records])

    # PPL features (some may be None)
    has_ppl = [r["bfn_ppl"] is not None for r in records]
    bfn_ppl = np.array([r["bfn_ppl"] if r["bfn_ppl"] is not None else 50.0
                        for r in records])
    log_ppl = np.log1p(bfn_ppl)

    calibrators = {}

    # --- A. Global isotonic regression ---
    iso_plddt = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_plddt.fit(bfn_plddt, af2_plddt)
    calibrators["isotonic_plddt"] = iso_plddt

    iso_iptm = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_iptm.fit(bfn_iptm, af2_iptm)
    calibrators["isotonic_iptm"] = iso_iptm

    # --- B. Per-model affine calibration ---
    models_seen = set(r["model"] for r in records)
    affine_plddt = {}
    affine_iptm = {}
    for m in models_seen:
        mask = np.array([r["model"] == m for r in records])
        if mask.sum() < 2:
            continue
        a_p, b_p = fit_affine(bfn_plddt[mask], af2_plddt[mask])
        a_i, b_i = fit_affine(bfn_iptm[mask], af2_iptm[mask])
        affine_plddt[m] = (a_p, b_p)
        affine_iptm[m] = (a_i, b_i)
    calibrators["affine_plddt"] = affine_plddt
    calibrators["affine_iptm"] = affine_iptm

    # --- C. PPL-augmented Ridge regression (multi-feature → AF2) ---
    # Features: [bfn_plddt, bfn_iptm, log_ppl]
    X_multi = np.column_stack([bfn_plddt, bfn_iptm, log_ppl])
    ridge_plddt = Ridge(alpha=1.0)
    ridge_plddt.fit(X_multi, af2_plddt)
    calibrators["ridge_plddt"] = ridge_plddt

    ridge_iptm = Ridge(alpha=1.0)
    ridge_iptm.fit(X_multi, af2_iptm)
    calibrators["ridge_iptm"] = ridge_iptm

    # --- D. Global affine (single scalar mapping) ---
    a_gp, b_gp = fit_affine(bfn_plddt, af2_plddt)
    a_gi, b_gi = fit_affine(bfn_iptm, af2_iptm)
    calibrators["global_affine_plddt"] = (a_gp, b_gp)
    calibrators["global_affine_iptm"] = (a_gi, b_gi)

    return calibrators


# ─── 3. Evaluate ─────────────────────────────────────────────────────────────

def evaluate(records, calibrators):
    """Compute before/after metrics for each calibration method."""
    bfn_plddt = np.array([r["bfn_plddt"] for r in records])
    bfn_iptm = np.array([r["bfn_iptm"] for r in records])
    af2_plddt = np.array([r["af2_plddt"] for r in records])
    af2_iptm = np.array([r["af2_iptm"] for r in records])
    bfn_ppl = np.array([r["bfn_ppl"] if r["bfn_ppl"] is not None else 50.0
                        for r in records])
    log_ppl = np.log1p(bfn_ppl)

    results = {}

    # Baseline (uncalibrated)
    sp_plddt_raw, _ = stats.spearmanr(bfn_plddt, af2_plddt)
    sp_iptm_raw, _ = stats.spearmanr(bfn_iptm, af2_iptm)
    mae_plddt_raw = mean_absolute_error(af2_plddt, bfn_plddt)
    mae_iptm_raw = mean_absolute_error(af2_iptm, bfn_iptm)
    oc_plddt_raw = float(np.mean(bfn_plddt / np.clip(af2_plddt, 1e-6, None)))
    oc_iptm_raw = float(np.mean(bfn_iptm / np.clip(af2_iptm, 1e-6, None)))

    results["baseline"] = {
        "spearman_plddt": float(sp_plddt_raw) if not np.isnan(sp_plddt_raw) else 0.0,
        "spearman_iptm": float(sp_iptm_raw) if not np.isnan(sp_iptm_raw) else 0.0,
        "mae_plddt": float(mae_plddt_raw),
        "mae_iptm": float(mae_iptm_raw),
        "oc_ratio_plddt": oc_plddt_raw,
        "oc_ratio_iptm": oc_iptm_raw,
    }

    # Isotonic
    iso_p = calibrators["isotonic_plddt"].predict(bfn_plddt)
    iso_i = calibrators["isotonic_iptm"].predict(bfn_iptm)
    sp_p, _ = stats.spearmanr(iso_p, af2_plddt)
    sp_i, _ = stats.spearmanr(iso_i, af2_iptm)
    results["isotonic"] = {
        "spearman_plddt": float(sp_p) if not np.isnan(sp_p) else 0.0,
        "spearman_iptm": float(sp_i) if not np.isnan(sp_i) else 0.0,
        "mae_plddt": float(mean_absolute_error(af2_plddt, iso_p)),
        "mae_iptm": float(mean_absolute_error(af2_iptm, iso_i)),
        "oc_ratio_plddt": float(np.mean(iso_p / np.clip(af2_plddt, 1e-6, None))),
        "oc_ratio_iptm": float(np.mean(iso_i / np.clip(af2_iptm, 1e-6, None))),
    }

    # Global affine
    a_gp, b_gp = calibrators["global_affine_plddt"]
    a_gi, b_gi = calibrators["global_affine_iptm"]
    aff_p = np.clip(a_gp * bfn_plddt + b_gp, 0, 1)
    aff_i = np.clip(a_gi * bfn_iptm + b_gi, 0, 1)
    sp_p2, _ = stats.spearmanr(aff_p, af2_plddt)
    sp_i2, _ = stats.spearmanr(aff_i, af2_iptm)
    results["global_affine"] = {
        "spearman_plddt": float(sp_p2) if not np.isnan(sp_p2) else 0.0,
        "spearman_iptm": float(sp_i2) if not np.isnan(sp_i2) else 0.0,
        "mae_plddt": float(mean_absolute_error(af2_plddt, aff_p)),
        "mae_iptm": float(mean_absolute_error(af2_iptm, aff_i)),
        "oc_ratio_plddt": float(np.mean(aff_p / np.clip(af2_plddt, 1e-6, None))),
        "oc_ratio_iptm": float(np.mean(aff_i / np.clip(af2_iptm, 1e-6, None))),
        "params_plddt": {"slope": a_gp, "intercept": b_gp},
        "params_iptm": {"slope": a_gi, "intercept": b_gi},
    }

    # Ridge (PPL-augmented)
    X_multi = np.column_stack([bfn_plddt, bfn_iptm, log_ppl])
    ridge_p = np.clip(calibrators["ridge_plddt"].predict(X_multi), 0, 1)
    ridge_i = np.clip(calibrators["ridge_iptm"].predict(X_multi), 0, 1)
    sp_p3, _ = stats.spearmanr(ridge_p, af2_plddt)
    sp_i3, _ = stats.spearmanr(ridge_i, af2_iptm)
    results["ridge_ppl_augmented"] = {
        "spearman_plddt": float(sp_p3) if not np.isnan(sp_p3) else 0.0,
        "spearman_iptm": float(sp_i3) if not np.isnan(sp_i3) else 0.0,
        "mae_plddt": float(mean_absolute_error(af2_plddt, ridge_p)),
        "mae_iptm": float(mean_absolute_error(af2_iptm, ridge_i)),
        "oc_ratio_plddt": float(np.mean(ridge_p / np.clip(af2_plddt, 1e-6, None))),
        "oc_ratio_iptm": float(np.mean(ridge_i / np.clip(af2_iptm, 1e-6, None))),
        "coefficients_plddt": calibrators["ridge_plddt"].coef_.tolist(),
        "coefficients_iptm": calibrators["ridge_iptm"].coef_.tolist(),
    }

    # LOO cross-validation for Ridge (honest estimate)
    loo = LeaveOneOut()
    loo_pred_p = np.zeros_like(af2_plddt)
    loo_pred_i = np.zeros_like(af2_iptm)
    for train_idx, test_idx in loo.split(X_multi):
        rp = Ridge(alpha=1.0).fit(X_multi[train_idx], af2_plddt[train_idx])
        ri = Ridge(alpha=1.0).fit(X_multi[train_idx], af2_iptm[train_idx])
        loo_pred_p[test_idx] = rp.predict(X_multi[test_idx])
        loo_pred_i[test_idx] = ri.predict(X_multi[test_idx])
    loo_pred_p = np.clip(loo_pred_p, 0, 1)
    loo_pred_i = np.clip(loo_pred_i, 0, 1)
    sp_loo_p, _ = stats.spearmanr(loo_pred_p, af2_plddt)
    sp_loo_i, _ = stats.spearmanr(loo_pred_i, af2_iptm)
    results["ridge_loo_cv"] = {
        "spearman_plddt": float(sp_loo_p) if not np.isnan(sp_loo_p) else 0.0,
        "spearman_iptm": float(sp_loo_i) if not np.isnan(sp_loo_i) else 0.0,
        "mae_plddt": float(mean_absolute_error(af2_plddt, loo_pred_p)),
        "mae_iptm": float(mean_absolute_error(af2_iptm, loo_pred_i)),
    }

    return results


# ─── 4. Inference-time calibration wrapper ───────────────────────────────────

def make_calibration_fn(calibrators, method="isotonic"):
    """Return a callable: (bfn_plddt, bfn_iptm, ppl) -> (cal_plddt, cal_iptm)."""
    if method == "isotonic":
        iso_p = calibrators["isotonic_plddt"]
        iso_i = calibrators["isotonic_iptm"]
        def calibrate(plddt, iptm, ppl=None):
            p = float(iso_p.predict(np.array([plddt]))[0])
            i = float(iso_i.predict(np.array([iptm]))[0])
            return p, i
        return calibrate
    elif method == "global_affine":
        a_p, b_p = calibrators["global_affine_plddt"]
        a_i, b_i = calibrators["global_affine_iptm"]
        def calibrate(plddt, iptm, ppl=None):
            p = float(np.clip(a_p * plddt + b_p, 0, 1))
            i = float(np.clip(a_i * iptm + b_i, 0, 1))
            return p, i
        return calibrate
    elif method == "ridge":
        rp = calibrators["ridge_plddt"]
        ri = calibrators["ridge_iptm"]
        def calibrate(plddt, iptm, ppl=50.0):
            x = np.array([[plddt, iptm, np.log1p(ppl)]])
            p = float(np.clip(rp.predict(x)[0], 0, 1))
            i = float(np.clip(ri.predict(x)[0], 0, 1))
            return p, i
        return calibrate
    else:
        raise ValueError(f"Unknown method: {method}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("BFN Confidence Head Calibration (方案B)")
    print("=" * 70)

    # Collect
    records = collect_paired_data()
    print(f"\n[1] Collected {len(records)} paired (BFN, AF2) data points")
    models_seen = {}
    for r in records:
        models_seen.setdefault(r["model"], []).append(r)
    for m, rs in sorted(models_seen.items()):
        print(f"    {m}: {len(rs)} points | "
              f"BFN pLDDT={np.mean([r['bfn_plddt'] for r in rs]):.4f}±{np.std([r['bfn_plddt'] for r in rs]):.6f} | "
              f"AF2 pLDDT={np.mean([r['af2_plddt'] for r in rs]):.4f} | "
              f"AF2 ipTM={np.mean([r['af2_iptm'] for r in rs]):.4f}")

    # Fit
    print("\n[2] Fitting calibrators...")
    calibrators = fit_calibrators(records)
    print("    - Isotonic regression (global)")
    print("    - Per-model affine")
    print("    - PPL-augmented Ridge")
    print("    - Global affine")

    # Evaluate
    print("\n[3] Evaluation:")
    metrics = evaluate(records, calibrators)
    for method, m in metrics.items():
        print(f"\n  [{method}]")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            elif isinstance(v, dict):
                print(f"    {k}: {v}")

    # Save
    pkl_path = OUT_DIR / "calibration_functions.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(calibrators, f)
    print(f"\n[4] Saved calibrators → {pkl_path}")

    report_path = OUT_DIR / "calibration_report.json"
    report = {
        "n_records": len(records),
        "models": {m: len(rs) for m, rs in models_seen.items()},
        "metrics": metrics,
        "diagnosis": {
            "core_problem": "BFN confidence outputs are near-constant within each model "
                           "(std ~1e-5). Isotonic/affine calibration fixes MAGNITUDE but "
                           "cannot recover DISCRIMINATION. PPL remains the primary ranking signal.",
            "recommendation": "Use ridge_ppl_augmented for combined scoring; "
                             "for head discrimination fix, see finetune_confidence_head_v2.py (方案C).",
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[5] Saved report → {report_path}")

    # Demo: calibrate a sample
    print("\n[6] Demo calibration (V14 design #1):")
    cal_fn = make_calibration_fn(calibrators, "ridge")
    cal_p, cal_i = cal_fn(0.2668, 0.3379, ppl=61.0)
    print(f"    Raw BFN:  pLDDT=0.2668, ipTM=0.3379")
    print(f"    AF2 true: pLDDT=0.2661, ipTM=0.0528")
    print(f"    Calibrated: pLDDT={cal_p:.4f}, ipTM={cal_i:.4f}")

    return metrics


if __name__ == "__main__":
    main()
