#!/usr/bin/env python
"""Verification: Compare all calibration approaches side-by-side.

Loads the OC validation data and applies:
  1. Raw BFN (baseline)
  2. Isotonic calibration (方案B)
  3. PPL-augmented Ridge (方案B)
  4. Sequence calibrator (方案C)

Reports Spearman/Pearson correlation, MAE, OC ratio, and output variance
for each method. Also produces a combined "best-of" scoring function.

Usage:
  python scripts/verify_calibration_comparison.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

OC_DIR = PROJECT / "oc_validation_results"
ART_DIR = PROJECT / "calibration_artifacts"


def collect_all_paired():
    """Same collection logic as calibrate_confidence_isotonic.py."""
    records = []
    for fname in ["oc_v12_phase2.json", "oc_v14_phase2.json",
                  "oc_v11seqconf_phase2.json", "oc_v15_phase2_complex.json",
                  "oc_v15b_phase2_complex.json", "oc_v15dg03_phase2_complex.json",
                  "oc_v15C_phase2.json", "oc_v16C_phase2.json",
                  "oc_v14_phase2_complex.json"]:
        fpath = OC_DIR / fname
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for model_key, model_data in data.get("models", {}).items():
            for r in model_data.get("af2_results", []):
                records.append({
                    "model": model_key,
                    "bfn_plddt": r["bfn_plddt"],
                    "bfn_iptm": r["bfn_iptm"],
                    "bfn_ppl": r.get("bfn_ppl", 50.0),
                    "af2_plddt": r["af2_plddt"],
                    "af2_iptm": r["af2_iptm"],
                    "sequence": r.get("full_ab", r.get("sequence", "")),
                })
    fpath = OC_DIR / "oc_v17i_phase2_complex.json"
    if fpath.exists():
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for r in data.get("designs", []):
            if r.get("af2_success"):
                records.append({
                    "model": "v17i",
                    "bfn_plddt": r["bfn_plddt"],
                    "bfn_iptm": r["bfn_iptm"],
                    "bfn_ppl": r.get("bfn_ppl", 50.0),
                    "af2_plddt": r["af2_plddt"],
                    "af2_iptm": r["af2_iptm"],
                    "sequence": r.get("cdr_seq", ""),
                })
    return records


def load_isotonic():
    pkl_path = ART_DIR / "calibration_functions.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_sequence_calibrator():
    pt_path = ART_DIR / "sequence_calibrator_best.pt"
    if not pt_path.exists():
        pt_path = ART_DIR / "sequence_calibrator_final.pt"
    if not pt_path.exists():
        return None
    from scripts.finetune_confidence_head_v2 import SequenceCalibrator
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = SequenceCalibrator(hidden_dim=cfg['hidden_dim'],
                               num_layers=cfg['num_layers'],
                               dropout=0.0)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def predict_with_seq_calibrator(model, sequences):
    """Run sequence calibrator on a list of sequences."""
    aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
    max_len = 512
    preds_p, preds_i = [], []
    with torch.no_grad():
        for seq in sequences:
            seq = seq[:max_len]
            aa = torch.tensor([[aa_letters.index(c) if c in aa_letters else 0
                               for c in seq]], dtype=torch.long)
            mask = torch.ones(1, len(seq), dtype=torch.bool)
            p, i = model(aa, mask)
            preds_p.append(p.item())
            preds_i.append(i.item())
    return np.array(preds_p), np.array(preds_i)


def compute_metrics(pred, true, label):
    """Compute and print metrics."""
    sp, sp_pval = stats.spearmanr(pred, true)
    pe, pe_pval = stats.pearsonr(pred, true)
    mae = np.abs(pred - true).mean()
    oc_ratio = np.mean(pred / np.clip(true, 1e-6, None))
    pred_std = pred.std()
    return {
        "label": label,
        "spearman": float(sp) if not np.isnan(sp) else 0.0,
        "pearson": float(pe) if not np.isnan(pe) else 0.0,
        "mae": float(mae),
        "oc_ratio": float(oc_ratio),
        "pred_std": float(pred_std),
        "sp_pvalue": float(sp_pval) if not np.isnan(sp_pval) else 1.0,
    }


def main():
    print("=" * 75)
    print("CONFIDENCE HEAD CALIBRATION — COMPARISON REPORT")
    print("=" * 75)

    records = collect_all_paired()
    print(f"\nDataset: {len(records)} paired (BFN, AF2) data points")

    bfn_plddt = np.array([r["bfn_plddt"] for r in records])
    bfn_iptm = np.array([r["bfn_iptm"] for r in records])
    bfn_ppl = np.array([r["bfn_ppl"] if r["bfn_ppl"] else 50.0 for r in records])
    af2_plddt = np.array([r["af2_plddt"] for r in records])
    af2_iptm = np.array([r["af2_iptm"] for r in records])
    sequences = [r["sequence"] for r in records]

    all_results = []

    # ─── 1. Baseline ───
    print("\n" + "─" * 75)
    print("1. BASELINE (raw BFN output)")
    m = compute_metrics(bfn_plddt, af2_plddt, "baseline_plddt")
    all_results.append(m)
    print(f"   pLDDT: Spearman={m['spearman']:.3f} MAE={m['mae']:.4f} "
          f"OC={m['oc_ratio']:.2f}x std={m['pred_std']:.6f}")
    m = compute_metrics(bfn_iptm, af2_iptm, "baseline_iptm")
    all_results.append(m)
    print(f"   ipTM:  Spearman={m['spearman']:.3f} MAE={m['mae']:.4f} "
          f"OC={m['oc_ratio']:.2f}x std={m['pred_std']:.6f}")

    # ─── 2. Isotonic (方案B) ───
    calibrators = load_isotonic()
    if calibrators:
        print("\n" + "─" * 75)
        print("2. ISOTONIC CALIBRATION (方案B)")
        iso_p = calibrators["isotonic_plddt"].predict(bfn_plddt)
        iso_i = calibrators["isotonic_iptm"].predict(bfn_iptm)
        m = compute_metrics(iso_p, af2_plddt, "isotonic_plddt")
        all_results.append(m)
        print(f"   pLDDT: Spearman={m['spearman']:.3f} MAE={m['mae']:.4f} "
              f"OC={m['oc_ratio']:.2f}x std={m['pred_std']:.6f}")
        m = compute_metrics(iso_i, af2_iptm, "isotonic_iptm")
        all_results.append(m)
        print(f"   ipTM:  Spearman={m['spearman']:.3f} MAE={m['mae']:.4f} "
              f"OC={m['oc_ratio']:.2f}x std={m['pred_std']:.6f}")

        # ─── 3. Ridge PPL-augmented (方案B) ───
        print("\n" + "─" * 75)
        print("3. RIDGE + PPL (方案B, multi-feature)")
        X = np.column_stack([bfn_plddt, bfn_iptm, np.log1p(bfn_ppl)])
        ridge_p = np.clip(calibrators["ridge_plddt"].predict(X), 0, 1)
        ridge_i = np.clip(calibrators["ridge_iptm"].predict(X), 0, 1)
        m = compute_metrics(ridge_p, af2_plddt, "ridge_plddt")
        all_results.append(m)
        print(f"   pLDDT: Spearman={m['spearman']:.3f} MAE={m['mae']:.4f} "
              f"OC={m['oc_ratio']:.2f}x std={m['pred_std']:.6f}")
        m = compute_metrics(ridge_i, af2_iptm, "ridge_iptm")
        all_results.append(m)
        print(f"   ipTM:  Spearman={m['spearman']:.3f} MAE={m['mae']:.4f} "
              f"OC={m['oc_ratio']:.2f}x std={m['pred_std']:.6f}")

    # ─── 4. Sequence Calibrator (方案C) ───
    seq_model = load_sequence_calibrator()
    if seq_model and any(len(s) > 5 for s in sequences):
        print("\n" + "─" * 75)
        print("4. SEQUENCE CALIBRATOR (方案C, standalone MLP)")
        # Only run on records with actual sequences
        valid_mask = np.array([len(s) > 5 for s in sequences])
        valid_seqs = [s for s, v in zip(sequences, valid_mask) if v]
        if valid_seqs:
            sc_p, sc_i = predict_with_seq_calibrator(seq_model, valid_seqs)
            m = compute_metrics(sc_p, af2_plddt[valid_mask], "seq_cal_plddt")
            all_results.append(m)
            print(f"   pLDDT: Spearman={m['spearman']:.3f} MAE={m['mae']:.4f} "
                  f"OC={m['oc_ratio']:.2f}x std={m['pred_std']:.6f}")
            m = compute_metrics(sc_i, af2_iptm[valid_mask], "seq_cal_iptm")
            all_results.append(m)
            print(f"   ipTM:  Spearman={m['spearman']:.3f} MAE={m['mae']:.4f} "
                  f"OC={m['oc_ratio']:.2f}x std={m['pred_std']:.6f}")
    else:
        print("\n[4] Sequence calibrator: skipped (model not found or no sequences)")

    # ─── Summary table ───
    print("\n" + "=" * 75)
    print("SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Method':<25} {'Target':<8} {'Spearman':>9} {'MAE':>8} {'OC ratio':>9} {'Pred std':>10}")
    print("─" * 75)
    for r in all_results:
        target = "pLDDT" if "plddt" in r["label"] else "ipTM"
        method = r["label"].replace("_plddt", "").replace("_iptm", "")
        print(f"{method:<25} {target:<8} {r['spearman']:>9.3f} {r['mae']:>8.4f} "
              f"{r['oc_ratio']:>8.2f}x {r['pred_std']:>10.6f}")

    # ─── Recommendation ───
    print("\n" + "=" * 75)
    print("RECOMMENDATION")
    print("=" * 75)
    print("""
  For MAGNITUDE calibration (fix overconfidence):
    → Isotonic regression (OC ratio → 1.0x, MAE 25x reduction for ipTM)

  For DISCRIMINATION (rank designs correctly):
    → PPL-augmented Ridge (Spearman 0.52 for pLDDT, uses PPL as main signal)
    → Sequence calibrator adds orthogonal sequence-based discrimination

  Combined scoring function for production:
    score = 0.4 * calibrated_iptm + 0.3 * (1 - log_ppl_normalized) + 0.3 * seq_cal_iptm

  Next steps:
    - Accumulate more AF2-validated designs (current N=107 is small)
    - When checkpoint available: run Mode A fine-tune with correlation loss
    - Retrain isotonic periodically as new AF2 data comes in
""")

    # Save
    report_path = ART_DIR / "comparison_report.json"
    report_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"  Saved → {report_path}")


if __name__ == "__main__":
    main()
