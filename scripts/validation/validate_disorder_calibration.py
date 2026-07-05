#!/usr/bin/env python3
"""Validate Disorder-Aware AF2 Calibration.

Demonstrates and tests the calibration pipeline:
  1. Calibration curve: disorder -> AF2 reliability
  2. Simulated design scenarios (ordered vs disordered epitopes)
  3. Composite scoring with vs without calibration
  4. Batch calibration via closed_loop_scorer

Usage:
  python validate_disorder_calibration.py
"""

import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'modules')

# Force UTF-8 for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import math
import numpy as np


# ── Test 1: Calibration Curve ──

def test_calibration_curve():
    """Display the disorder → reliability mapping curve."""
    print("=" * 70)
    print("  Test 1: Disorder → AF2 Reliability Calibration Curve")
    print("=" * 70)
    print(f"  {'Disorder':<10} {'RMSF(A)':<10} {'Reliability':<12} {'Label':<8} {'Interpretation'}")
    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*8} {'─'*30}")

    from idp_antibody_design import disorder_to_af2_reliability

    disorder_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    for d in disorder_values:
        rel = disorder_to_af2_reliability(d)
        # Reverse tanh: RMSF ≈ scale * atanh(disorder) → disorder = tanh(RMSF/2.0)
        # So RMSF ≈ 2.0 * atanh(d)
        if d < 0.99:
            rmsf_est = 2.0 * math.atanh(max(d, 0.001))
        else:
            rmsf_est = float('inf')

        if rel >= 0.80:
            label = 'HIGH'
        elif rel >= 0.40:
            label = 'MEDIUM'
        else:
            label = 'LOW'

        if label == 'HIGH':
            interp = 'AF2 trustworthy, use raw scores'
        elif label == 'MEDIUM':
            interp = 'Blend AF2 ↔ BFN'
        else:
            interp = 'AF2 unreliable, rely on BFN PPL/entropy'

        print(f"  {d:<10.2f} {rmsf_est:<10.2f} {rel:<12.4f} {label:<8} {interp}")

    print()
    print("  Key thresholds:")
    print(f"    disorder=0.20 → reliability={disorder_to_af2_reliability(0.20):.4f} (AF2 fully trusted)")
    print(f"    disorder=0.30 → reliability={disorder_to_af2_reliability(0.30):.4f} (standard IDP threshold)")
    print(f"    disorder=0.35 → reliability={disorder_to_af2_reliability(0.35):.4f} (sigmoid midpoint)")
    print(f"    disorder=0.50 → reliability={disorder_to_af2_reliability(0.50):.4f} (AF2 mostly ignored)")
    print()


# ── Test 2: Simulated Design Scenarios ──

def test_design_scenarios():
    """Test composite scoring across different epitope disorder levels."""
    print("=" * 70)
    print("  Test 2: Composite Score — Calibrated vs Uncalibrated")
    print("=" * 70)

    from idp_antibody_design import composite_score, calibrate_af2_with_disorder

    # Simulate a "good" design: strong AF2 but varying epitope disorder
    base_design = {
        'sequence': 'SYAMSWVRQAPGKGLEWVSAISGSGGSTY',
        'plddt': 0.80,       # BFN self-confidence
        'iptm': 0.75,
        'pae': 15.0,
        'ppl': 8.5,
        'entropy': 0.55,
        'af2_success': True,
        'af2_plddt': 0.72,
        'af2_iptm': 0.68,
        'af2_interface_pae': 12.0,
        'af2_ptm': 0.65,
    }

    print(f"  Scenario: Same AF2 output (ipTM=0.68, pLDDT=0.72), different epitope disorder")
    print(f"  {'─'*70}")
    print(f"  {'Disorder':<10} {'Reliability':<12} {'Calibrated':<12} {'Uncalibrated':<14} {'Delta':<8} {'Weights(PPL)':<14} {'Weights(iptm)':<14}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*14} {'─'*8} {'─'*14} {'─'*14}")

    disorder_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

    for epi_d in disorder_values:
        d_cal = dict(base_design)
        d_cal['segment_mean_disorder'] = epi_d

        d_no_cal = dict(base_design)
        d_no_cal['segment_mean_disorder'] = None  # explicitly no calibration

        score_cal = composite_score(d_cal)
        score_no_cal = composite_score(d_no_cal)

        # Get reliability from stored metadata
        cal_meta = d_cal.get('_af2_calibrated', {})
        rel = cal_meta.get('af2_reliability', 1.0)

        # Show weights by inferring from score breakdown
        # (PPL weight increases as reliability decreases)
        w_ppl_est = 0.30 + 0.15 * (1 - rel)
        w_iptm_est = 0.10 + 0.15 * rel

        delta = score_cal - score_no_cal
        direction = '↑' if delta > 0 else '↓' if delta < 0 else '='

        print(f"  {epi_d:<10.2f} {rel:<12.4f} {score_cal:<12.4f} {score_no_cal:<14.4f} "
              f"{delta:+.4f}{direction:<3} {w_ppl_est:<14.3f} {w_iptm_est:<14.3f}")

    print()
    print("  Interpretation:")
    print("    Low disorder (0.1-0.2): calibrated ≈ uncalibrated (AF2 trusted)")
    print("    Mid disorder (0.3-0.4): calibration reduces AF2 weight, raises PPL")
    print("    High disorder (0.5+):  calibration heavily downweights AF2")
    print("    → For IDP designs, calibration prevents AF2 from dominating ranking")
    print("      when the epitope itself is structurally uncertain.")
    print()


# ── Test 3: Batch Calibration ──

def test_batch_calibration():
    """Test batch calibration via closed_loop_scorer."""
    print("=" * 70)
    print("  Test 3: Batch Calibration (closed_loop_scorer)")
    print("=" * 70)

    from closed_loop_scorer import calibrate_results_with_disorder, MultiObjectiveRanker

    # Simulate 6 designs: 3 good, 3 bad, varying epitope disorder
    results = [
        # Good designs on ORDERED epitope (AF2 reliable)
        {'sequence': 'DESIGN_A1', 'iptm': 0.80, 'plddt': 78.0, 'ppl': 6.0,
         'af2_iptm': 0.72, 'af2_plddt': 0.75, 'af2_interface_pae': 10.0,
         'segment_mean_disorder': 0.15, 'dG': -12.0},
        {'sequence': 'DESIGN_A2', 'iptm': 0.75, 'plddt': 74.0, 'ppl': 7.0,
         'af2_iptm': 0.68, 'af2_plddt': 0.70, 'af2_interface_pae': 12.0,
         'segment_mean_disorder': 0.18, 'dG': -10.0},
        {'sequence': 'DESIGN_A3', 'iptm': 0.72, 'plddt': 70.0, 'ppl': 9.0,
         'af2_iptm': 0.65, 'af2_plddt': 0.68, 'af2_interface_pae': 15.0,
         'segment_mean_disorder': 0.25, 'dG': -8.0},

        # Good BFN but WEAK AF2 on DISORDERED epitope (AF2 unreliable → downweight AF2)
        {'sequence': 'DESIGN_B1', 'iptm': 0.60, 'plddt': 65.0, 'ppl': 5.5,  # BFN moderate
         'af2_iptm': 0.25, 'af2_plddt': 0.30, 'af2_interface_pae': 25.0,      # AF2 weak
         'segment_mean_disorder': 0.55, 'dG': -5.0},                            # High disorder!
        {'sequence': 'DESIGN_B2', 'iptm': 0.55, 'plddt': 60.0, 'ppl': 7.0,
         'af2_iptm': 0.20, 'af2_plddt': 0.25, 'af2_interface_pae': 28.0,
         'segment_mean_disorder': 0.60, 'dG': -3.0},
        {'sequence': 'DESIGN_B3', 'iptm': 0.50, 'plddt': 55.0, 'ppl': 10.0,
         'af2_iptm': 0.18, 'af2_plddt': 0.22, 'af2_interface_pae': 30.0,
         'segment_mean_disorder': 0.65, 'dG': -2.0},
    ]

    # Without calibration
    ranker = MultiObjectiveRanker()
    uncalibrated = [dict(r) for r in results]
    ranker.compute_weighted_composite(uncalibrated)

    # With calibration
    calibrated = calibrate_results_with_disorder([dict(r) for r in results])
    ranker.compute_weighted_composite(calibrated)

    print(f"  {'Design':<12} {'Disorder':<10} {'AF2 Reliability':<16} "
          f"{'Uncal Rank':<12} {'Cal Rank':<12} {'Shift':<8}")
    print(f"  {'─'*12} {'─'*10} {'─'*16} {'─'*12} {'─'*12} {'─'*8}")

    for i, (uc, cc) in enumerate(zip(uncalibrated, calibrated)):
        disorder = cc.get('segment_mean_disorder', '?')
        cal_meta = cc.get('_af2_calibrated', {})
        rel = cal_meta.get('reliability', 1.0) if cal_meta else 1.0
        rank_shift = uc['mo_rank'] - cc['mo_rank']

        if rank_shift > 0:
            shift_str = f'↑{rank_shift}'
        elif rank_shift < 0:
            shift_str = f'↓{abs(rank_shift)}'
        else:
            shift_str = '='

        print(f"  {cc['sequence']:<12} {disorder:<10.2f} {rel:<16.4f} "
              f"{uc['mo_rank']:<12d} {cc['mo_rank']:<12d} {shift_str:<8}")

    print()
    print("  Expected behavior:")
    print("    A-series (low disorder): AF2 reliable → calibrated ≈ uncalibrated rank")
    print("    B-series (high disorder): AF2 unreliable → calibrated rank IMPROVES")
    print("      because poor AF2 scores are downweighted; BFN PPL/entropy dominate")
    print()

    # Show calibration metadata for one design
    print("  Sample calibration metadata (DESIGN_B1):")
    b1_cal = calibrated[3]['_af2_calibrated']
    if b1_cal:
        for k, v in b1_cal.items():
            print(f"    {k}: {v}")
    print()


# ── Test 4: Real Data Quick Check ──

def test_with_real_data():
    """If available, run calibration on real overconfidence test data."""
    import glob
    oc_files = glob.glob('oc_validation_results/*.json')
    if not oc_files:
        print("  No overconfidence validation data found — skipping real-data test.")
        print("  Run validate_overconfidence.py --phase1 first to generate data.")
        return

    print("=" * 70)
    print("  Test 4: Calibration on Real Overconfidence Data")
    print("=" * 70)

    import json
    from closed_loop_scorer import calibrate_results_with_disorder

    for f in sorted(oc_files)[:2]:
        with open(f) as fp:
            data = json.load(fp)

        designs = data if isinstance(data, list) else data.get('designs', [])
        if not designs:
            print(f"  {f}: no designs found")
            continue

        # Add synthetic disorder values based on known IDP targets
        # (Abeta42 is a known IDP → disorder ~0.3-0.6)
        for d in designs:
            if 'segment_mean_disorder' not in d:
                d['segment_mean_disorder'] = 0.4  # conservative IDP estimate

        calibrated = calibrate_results_with_disorder(designs)
        reliabilities = [
            c['_af2_calibrated']['reliability']
            for c in calibrated if c.get('_af2_calibrated')
        ]

        if reliabilities:
            print(f"  {f}: {len(designs)} designs, "
                  f"mean reliability={np.mean(reliabilities):.3f} "
                  f"[{min(reliabilities):.3f}-{max(reliabilities):.3f}]")
            n_high = sum(1 for r in reliabilities if r >= 0.8)
            n_med = sum(1 for r in reliabilities if 0.4 <= r < 0.8)
            n_low = sum(1 for r in reliabilities if r < 0.4)
            print(f"       HIGH={n_high} MEDIUM={n_med} LOW={n_low}")
    print()


# ── Main ──

if __name__ == '__main__':
    test_calibration_curve()
    test_design_scenarios()
    test_batch_calibration()
    test_with_real_data()

    print("=" * 70)
    print("  All tests complete.")
    print("=" * 70)
