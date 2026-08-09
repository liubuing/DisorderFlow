#!/usr/bin/env python
"""Generate T2.1 statistics, tables, and decision report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.t21_statistics import aggregate as corrected_aggregate  # noqa: E402


def load_results(results_path):
    return json.loads(Path(results_path).read_text(encoding="utf-8"))


def cluster_table(results, aggregate):
    lines = []
    lines.append("T2.1 Temporal Final Results")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Records: {aggregate['n_records']}")
    lines.append(f"Torsion-tier hits: {aggregate['n_torsion_hits']}")
    lines.append(f"Valid structures: {aggregate['n_valid_structures']}")
    lines.append(
        f"Overall valid fraction: {aggregate['overall_valid_structure_fraction']:.4f}")
    lines.append(f"Post-tier QC fraction: {aggregate['post_tier_qc_fraction']:.4f}")
    lines.append("")
    lines.append("Primary metrics (supplied-contact arm):")
    lines.append(f"  Mean held-out contact recovery: {aggregate.get('mean_held_out_contact_recovery', 'N/A'):.4f}")
    ci = aggregate.get("held_out_contact_recovery_ci95")
    lines.append(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "  95% CI: N/A")
    lines.append(f"  Positive-structure fraction: {aggregate.get('fraction_structures_positive_held_out_recovery', 'N/A'):.4f}")
    lines.append(f"  Mean RMSD recovery (A): {aggregate.get('mean_rmsd_recovery_angstrom', 'N/A'):.4f}")
    rmsd_ci = aggregate.get("rmsd_recovery_ci95")
    lines.append(f"  RMSD 95% CI: [{rmsd_ci[0]:.4f}, {rmsd_ci[1]:.4f}]" if rmsd_ci else "  RMSD 95% CI: N/A")
    lines.append("")
    lines.append("Arm comparisons (held-out contact recovery):")
    for arm_name, arm_data in aggregate.get("arm_comparisons", {}).items():
        arm_ci = arm_data.get("ci95", [None, None])
        lines.append(f"  {arm_name}: mean={arm_data['mean']:.4f}, "
                     f"95% CI=[{arm_ci[0]:.4f}, {arm_ci[1]:.4f}]")
    lines.append("")
    lines.append("Paired supplied-minus-control contrasts (cluster bootstrap):")
    for name, contrast in aggregate.get("paired_arm_contrasts", {}).items():
        ci95 = contrast["ci95"]
        lines.append(
            f"  {name}: mean={contrast['mean']:.4f}, "
            f"95% CI=[{ci95[0]:.4f}, {ci95[1]:.4f}], "
            f"n={contrast['n_inference_units']} {contrast['inference_unit']}s")
    lines.append("")
    gate_status = "PASSED" if aggregate.get("gate_pass") else "FAILED"
    lines.append(f"Gate status: {gate_status}")
    lines.append("")

    valid = [row for row in results if row.get("valid_t2_1")]
    if valid:
        lines.append("Per-cluster metrics:")
        lines.append(f"{'ID':<30} {'Contact':>10} {'RMSD rec':>10} {'Scale':>8}")
        lines.append("-" * 62)
        for row in valid:
            mid = row.get("metrics", {})
            lines.append(
                f"{row['id']:<30} {mid.get('mean_held_out_contact_recovery', 0):>10.4f} "
                f"{mid.get('mean_rmsd_recovery_angstrom', 0):>10.4f} "
                f"{row.get('torsion_scale_degrees', 0):>8.2f}")

    failed_tier = [row for row in results if not row.get("torsion_hit_tier")]
    if failed_tier:
        lines.append("")
        lines.append(f"Failed torsion tier ({len(failed_tier)} records):")
        for row in failed_tier:
            lines.append(f"  {row['id']}")

    failed_valid = [row for row in results
                    if row.get("torsion_hit_tier") and not row.get("valid_t2_1")]
    if failed_valid:
        lines.append("")
        lines.append(f"Tier hit but failed validation ({len(failed_valid)} records):")
        for row in failed_valid:
            lines.append(f"  {row['id']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=str(ROOT / "results_t2.1_v2_final.json"))
    parser.add_argument(
        "--out",
        default=str(ROOT / "results/publication/h3_t2.1_v2_posthoc/report.txt"))
    args = parser.parse_args()
    data = load_results(args.results)
    config_path = ROOT / "configs/benchmarks/peptide_h3_t2.1_v2_temporal_final.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = load_results(ROOT / config["split_audit"])
    cluster_by_id = {
        record["id"]: record["axis_values"]["official_antigen_cluster"][0]
        for record in audit["records"][config["benchmark_split"]]
    }
    aggregate = corrected_aggregate(data["results"], config, cluster_by_id)
    report = cluster_table(data["results"], aggregate)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    decision = {
        "schema_version": 1,
        "benchmark": "h3_t2.1_v2_posthoc_corrected_analysis",
        "status": "post_hoc_reanalysis_of_immutable_saved_results",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "aggregate": aggregate,
        "gate_passed": aggregate["gate_pass"],
        "decision": ("T2.1 overall validity gate passed, but contact-specific guidance "
                     "is unsupported because paired contrasts favor all-contact and random controls"),
    }
    decision_path = out_path.parent / "final_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="ascii")
    print(f"\nDecision written to {decision_path}")


if __name__ == "__main__":
    main()
