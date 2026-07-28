#!/usr/bin/env python
"""Run a StateContrast-Ab domain workflow from a generic YAML config."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_RUNNER = PROJECT_ROOT / "scripts" / "pipeline" / "run_statecontrast_ab_effect_guided_v4.py"
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from statecontrast_workflow import integration_claim_level, readiness_requirements, validate_reference_count  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run StateContrast domain workflow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    metadata = workflow_metadata(cfg)
    with open(out_dir / "statecontrast_domain_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if metadata["reference_check"]["status"] != "pass":
        raise SystemExit("Verified reference-count gate failed; workflow was not executed")
    whitelist = write_derived_whitelist(cfg, out_dir / "derived_reference_whitelist.yml")
    if not args.dry_run:
        run_v4(cfg, whitelist, out_dir / "v4_position_effect")
    write_domain_report(out_dir / "statecontrast_domain_workflow_report.md", cfg, metadata, out_dir / "v4_position_effect")
    print(f"Wrote StateContrast domain workflow to {out_dir}")
    print(f"claim_level={metadata['claim_level']} dry_run={args.dry_run}")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return normalize_config(cfg)


def normalize_config(cfg):
    cfg.setdefault("references", [])
    cfg.setdefault("negative_states", {})
    cfg.setdefault("candidate_generation", {})
    cfg.setdefault("statecontrast", {})
    cfg.setdefault("claim_gate", {})
    cfg.setdefault("fold_sidecheck", {})
    return cfg


def workflow_metadata(cfg):
    gate = cfg.get("claim_gate", {})
    reference_check = validate_reference_count(
        cfg.get("references", []),
        gate.get("minimum_references", 2),
        require_verified=True,
    )
    claim = integration_claim_level(
        reference_check["n_references"],
        bool(gate.get("has_real_negative_ensembles", False)),
        bool(gate.get("has_wetlab", False)),
    )
    return {
        "domain": cfg.get("domain", ""),
        "name": cfg.get("name", ""),
        "claim_level": claim,
        "reference_check": reference_check,
        "readiness_requirements": readiness_requirements(),
        "controls": cfg.get("statecontrast", {}).get("required_controls", []),
    }


def write_derived_whitelist(cfg, path):
    refs = []
    for ref in cfg.get("references", []):
        refs.append({
            "pdb": ref["pdb"],
            "path": str(PROJECT_ROOT / ref["complex_pdb"]),
            "peptide_chain": ref.get("peptide_chain"),
            "peptide_sequence": ref.get("peptide_sequence", ""),
            "heavy_chain": ref.get("heavy_chain"),
            "light_chain": ref.get("light_chain"),
            "reliability": ref.get("reliability"),
            "provenance_verified": ref.get("provenance_verified", False),
        })
    payload = {"abeta_references": refs}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def run_v4(cfg, whitelist, out_dir):
    sc = cfg.get("statecontrast", {})
    gen = cfg.get("candidate_generation", {})
    neg = cfg.get("negative_states", {})
    seeds = ",".join(str(s) for s in sc.get("seeds", [31, 41, 51, 61, 71]))
    cmd = [
        sys.executable,
        str(V4_RUNNER),
        "--whitelist", str(whitelist),
        "--seeds", seeds,
        "--samples-per-ref", str(gen.get("samples_per_ref", 150)),
        "--v4-samples-per-ref", str(gen.get("v4_samples_per_ref", 150)),
        "--max-mutations", str(gen.get("max_mutations", 4)),
        "--min-support", str(sc.get("min_support", 4)),
        "--min-effect", str(sc.get("min_effect", 0.004)),
        "--min-recurrence", str(sc.get("min_recurrence", 2)),
        "--rule-mode", str(sc.get("rule_mode", "position")),
        "--heldout-seed-offset", str(neg.get("heldout_seed_offset", 10000)),
        "--validation-mode", str(sc.get("validation_mode", "leave_seed_out")),
        "--out", str(out_dir),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def load_summary(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_domain_report(path, cfg, metadata, v4_dir):
    summary_rows = load_summary(v4_dir / "statecontrast_effect_guided_v4_summary.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast Domain Workflow Report\n\n")
        f.write(f"Domain: `{metadata['domain']}`\n\n")
        f.write(f"Config: `{metadata['name']}`\n\n")
        f.write(f"Claim level: `{metadata['claim_level']}`\n\n")
        f.write(f"References: `{metadata['reference_check']['n_references']}`\n\n")
        f.write("## Controls\n\n")
        for control in metadata.get("controls", []):
            f.write(f"- {control}\n")
        f.write("\n## Top10 Summary\n\n")
        f.write("| Ref | V4 Mean | V4-Parent | V4-Random | V4-Shuffled | Win Rate | Verdict |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for row in summary_rows:
            if row.get("metric") != "top10_mean_gap_delta":
                continue
            f.write(
                f"| {row['reference_pdb']} | {row['v4_mean']} | {row['v4_minus_parent_only_mean']} | "
                f"{row['v4_minus_random_mean']} | {row['v4_minus_shuffled_mean']} | "
                f"{row['v4_beats_parent_random_and_shuffled_rate']} | {row['verdict']} |\n"
            )
        f.write("\n## Interpretation\n\n")
        if cfg.get("statecontrast", {}).get("validation_mode") == "leave_reference_out":
            stable = [row["reference_pdb"] for row in summary_rows if row.get("metric") == "top10_mean_gap_delta" and row.get("verdict") == "stable_method_signal"]
            failed = [row["reference_pdb"] for row in summary_rows if row.get("metric") == "top10_mean_gap_delta" and row.get("verdict") == "mixed_or_overfit_risk"]
            f.write(f"Leave-one-reference-out transfer was stable for {len(stable)} of {len(stable) + len(failed)} references")
            if stable:
                f.write(f" ({', '.join(stable)})")
            f.write(". ")
            if failed:
                f.write(f"Transfer failed for {', '.join(failed)}. ")
            f.write("This is partial cross-reference evidence, not a generalization claim; real negative-state ensembles remain absent.\n")
        else:
            f.write("This domain workflow treats StateContrast as a state-sensitive position discovery layer. A-beta is a config instance, not the method boundary.\n")


if __name__ == "__main__":
    main()
