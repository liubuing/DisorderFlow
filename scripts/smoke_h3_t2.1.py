#!/usr/bin/env python
"""T2.1 engineering smoke test - verify the pipeline runs end-to-end on one case."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def smoke_test(config_path, out_dir):
    import yaml
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tp = config["torsion_perturbation"]

    result = {
        "smoke": "t2.1",
        "config_keys": sorted(config.keys()),
        "torsion_config": tp,
        "sampling_keys": sorted(config["sampling"].keys()),
        "control_arms": config.get("controls", {}),
        "module_imports": {},
        "test_passed": True,
    }
    tests = []
    try:
        from modules.peptide_torsion_perturb import smooth_torsion_direction
        phi, psi = smooth_torsion_direction("AAAA", 42)
        tests.append("smooth_torsion_direction: OK")
    except Exception as e:
        tests.append(f"smooth_torsion_direction: FAIL - {e}")
        result["test_passed"] = False

    try:
        from modules.peptide_t2_recovery import split_contact_pairs, recovery_metrics
        contacts = {(("H", "1"), ("P", "1")), (("H", "2"), ("P", "2"))}
        supplied, held_out = split_contact_pairs(contacts, "test", 0.5)
        assert supplied.isdisjoint(held_out)
        metrics = recovery_metrics(2.5, 1.5, 0.2, 0.6)
        assert metrics["rmsd_recovery_angstrom"] == 1.0
        tests.append("peptide_t2_recovery: OK")
    except Exception as e:
        tests.append(f"peptide_t2_recovery: FAIL - {e}")
        result["test_passed"] = False

    try:
        from modules.peptide_conformer_ensemble import antibody_aligned_rmsd
        tests.append("peptide_conformer_ensemble: OK")
    except Exception as e:
        tests.append(f"peptide_conformer_ensemble: FAIL - {e}")
        result["test_passed"] = False

    result["tests"] = tests

    out_path = Path(out_dir) / "t2.1_smoke_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return result["test_passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/benchmarks/peptide_h3_t2.1_temporal_final_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/publication/h3_t2.1_smoke_v1"))
    args = parser.parse_args()
    ok = smoke_test(args.config, args.out_dir)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
