#!/usr/bin/env python
"""IDP-targeted antibody design research platform (honest capability framing).

Wires the existing disorder -> segment -> design -> AF2 validation loop and
applies the deployment-grade confidence policy: rank by interface PAE only,
report pLDDT/ipTM but flag them as abstained. Emits a report that never
conflates tool existence with design effectiveness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))

from honest_confidence import rank_designs_honest  # noqa: E402

CAPABILITY_MANIFEST = "publication/idp_platform_capability_v1.json"
PAE_DEPLOYMENT_CONTRACT = "publication/candidate_interface_pae_deployment_v1.json"
CLAIM_BOUNDARY = (
    "Research tool for IDP-targeted antibody design exploration. It does not "
    "claim to produce validated IDP binders. No wet-lab validation was performed."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_platform(target_pdb, target_chain, scaffold_pdb, scaffold_chain,
                 samples, segments, disorder_threshold, use_af2, device, output_dir):
    from idp_antibody_design import run_idp_antibody_design

    result = run_idp_antibody_design(
        target_pdb=str(target_pdb), target_chain=target_chain,
        scaffold_pdb=str(scaffold_pdb), scaffold_chain=scaffold_chain,
        disorder_threshold=disorder_threshold, num_segments=segments,
        num_samples=samples, stochastic=True, output_dir=str(output_dir),
        device=device, verbose=True, use_af2=use_af2)

    honest_ranked = rank_designs_honest(result.get("segment_results", []))
    return result, honest_ranked


def write_report(output_dir: Path, result: dict, honest_ranked: list, meta: dict):
    capability = json.loads((ROOT / CAPABILITY_MANIFEST).read_text(encoding="utf-8"))
    report = {
        "schema_version": "idp_platform_report_v1",
        "classification": "development_only",
        "generated_at_utc": iso_utc(),
        "claim_boundary": CLAIM_BOUNDARY,
        "capability_manifest_sha256": sha256(ROOT / CAPABILITY_MANIFEST),
        "pae_deployment_contract_sha256": sha256(ROOT / PAE_DEPLOYMENT_CONTRACT),
        "inputs": meta,
        "confidence_policy": {
            "deployment_grade": ["pae"],
            "reported_but_abstained": ["plddt", "iptm"],
        },
        "disorder": {
            "mean_disorder": (
                float(result["disorder_result"]["disorder_scores"].mean())
                if result.get("disorder_result") else None),
            "n_segments": len(result.get("segments", [])),
        },
        "ranked_designs": honest_ranked,
        "n_designs": len(honest_ranked),
    }
    path = output_dir / "platform_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="IDP target PDB")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--scaffold", required=True, help="antibody scaffold PDB")
    parser.add_argument("--scaffold-chain", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--af2", action="store_true", help="enable AF2 validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result, honest_ranked = run_platform(
        args.target, args.target_chain, args.scaffold, args.scaffold_chain,
        args.samples, args.segments, args.threshold, args.af2, args.device,
        output_dir)

    report, path = write_report(output_dir, result, honest_ranked, {
        "target": args.target, "target_chain": args.target_chain,
        "scaffold": args.scaffold, "scaffold_chain": args.scaffold_chain,
        "samples": args.samples, "segments": args.segments,
        "threshold": args.threshold, "use_af2": args.af2,
    })
    print(json.dumps({
        "n_designs": report["n_designs"],
        "confidence_policy": report["confidence_policy"],
        "claim_boundary": report["claim_boundary"],
        "report": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
