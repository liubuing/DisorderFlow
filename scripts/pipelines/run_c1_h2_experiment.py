#!/usr/bin/env python
"""C1 H2 experiment: disorder-guided design vs disorder-agnostic design.

Wires disorder profile -> epitope positioning -> BFN CDR design (disorder_guided
ON/OFF) -> AF2 interface PAE, testing preregistration
``publication/idp_rd_c1_preregistration.json`` H2.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))

from bfn_loader import run_bfn_design  # noqa: E402
from idp_disorder_analysis import predict_disorder  # noqa: E402
from epitope_structure_builder import build_epitope_structure  # noqa: E402
from antibody_epitope_complex import position_epitope  # noqa: E402

DEFAULT_TARGET = "data/misfolding_targets/2NAO_model1_A_1-42.pdb"
DEFAULT_SCAFFOLD = "data/misfolding_targets/5IMK.pdb"


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rename_chain(pdb_path: str, from_chain: str, to_chain: str) -> str:
    from Bio.PDB import PDBIO, PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    model = structure[0]
    if from_chain not in model:
        return pdb_path
    chain = model[from_chain]
    chain.id = to_chain
    out = pdb_path.replace(".pdb", f"_{to_chain}.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(out)
    return out


def design_candidates(model, config, target_pdb, target_chain, scaffold_pdb,
                      scaffold_chain, ep_start, ep_end, samples, device,
                      output_dir, disorder_guided, profile):
    from idp_disorder_analysis import predict_disorder as _pd

    if profile is None:
        disorder_result = _pd(model, config, target_pdb, target_chain, device)
        full_scores = disorder_result["disorder_scores"]
        target_seq = disorder_result["sequence"]
    else:
        full_scores = None
        target_seq = None

    segment = {"start": ep_start, "end": ep_end}
    epi_info = build_epitope_structure(
        segment, target_seq=target_seq, target_pdb=target_pdb,
        chain_id=target_chain, output_dir=output_dir)
    epitope_pdb = rename_chain(epi_info["pdb_path"], "B", "P")
    complex_info = position_epitope(
        scaffold_pdb, epitope_pdb, scaffold_chain=scaffold_chain,
        epitope_chain="P", distance=6.0, output_dir=output_dir)

    if profile is None:
        profile = [float(full_scores[i]) for i in range(ep_start - 1, ep_end)]

    cdr_spec = f"{scaffold_chain}:26-33,51-58,97-113"
    designs = run_bfn_design(
        complex_info["pdb_path"], cdr_spec, num_samples=samples,
        stochastic=True, context_chains=["P"], device=device,
        disorder_guided=disorder_guided,
        epitope_disorder_profile=profile if disorder_guided else None,
        antigen_chains=["P"],
    )
    return {
        "designs": designs,
        "complex_pdb": complex_info["pdb_path"],
        "epitope_pdb": epi_info["pdb_path"],
        "epitope_sequence": epi_info["sequence"],
        "epitope_disorder_profile": profile,
        "disorder_guided": disorder_guided,
    }


def validate(designs, epitope_sequence, num_recycle=3):
    import subprocess

    jobs = []
    for index, design in enumerate(designs):
        ab_seq = design.get("full_heavy_sequence") or design.get("sequence")
        if not ab_seq:
            jobs.append(None)
            continue
        jobs.append({
            "seq": ab_seq,
            "epi_seq": epitope_sequence,
            "id": index,
            "recycle": num_recycle,
            "seed": 42,
        })
    payload = "".join(
        json.dumps(job) + "\n" for job in jobs if job is not None)
    command = (
        "cd /mnt/d/biological/DisorderFlow && "
        "source venv_wsl/bin/activate && "
        f"python scripts/utils/af2_wsl_batch.py --recycle {num_recycle} "
        "--model-number 1"
    )
    completed = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
        input=payload, capture_output=True, text=True,
        timeout=600 + 600 * len(jobs))
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    results = {}
    for line in completed.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in row:
            results[row["id"]] = row

    out = []
    for index, design in enumerate(designs):
        row = results.get(index, {})
        out.append({
            "interface_pae": row.get("interface_pae"),
            "iptm_reported": row.get("iptm"),
            "plddt_reported": row.get("plddt"),
            "success": row.get("success"),
        })
    return out


def summarize(name, validated):
    paes = [v["interface_pae"] for v in validated if v.get("interface_pae") is not None]
    if not paes:
        return {"name": name, "n": len(validated), "median_pae": None,
                "min_pae": None, "per_design_pae": [None] * len(validated)}
    return {
        "name": name,
        "n": len(validated),
        "median_pae": float(np.median(paes)),
        "min_pae": float(min(paes)),
        "per_design_pae": [v.get("interface_pae") for v in validated],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--scaffold", default=DEFAULT_SCAFFOLD)
    parser.add_argument("--scaffold-chain", default="B")
    parser.add_argument("--epitope", nargs=2, type=int, default=[16, 24])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--design-only", action="store_true",
                        help="skip AF2 validation (smoke test)")
    args = parser.parse_args()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    from idp_antibody_design import _load_bfn

    model, config = _load_bfn()

    on = design_candidates(
        model, config, args.target, args.target_chain, args.scaffold,
        args.scaffold_chain, args.epitope[0], args.epitope[1], args.samples,
        args.device, output_dir, disorder_guided=True, profile=None)
    off = design_candidates(
        model, config, args.target, args.target_chain, args.scaffold,
        args.scaffold_chain, args.epitope[0], args.epitope[1], args.samples,
        args.device, output_dir, disorder_guided=False, profile=on["epitope_disorder_profile"])

    report = {
        "schema_version": "idp_c1_h2_experiment_v1",
        "classification": "development_only",
        "generated_at_utc": iso_utc(),
        "preregistration": "publication/idp_rd_c1_preregistration.json",
        "hypothesis": "h2_disorder_guided_design",
        "inputs": {
            "target": args.target, "scaffold": args.scaffold,
            "epitope": args.epitope, "samples": args.samples,
        },
        "epitope_disorder_profile": on["epitope_disorder_profile"],
        "epitope_sequence": on["epitope_sequence"],
        "confidence_policy": "pae_only; plddt/iptm reported but abstained",
    }

    if not args.design_only:
        report["disorder_guided_on"] = summarize(
            "on", validate(on["designs"], on["epitope_sequence"]))
        report["disorder_guided_off"] = summarize(
            "off", validate(off["designs"], off["epitope_sequence"]))
    else:
        report["disorder_guided_on"] = {"n": len(on["designs"])}
        report["disorder_guided_off"] = {"n": len(off["designs"])}

    report["claim_boundary"] = (
        "H2 exploratory factor comparison only; no validated binder claim; "
        "success does not establish IDP design effectiveness")
    Path(args.output).write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
