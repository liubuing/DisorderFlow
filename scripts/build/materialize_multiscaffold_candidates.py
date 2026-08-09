#!/usr/bin/env python
"""Download and structurally annotate newly discovered SAbDab peptide complexes."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[2]
API = "https://sabdab.opig.stats.ox.ac.uk/api/download/structures"
AA = set("ACDEFGHIKLMNPQRSTVWY")
CHOTHIA_CDR_RANGES = {
    "H": {"H1": (26, 32), "H2": (52, 56), "H3": (93, 102)},
    "L": {"L1": (24, 34), "L2": (50, 56), "L3": (89, 97)},
}


def chain_sequence(chain):
    residues = []
    for residue in chain.get_residues():
        if residue.id[0] != " " or "CA" not in residue:
            continue
        aa = seq1(residue.resname, custom_map={"MSE": "M"})
        if aa in AA:
            residues.append({
                "aa": aa,
                "auth_seq_id": int(residue.id[1]),
                "insertion_code": residue.id[2].strip(),
            })
    return residues


def download_crop(row, path, retries=3):
    body = json.dumps({
        "structures": [{
            "pdb_id": f"pdb_0000{row['pdb_id']}",
            "full_structure": False,
            "antibody_instances": [{
                "heavy_chain": row["heavy_chain"],
                "light_chain": row["light_chain"],
                "include_antigen": True,
            }],
        }],
    }).encode("ascii")
    request = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if not payload.startswith(b"data_"):
                raise RuntimeError("SAbDab response is not an mmCIF file")
            path.write_bytes(payload)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def number_variable_domains(rows):
    from anarcii import Anarcii

    model = Anarcii(seq_type="antibody", mode="accuracy", batch_size=32,
                    cpu=True, ncpu=1, verbose=False)
    sequences = {}
    for row in rows:
        sequences[f"{row['instance']}|H"] = row["heavy_sequence"]
        sequences[f"{row['instance']}|L"] = row["light_sequence"]
    numbered = model.number(sequences)
    numbered = model.to_scheme("chothia")
    domains = {}
    for identifier, result in numbered.items():
        if not result or result.get("error"):
            continue
        instance, role = identifier.rsplit("|", 1)
        query_index = int(result["query_start"])
        domain_sequence = []
        cdrs = {name: [] for name in CHOTHIA_CDR_RANGES[role]}
        indices = []
        for (position, _insertion), aa in result["numbering"]:
            if aa == "-":
                continue
            domain_sequence.append(aa)
            for cdr_name, (start, end) in CHOTHIA_CDR_RANGES[role].items():
                if start <= int(position) <= end:
                    cdrs[cdr_name].append(aa)
            if role == "H" and 93 <= int(position) <= 102:
                indices.append(query_index)
            query_index += 1
        domains.setdefault(instance, {})[role] = {
            "sequence": "".join(domain_sequence),
            "chain_type": result["chain_type"],
            "cdrs": {name: "".join(sequence) for name, sequence in cdrs.items()},
            "h3_indices": indices,
        }
    return domains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery", default="data/multiscaffold_confirmatory_v1/discovery.json")
    parser.add_argument(
        "--out-dir", default="data/multiscaffold_confirmatory_v1")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    discovery = json.loads((ROOT / args.discovery).read_text(encoding="utf-8"))
    candidates = discovery["candidates"]
    if args.max_records is not None:
        candidates = candidates[:args.max_records]
    out_dir = ROOT / args.out_dir
    cif_dir = out_dir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    parser_cif = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    annotated = []
    exclusions = []
    for index, row in enumerate(candidates, 1):
        cif_path = cif_dir / f"{row['instance']}.cif"
        try:
            if not cif_path.exists():
                download_crop(row, cif_path)
            structure = parser_cif.get_structure(row["instance"], str(cif_path))
            model = next(structure.get_models())
            heavy = chain_sequence(model[row["heavy_chain"]])
            light = (chain_sequence(model[row["light_chain"]])
                     if row["light_chain"] in model else [])
            antigen_options = []
            for chain_id in row["antigen_chains"]:
                if chain_id not in model:
                    continue
                residues = chain_sequence(model[chain_id])
                if 5 <= len(residues) <= 50:
                    antigen_options.append((chain_id, residues))
            if len(heavy) < 90 or len(light) < 80 or len(antigen_options) != 1:
                raise ValueError(
                    f"requires paired variable chains and exactly one 5-50 aa peptide; "
                    f"observed H={len(heavy)}, L={len(light)}, peptides={len(antigen_options)}")
            antigen_chain, antigen = antigen_options[0]
            annotated.append({
                **row,
                "cif_path": cif_path.relative_to(ROOT).as_posix(),
                "heavy_sequence": "".join(item["aa"] for item in heavy),
                "light_sequence": "".join(item["aa"] for item in light),
                "antigen_chain": antigen_chain,
                "antigen_sequence": "".join(item["aa"] for item in antigen),
            })
        except Exception as error:  # noqa: BLE001
            exclusions.append({"instance": row["instance"], "reason": str(error)})
        print(f"Structured {index}/{len(candidates)}", flush=True)

    domains_by_id = {}
    if annotated:
        domains_by_id = number_variable_domains(annotated)
    eligible = []
    for row in annotated:
        domains = domains_by_id.get(row["instance"], {})
        heavy_domain = domains.get("H", {})
        light_domain = domains.get("L", {})
        heavy_cdrs = heavy_domain.get("cdrs", {})
        light_cdrs = light_domain.get("cdrs", {})
        sequence = heavy_cdrs.get("H3", "")
        if not 3 <= len(sequence) <= 35:
            exclusions.append({
                "instance": row["instance"],
                "reason": f"invalid anchor-bounded Chothia H3: {sequence!r}"})
            continue
        if heavy_domain.get("chain_type") != "H" or light_domain.get("chain_type") not in {"K", "L"}:
            exclusions.append({
                "instance": row["instance"], "reason": "ANARCII variable-domain role mismatch"})
            continue
        eligible.append({
            **row,
            "vh_sequence": heavy_domain["sequence"],
            "vl_sequence": light_domain["sequence"],
            "cdr_sequences": {**heavy_cdrs, **light_cdrs},
            "paired_cdr_sequence": "".join(
                {**heavy_cdrs, **light_cdrs}[name]
                for name in ("H1", "H2", "H3", "L1", "L2", "L3")),
            "cdr_h3_sequence": sequence,
            "h3_heavy_indices_zero_based": heavy_domain["h3_indices"],
            "h3_definition": "Chothia positions 93-102, bounded by Cys92 and Trp103",
        })
    report = {
        "schema_version": 1,
        "status": "structure_annotation_complete; homology_audit_pending",
        "source_discovery": args.discovery,
        "n_discovered": len(candidates),
        "n_structurally_eligible": len(eligible),
        "n_excluded": len(exclusions),
        "numbering": {"tool": "ANARCII", "version": "2.0.8", "scheme": "chothia"},
        "records": eligible,
        "exclusions": exclusions,
    }
    (out_dir / "structural_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"records", "exclusions"}}, indent=2))


if __name__ == "__main__":
    main()
