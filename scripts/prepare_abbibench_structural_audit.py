"""Prepare a label-free, version-pinned AbBiBench structural overlap audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build.materialize_rcsb_candidate_interface_extension import number_sequences
from scripts.audit_successor_v3_isolation import AXES, connected_components, run_mmseqs_search, mmseqs_version, write_fasta

SOURCE = ROOT / "data/ecls_independent_feasibility_v1/abbibench_metadata"
OUT = ROOT / "data/abbibench_structural_audit_v1"
MMSEQS = "C:/biological/Metabolic model prediction/Integrated_Yeast_MetaTwin_Deployment/tools/mmseqs2/mmseqs/bin/mmseqs.exe"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, data):
    text = json.dumps(data, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Frozen output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def resolve_path(requested, files):
    requested = requested.removeprefix("./data/")
    if requested in files:
        return requested, "exact"
    matches = [f for f in files if f.casefold() == requested.casefold()]
    if len(matches) == 1:
        return matches[0], "unique_case_correction"
    return None, "missing_or_ambiguous"


def prepare():
    info, metadata = read(SOURCE / "repository.json"), read(SOURCE / "metadata.json")
    files = [r["rfilename"] for r in info["siblings"]]
    mappings, grouped = [], {}
    for key, item in sorted(metadata.items()):
        path, status = resolve_path(item["pdb_path"], files)
        tables = [{"requested": name, "resolved": resolve_path(name, files)[0]} for name in item["affinity_data"]]
        warnings = []
        if item.get("epitope_chain") not in item["antigen_chains"]:
            warnings.append("epitope_chain_conflicts_with_antigen_chains")
        if item.get("paratope_chain") not in [item["heavy_chain"], item["light_chain"]]:
            warnings.append("paratope_chain_conflicts_with_antibody_chains")
        mapping = {"dataset": key, "structure": path, "path_resolution": status,
                   "tables": tables, "metadata_warnings": warnings}
        mappings.append(mapping)
        if path is None:
            continue
        signature = (path, item["heavy_chain"], item["light_chain"], tuple(item["antigen_chains"]))
        grouped.setdefault(signature, []).append(key)
    freeze(OUT / "mapping.json", {"revision": info["sha"], "rows": mappings,
            "source_hashes": {p.name: digest(p) for p in (SOURCE / "metadata.json", SOURCE / "repository.json")}})
    raw = []
    numbering_inputs = {}
    for index, ((remote, heavy, light, antigen), datasets) in enumerate(sorted(grouped.items()), 1):
        path = OUT / "structures" / Path(remote).name
        if not path.exists():
            url = f"https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset/resolve/{info['sha']}/{remote}"
            payload = urllib.request.urlopen(url, timeout=60).read()
            if b"ATOM " not in payload:
                raise ValueError(f"Not a coordinate file: {remote}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        model = PDBParser(QUIET=True).get_structure(path.stem, str(path))[0]
        ids = [heavy, light, *antigen]
        if len(ids) != len(set(ids)) or any(c not in model for c in ids):
            raise ValueError(f"Missing/overlapping role chains: {remote}")
        residues = {c: [r for r in model[c] if r.id[0] == " " and "CA" in r] for c in ids}
        sequences = {c: "".join(seq1(r.resname) for r in residues[c]) for c in ids}
        if any(not s or set(s) - set("ACDEFGHIKLMNPQRSTVWY") for s in sequences.values()):
            raise ValueError(f"Empty/noncanonical sequence: {remote}")
        instance = f"ABB{index:03d}"
        for role, chain in (("H", heavy), ("L", light)):
            numbering_inputs[f"{instance}|{role}"] = sequences[chain]
        raw.append({"instance": instance, "file": remote, "sha256": digest(path), "datasets": datasets,
                    "heavy_chain": heavy, "light_chain": light, "antigen_chains": list(antigen),
                    "sequences": sequences, "residues": residues})
    cache = OUT / "numbering.json"
    if cache.exists():
        numbered = read(cache)
    else:
        numbered = number_sequences(numbering_inputs)
        freeze(cache, numbered)
    records = []
    for row in raw:
        instance = row["instance"]
        h, l = numbered.get(instance + "|H"), numbered.get(instance + "|L")
        if not h or not l or h["chain_type"] != "H" or l["chain_type"] not in ("K", "L"):
            raise ValueError(f"Antibody chain roles/numbering failed: {instance}")
        seqs = row["sequences"]
        hr = row["residues"][row["heavy_chain"]]
        h3 = h["h3_indices"]
        if not h3 or max(h3) >= len(hr):
            raise ValueError(f"H3 indexing failed: {instance}")
        ag = [r for c in row["antigen_chains"] for r in row["residues"][c]]
        a = np.array([hr[i]["CA"].coord for i in h3])
        b = np.array([r["CA"].coord for r in ag])
        distances = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
        cdrs = {**h["cdrs"], **l["cdrs"]}
        stem = Path(row["file"]).stem.split("_")[0].lower()
        records.append({k: v for k, v in row.items() if k != "residues"} | {
            "pdb_id_from_filename_unverified": stem if len(stem) == 4 and stem[0].isdigit() else None,
            "vh_sequence": h["sequence"], "vl_sequence": l["sequence"],
            "cdr_h3_sequence": cdrs["H3"], "h3_indices_zero_based_observed_heavy": h3,
            "h3_residue_ids": [list(hr[i].id) for i in h3],
            "paired_cdr_sequence": "".join(cdrs[k] for k in ("H1", "H2", "H3", "L1", "L2", "L3")),
            "antigen_sequence": "".join(seqs[c] for c in row["antigen_chains"]),
            "antigen_chain_sequences": {c: seqs[c] for c in row["antigen_chains"]},
            "minimum_h3_antigen_CA_distance": float(distances.min()),
            "n_h3_positions_CA_within_8A": int((distances.min(axis=1) < 8).sum()),
            "backbone_complete_by_chain": {c: all(all(a in r for a in ("N", "CA", "C", "O")) for r in rr) for c, rr in row["residues"].items()},
        })
    freeze(OUT / "structural_manifest.json", {"classification": "label_free_functional_extension_structures",
            "revision": info["sha"], "records": records, "labels_accessed": False,
            "pretraining_independence": "unknown", "pdb_provenance_for_AAYL": "not_established",
            "numbering_input_sha256": hashlib.sha256(json.dumps(numbering_inputs, sort_keys=True).encode()).hexdigest()})
    print(json.dumps({"structures": len(records), "mapped_datasets": len(mappings)}, indent=2), flush=True)


def audit():
    records = read(OUT / "structural_manifest.json")["records"]
    reference_path = SOURCE.parent / "reference_union.json"
    reference = read(reference_path)
    work = OUT / "search"
    work.mkdir(exist_ok=True)
    hits, self_hits, commands = {}, {}, []
    version, _ = mmseqs_version(MMSEQS)
    for axis, (field, threshold) in AXES.items():
        candidates = records
        aliases = {r["instance"]: r["instance"] for r in records}
        if axis == "antigen":
            candidates, aliases = [], {}
            for row in records:
                for chain, seq in row["antigen_chain_sequences"].items():
                    identifier = row["instance"] + "_" + chain
                    candidates.append({"instance": identifier, field: seq})
                    aliases[identifier] = row["instance"]
        q, t = work / f"{axis}_query.fa", work / f"{axis}_reference.fa"
        if not q.exists():
            write_fasta(q, candidates, field, "instance")
        if not t.exists():
            write_fasta(t, reference["records"], field, "reference_id")
        for name, target, bucket in (("reference", t, hits), ("self", q, self_hits)):
            cache = work / f"{axis}_{name}.json"
            if cache.exists():
                payload = read(cache)
            else:
                rows, command = run_mmseqs_search(MMSEQS, q, target, work / f"{axis}_{name}.tsv", work / f"{axis}_{name}_work", threshold, 2)
                payload = {"hits": rows, "command": command}
                freeze(cache, payload)
            commands.append(payload["command"])
            bucket[axis] = [{**r, "query": aliases[r["query"]],
                            "target": aliases[r["target"]] if name == "self" else r["target"]} for r in payload["hits"]]
    rows = []
    for row in records:
        rid = row["instance"]
        failed = [axis for axis in AXES if any(h["query"] == rid for h in hits[axis])]
        exact = row["pdb_id_from_filename_unverified"] in reference["exact_exposed_pdb_ids"]
        rows.append({"instance": rid, "datasets": row["datasets"], "failed_axes": failed,
                     "exact_filename_pdb_overlap": exact,
                     "passes_project_sequence_screen": not failed and not exact,
                     "best_reference_hits": {axis: sorted([h for h in hits[axis] if h["query"] == rid], key=lambda h: -h["identity"])[:3] for axis in AXES}})
    passed = [r["instance"] for r in rows if r["passes_project_sequence_screen"]]
    freeze(OUT / "isolation_audit.json", {"classification": "functional_extension_overlap_audit_not_confirmatory_validation",
        "reference_sha256": digest(reference_path), "manifest_sha256": digest(OUT / "structural_manifest.json"),
        "thresholds": {a: {"identity": v[1], "coverage": .8, "coverage_mode": 0} for a, v in AXES.items()},
        "antigen_query_policy": "each antigen chain searched separately; any hit excludes the structure",
        "mmseqs_version": version, "commands": commands, "audit": rows,
        "all_candidate_components": connected_components([r["instance"] for r in records], self_hits),
        "passing_components": connected_components(passed, self_hits),
        "counts": {"structures": len(records), "passing_project_sequence_screen": len(passed)},
        "limitations": ["MMseqs heuristic search; no hit is not proof of no homology", "AAYL original structure provenance unresolved", "pretraining overlap unknown", "observed coordinates may omit residues", "no individual affinity labels or model scores accessed"]})
    print(json.dumps({"passing_sequence_screen": len(passed), "rows": [{k: v for k, v in r.items() if k != "best_reference_hits"} for r in rows]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "audit"), default="prepare")
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else audit()
