"""Prepare score-blind near-native controls and audit structural eligibility."""
from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_ecls_pae_joint import digest, read, write
from scripts.evaluate_mpnn_likelihood_baseline import chain_residues, locate_h3

OUT = ROOT / "results/ecls_pae_evidence_dev_v1"


def near_native_controls(sequence, count, seed):
    variants = {}
    for i, j in itertools.combinations(range(len(sequence)), 2):
        if sequence[i] == sequence[j]:
            continue
        v = list(sequence)
        v[i], v[j] = v[j], v[i]
        variants.setdefault("".join(v), (i + 1, j + 1))
    keys = sorted(variants)
    selected = np.random.default_rng(seed).choice(len(keys), min(count, len(keys)), replace=False)
    result = []
    for idx in selected:
        seq = keys[int(idx)]
        assert Counter(seq) == Counter(sequence)
        assert sum(a != b for a, b in zip(seq, sequence)) == 2
        result.append({"h3_sequence": seq, "swapped_positions_1_indexed": variants[seq], "identity_to_native": 1 - 2 / len(seq)})
    return result


def ca_coordinates(pdb, chain):
    coords = {}
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[21] == chain and line[12:16].strip() == "CA" and line[16] in " A":
            coords.setdefault((int(line[22:26]), line[26].strip()), [float(line[i:i + 8]) for i in (30, 38, 46)])
    return coords


def main():
    protocol_path = ROOT / "configs/benchmarks/ecls_hard_controls_dev_v1.json"
    protocol = read(protocol_path)
    metadata_path = ROOT / protocol["component_source"]
    source_path = ROOT / protocol["candidate_source"]
    source = read(source_path)
    components = {r["scaffold"] for r in source["rows"]}
    metadata = {r["component_id"]: r for r in read(metadata_path)["components"]}
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in (protocol_path, metadata_path, source_path, Path(__file__).resolve())}
    queue = []
    for index, scaffold in enumerate(sorted(components)):
        meta = metadata[scaffold]
        pdb = ROOT / f"data/multiscaffold_confirmatory_v2/generation_work/structures/{scaffold}.pdb"
        hashes[pdb.relative_to(ROOT).as_posix()] = digest(pdb)
        chains = sorted({line[21] for line in pdb.read_text().splitlines() if line.startswith("ATOM")})
        rr = {c: chain_residues(pdb, c) for c in chains}
        seqs = {c: "".join(r[2] for r in rs) for c, rs in rr.items()}
        heavy = [c for c, seq in seqs.items() if meta["vh_sequence"] in seq]
        antigen = [c for c, seq in seqs.items() if meta["antigen_sequence"] == seq]
        if len(heavy) != 1 or len(antigen) != 1 or heavy == antigen:
            raise ValueError("Ambiguous chain roles")
        hc, ac = heavy[0], antigen[0]
        hi = locate_h3(rr[hc], meta["cdr_h3_sequence"])
        hca, aca = ca_coordinates(pdb, hc), ca_coordinates(pdb, ac)
        h3_ca = np.array([hca[rr[hc][i][:2]] for i in hi])
        positions = [r[:2] for r in rr[ac]]
        antigen_ca = np.array([aca[p] for p in positions])
        distances = np.linalg.norm(antigen_ca[:, None] - h3_ca[None, :], axis=2).min(1)
        contact = [p for p, d in zip(positions, distances) if d < 8]
        remote = [p for p, d in zip(positions, distances) if d > 16]
        queue.append({"scaffold": scaffold, "pdb": pdb.relative_to(ROOT).as_posix(), "heavy_chain": hc, "antigen_chain": ac,
            "native_h3": meta["cdr_h3_sequence"],
            "controls": near_native_controls(meta["cdr_h3_sequence"], protocol["controls_per_component"], protocol["seed"] + index),
            "structural_audit": {"contact_residues": contact, "remote_residues": remote,
                "max_matched_residues": min(len(contact), len(remote)),
                "contact_remote_comparison_eligible": bool(contact and remote),
                "antigen_min_H3_CA_distances": distances.tolist()}})
    target = OUT / "hard_controls_manifest.json"
    result = {"status": "prepared_not_scored", "protocol": protocol, "source_sha256": hashes, "components": queue}
    if target.exists() and read(target) != __import__("json").loads(__import__("json").dumps(result)):
        raise ValueError("Prepared controls changed; use a new version")
    write(target, result)
    print(f"Prepared {sum(len(r['controls']) for r in queue)} near-native controls; {sum(r['structural_audit']['contact_remote_comparison_eligible'] for r in queue)}/{len(queue)} components have contact and remote antigen residues.")


if __name__ == "__main__":
    main()
