"""ProteinMPNN fixed-backbone likelihood baseline vs AF2 interface PAE.

Scores every entity (designed + native + composition-shuffle) of each scaffold
with ProteinMPNN conditional log-likelihood on the design-input backbone from
data/multiscaffold_confirmatory_v2/generation_work/structures/ — the same
backbones used for candidate generation and AF2 labeling. Baseline estimand:
within-scaffold Spearman between MPNN H3 NLL and the AF2 entity-mean interface
PAE target (2 AF2 models x 3 seeds). Historical v1 used the runner's global
interface field, NOT the surrogate's localized target. Do not use v1 as a
same-target comparison; use harden_pae_surrogate_v4.py for that comparison.

Run: python scripts/evaluate_mpnn_likelihood_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AA3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"}
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF2_FILES = {
    "transfer": [
        "results/candidate_interface_multiscaffold_calibration_ext/af2/results.json",
        "results/candidate_interface_multiscaffold_calibration_ext/af2_model2/results.json"],
    "holdout": [
        "results/candidate_interface_multiscaffold_v1/af2/results.json",
        "results/candidate_interface_multiscaffold_v1/af2_model2/results.json"],
}
HOLDOUT_COMPONENTS = {"V2C001", "V2C002", "V2C003"}
WORK = ROOT / "results" / "candidate_interface_mpnn_baseline_v1"
STRUCTURES = ROOT / "data/multiscaffold_confirmatory_v2/generation_work/structures"


def chain_residues(pdb_path, chain_id):
    """Ordered [(resseq, icode, aa1)] for one chain."""
    seen = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain_id:
            continue
        aa = AA3TO1.get(line[17:20].strip())
        if aa is None:
            continue
        key = (int(line[22:26]), line[26].strip())
        seen.setdefault(key, aa)
    return [(k[0], k[1], v) for k, v in sorted(seen.items(), key=lambda x: (x[0][0], x[0][1]))]


def locate_h3(heavy_residues, h3_sequence, h3_indices=None):
    """Row indices (0-based, in chain order) of the H3 span."""
    seq = "".join(r[2] for r in heavy_residues)
    starts = [i for i in range(len(seq)) if seq.startswith(h3_sequence, i)]
    if len(starts) == 1:
        start = starts[0]
    elif h3_indices is not None and starts:
        start = next((s for s in starts if s == h3_indices[0]), starts[0])
    elif not starts:
        raise ValueError(f"H3 sequence not found in heavy chain: {h3_sequence}")
    else:
        raise ValueError(f"H3 sequence ambiguous: {len(starts)} occurrences")
    return list(range(start, start + len(h3_sequence)))


def run_mpnn(pdb_path, heavy_chain, fixed_positions, out_dir, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / "conditional_probs_only" / pdb_path.stem
    if result.with_suffix(".npz").exists():
        return np.load(result.with_suffix(".npz"))["log_p"]
    fixed_path = out_dir / "fixed_positions.jsonl"
    fixed_path.write_text(
        json.dumps({pdb_path.stem: {heavy_chain: fixed_positions}}) + "\n",
        encoding="ascii")
    command = [
        sys.executable,
        str((ROOT / "ProteinMPNN/protein_mpnn_run.py").resolve()),
        "--pdb_path", pdb_path.resolve().as_posix(),
        "--pdb_path_chains", heavy_chain,
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", (ROOT / "ProteinMPNN/vanilla_model_weights").resolve().as_posix(),
        "--model_name", "v_48_020",
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1",
        "--num_seq_per_target", "1",
        "--batch_size", "1",
        "--seed", str(seed),
        "--suppress_print", "1",
        "--out_folder", out_dir.resolve().as_posix(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode:
        raise RuntimeError(f"ProteinMPNN failed: {completed.stderr[-2000:]}")
    return np.load(result.with_suffix(".npz"))["log_p"]


def score_sequence(log_p, rows, sequence):
    return -sum(
        float(log_p[row, MPNN_ALPHABET.index(aa)])
        for row, aa in zip(rows, sequence)
    ) / len(sequence)


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    if (WORK / "mpnn_likelihood_baseline.json").exists():
        raise FileExistsError(
            "Frozen v1 baseline exists. Use harden_pae_surrogate_v4.py for "
            "a signed, same-target comparison without overwriting history.")
    WORK.mkdir(parents=True, exist_ok=True)

    ext = json.loads(
        (ROOT / "data/candidate_interface_external_calibration_v1/extension_manifest_v1.json").read_text())
    hold = json.loads(
        (ROOT / "data/multiscaffold_confirmatory_v2/holdout_manifest.json").read_text())
    scaffolds = {}
    for record in ext["components"]:
        scaffolds[record["component_id"]] = ("transfer", record)
    for record in hold["components"]:
        if record["component_id"] in HOLDOUT_COMPONENTS:
            scaffolds[record["component_id"]] = ("holdout", record)

    # AF2 entity targets and H3 sequences
    cohorts = {}
    for cohort, files in AF2_FILES.items():
        pae_acc = defaultdict(list)
        entities = {}
        for rel in files:
            d = json.loads((ROOT / rel).read_text())
            for e in d["entities"]:
                if cohort == "holdout" and e["component_id"] not in HOLDOUT_COMPONENTS:
                    continue
                entities[e["entity_id"]] = e
            for r in d["results"]:
                if r["status"] != "success":
                    continue
                if cohort == "holdout" and r["component_id"] not in HOLDOUT_COMPONENTS:
                    continue
                pae_acc[r["entity_id"]].append(r["interface_pae"])
        cohorts[cohort] = {
            "pae": {k: sum(v) / len(v) for k, v in pae_acc.items()},
            "entities": entities,
        }

    results = {}
    for component_id, (cohort, record) in sorted(scaffolds.items()):
        pdb_path = STRUCTURES / f"{component_id}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(pdb_path)
        if cohort == "holdout":
            rep = record["representative"]
            h3_sequence = rep["cdr_h3_sequence"]
            h3_indices = rep.get("h3_heavy_indices_zero_based")
            # work structures are uniformly relabeled (H/L/P); resolve chain
            # roles by sequence matching instead of manifest chain labels
            chain_ids = {line[21] for line in pdb_path.read_text().splitlines()
                         if line.startswith("ATOM")}
            seqs = {cid: "".join(r[2] for r in chain_residues(pdb_path, cid))
                    for cid in chain_ids}
            heavy_chain = next(c for c, s in seqs.items() if rep["vh_sequence"] in s)
            light_chain = next(c for c, s in seqs.items() if rep["vl_sequence"] in s)
        else:
            chain_ids = {line[21] for line in pdb_path.read_text().splitlines()
                         if line.startswith("ATOM")}
            seqs = {cid: "".join(r[2] for r in chain_residues(pdb_path, cid))
                    for cid in chain_ids}
            heavy_chain = next(c for c, s in seqs.items() if record["vh_sequence"] in s)
            light_chain = next(c for c, s in seqs.items() if record["vl_sequence"] in s)
            h3_sequence = record["cdr_h3_sequence"]
            h3_indices = None
        heavy_residues = chain_residues(pdb_path, heavy_chain)
        rows = locate_h3(heavy_residues, h3_sequence, h3_indices)
        fixed_positions = [r[0] for i, r in enumerate(heavy_residues) if i not in set(rows)]

        log_p = run_mpnn(pdb_path, heavy_chain, fixed_positions, WORK / component_id)
        if log_p.ndim == 3:
            log_p = log_p[0]

        cohort_data = cohorts[cohort]
        by_scaffold = {
            eid: e for eid, e in cohort_data["entities"].items()
            if e["component_id"] == component_id and eid in cohort_data["pae"]}
        pairs_all, pairs_designed = [], []
        per_entity = {}
        for eid, e in sorted(by_scaffold.items()):
            nll = score_sequence(log_p, rows, e["h3_sequence"])
            per_entity[eid] = {"h3_nll": nll,
                               "af2_interface_pae": cohort_data["pae"][eid],
                               "entity_type": e["entity_type"]}
            pair = (nll, cohort_data["pae"][eid])
            pairs_all.append(pair)
            if e["entity_type"] == "candidate":
                pairs_designed.append(pair)

        results[component_id] = {
            "cohort": cohort,
            "heavy_chain": heavy_chain,
            "light_chain": light_chain,
            "n_all": len(pairs_all),
            "n_designed": len(pairs_designed),
            "spearman_all": spearman([p[0] for p in pairs_all], [p[1] for p in pairs_all]),
            "spearman_designed": spearman(
                [p[0] for p in pairs_designed], [p[1] for p in pairs_designed]),
            "per_entity": per_entity,
        }
        r = results[component_id]
        print(f"{component_id} ({cohort}): n_all={r['n_all']} rho_all="
              f"{None if r['spearman_all'] is None else round(r['spearman_all'],3)} "
              f"n_des={r['n_designed']} rho_des="
              f"{None if r['spearman_designed'] is None else round(r['spearman_designed'],3)}")

    out = WORK / "mpnn_likelihood_baseline.json"
    out.write_text(json.dumps({
        "schema_version": "mpnn_likelihood_baseline_v1",
        "estimand": "within-scaffold Spearman(ProteinMPNN v_48_020 H3 NLL, AF2 entity-mean interface PAE)",
        "direction_note": "NLL: lower = MPNN prefers; PAE: lower = better. Positive rho aligns the desired orders. Negative rho means MPNN-preferred sequences tend to have HIGHER PAE. Never replace signed rho with its absolute value.",
        "backbones": "design-input backbones from data/multiscaffold_confirmatory_v2/generation_work/structures (identical to generation and AF2 labeling inputs)",
        "results": results,
    }, indent=2) + "\n", encoding="ascii")
    print("wrote", out)


if __name__ == "__main__":
    main()
