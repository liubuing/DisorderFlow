"""Development-only joint screening on design backbones; no AF2 execution."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mpnn_likelihood_baseline import (
    MPNN_ALPHABET, chain_residues, locate_h3, run_mpnn, score_sequence,
)

OUT = ROOT / "results/ecls_pae_joint_dev_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def screening(scores, target, fraction):
    """Expected top-20% recall with fractional membership for boundary ties."""
    def inclusion(values, count):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite screening input")
        cutoff = np.sort(values)[count - 1]
        below, tied = values < cutoff, values == cutoff
        return below.astype(float) + tied * ((count - below.sum()) / tied.sum())
    n = len(target)
    if not n or len(scores) != n or not 0 < fraction <= 1:
        raise ValueError("Invalid screening input")
    k, m = math.ceil(n * fraction), math.ceil(n * .2)
    return float(inclusion(scores, k) @ inclusion(target, m) / m)


def joint_scores(ecls, pae):
    if len(ecls) != len(pae) or not len(ecls):
        raise ValueError("Unmatched joint scores")
    # Lower ranks are preferred; ECLS gain is higher-is-better.
    return (rankdata(-np.asarray(ecls)) + rankdata(pae)) / 2


def main():
    protocol_path = ROOT / "configs/benchmarks/ecls_pae_joint_dev_v1.json"
    protocol = read(protocol_path)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists() and read(OUT / "protocol.json") != protocol:
        raise ValueError("Protocol changed; use a new development version")
    write(OUT / "protocol.json", protocol)
    manifest_path = ROOT / "results/pae_surrogate_revision_v4/data_manifest.json"
    prediction_path = ROOT / "results/pae_surrogate_revision_v4/predictions.json"
    metadata_path = ROOT / "data/candidate_interface_external_calibration_v1/extension_manifest_v1.json"
    manifest, pred = read(manifest_path), read(prediction_path)
    rows = [r for r in manifest["rows"] if r["split"] == "transfer"]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(pred["entities"]) or len(set(pred["entities"])) != len(pred["entities"]):
        raise ValueError("Candidate identity mismatch")
    lookup = {eid: i for i, eid in enumerate(pred["entities"])}
    metadata = {r["component_id"]: r for r in read(metadata_path)["components"]}
    provenance = {r["scaffold"]: r for r in manifest["provenance"]}
    artifacts = {p.relative_to(ROOT).as_posix(): digest(p) for p in (protocol_path, manifest_path, prediction_path, metadata_path, Path(__file__).resolve())}
    weights = ROOT / "ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    artifacts[weights.relative_to(ROOT).as_posix()] = digest(weights)
    scored = []
    for scaffold in sorted({r["scaffold"] for r in rows}):
        meta = metadata[scaffold]
        pdb = ROOT / f"data/multiscaffold_confirmatory_v2/generation_work/structures/{scaffold}.pdb"
        if provenance[scaffold]["pdb"] != pdb.relative_to(ROOT).as_posix() or digest(pdb) != provenance[scaffold]["sha256"]:
            raise ValueError("Design backbone mismatch")
        chains = sorted({line[21] for line in pdb.read_text().splitlines() if line.startswith("ATOM")})
        residues = {c: chain_residues(pdb, c) for c in chains}
        sequences = {c: "".join(r[2] for r in rr) for c, rr in residues.items()}
        def unique_chain(sequence, exact=False):
            matches = [c for c, seq in sequences.items() if (sequence == seq if exact else sequence in seq)]
            if len(matches) != 1:
                raise ValueError("Ambiguous design chain")
            return matches[0]
        hc, lc = [unique_chain(meta[k]) for k in ("vh_sequence", "vl_sequence")]
        ac = unique_chain(meta["antigen_sequence"], exact=True)
        if len({hc, lc, ac}) != 3:
            raise ValueError("Chain roles overlap")
        indices = locate_h3(residues[hc], meta["cdr_h3_sequence"])
        if [r[0] for r in residues[hc]] != list(range(1, len(residues[hc]) + 1)) or any(r[1] for r in residues[hc]):
            raise ValueError("Noncanonical residue numbering")
        fixed = [i + 1 for i in range(len(residues[hc])) if i not in indices]
        work = OUT / "work" / scaffold
        work.mkdir(parents=True, exist_ok=True)
        apo = work / f"{scaffold}_apo.pdb"
        apo.write_text("".join(line for line in pdb.read_text().splitlines(keepends=True) if not line.startswith(("ATOM", "HETATM", "TER")) or line[21:22] != ac), encoding="ascii")
        arrays = []
        for state, source in (("complex", pdb), ("apo", apo)):
            state_dir = work / state
            cache_identity = {"pdb": digest(source), "weights": digest(weights), "seed": protocol["mpnn_seed"], "fixed": fixed, "heavy": hc}
            cache_key = state_dir / "input_identity.json"
            if cache_key.exists() and read(cache_key) != cache_identity:
                raise ValueError("Stale ProteinMPNN cache")
            if state_dir.exists() and not cache_key.exists():
                raise ValueError("Unverified ProteinMPNN cache")
            write(cache_key, cache_identity)
            logp = run_mpnn(source, hc, fixed, state_dir, protocol["mpnn_seed"])
            if logp.ndim == 3:
                logp = logp[0]
            npz = state_dir / "conditional_probs_only" / f"{source.stem}.npz"
            with np.load(npz) as output:
                decoded = "".join(MPNN_ALPHABET[int(i)] for i in output["S"][:len(residues[hc])])
                if decoded != sequences[hc]:
                    raise ValueError("ProteinMPNN heavy-chain row alignment mismatch")
                if not np.all(output["mask"][indices] > 0) or not np.all(output["design_mask"][indices] > 0):
                    raise ValueError("Invalid H3 scoring mask")
            arrays.append(logp)
            for artifact in (source, npz):
                artifacts[artifact.relative_to(ROOT).as_posix()] = digest(artifact)
        for row in (r for r in rows if r["scaffold"] == scaffold):
            if len(row["h3_sequence"]) != len(indices):
                raise ValueError("Candidate length mismatch")
            i = lookup[row["id"]]
            if not np.isclose(row["target"], pred["target"][i], rtol=0, atol=1e-12):
                raise ValueError("AF2 target mismatch")
            complex_nll, apo_nll = [score_sequence(a, indices, row["h3_sequence"]) for a in arrays]
            scored.append({**row, "ecls": apo_nll - complex_nll, "mpnn_nll": complex_nll,
                "pae": float(np.mean([pred["models"][f"full_s{s}"][i] for s in (2041, 2053, 2069)])),
                "ridge": pred["models"]["ridge_alpha_10"][i]})
        print(f"{scaffold}: ECLS and PAE matched", flush=True)
    write(OUT / "scores.json", {"status": protocol["status"], "rows": scored, "source_sha256": artifacts})
    summaries = {}
    for pool in ("candidate", "all_with_controls"):
        components = {}
        for scaffold in sorted({r["scaffold"] for r in scored}):
            group = [r for r in scored if r["scaffold"] == scaffold and (pool != "candidate" or r["entity_type"] == "candidate")]
            target = [r["target"] for r in group]
            methods = {k: [r[k] for r in group] for k in ("pae", "ridge", "mpnn_nll")}
            methods["ecls"] = [-r["ecls"] for r in group]
            methods["joint"] = joint_scores([r["ecls"] for r in group], methods["pae"])
            components[scaffold] = {"n": len(group), "recall": {str(b): {k: screening(v, target, b) for k, v in methods.items()} for b in protocol["budgets"]}}
        means = {str(b): {k: float(np.mean([c["recall"][str(b)][k] for c in components.values()])) for k in methods} for b in protocol["budgets"]}
        paired = {}
        rng = np.random.default_rng(protocol["bootstrap_seed"])
        for baseline in ("ecls", "pae", "ridge", "mpnn_nll"):
            delta = np.array([c["recall"]["0.2"]["joint"] - c["recall"]["0.2"][baseline] for c in components.values()])
            boot = rng.choice(delta, (protocol["bootstrap_replicates"], len(delta)), replace=True).mean(axis=1)
            paired[baseline] = {"mean_delta": float(delta.mean()), "descriptive_95_ci": np.quantile(boot, [.025, .975]).tolist()}
        summaries[pool] = {"components": components, "macro_mean_recall": means, "joint_minus_baseline_at_20_percent": paired}
    write(OUT / "report.json", {"status": protocol["status"], "independent_confirmation": False, "endpoint": "AF2 teacher top20 recall, not binding", "pools": summaries})
    print(json.dumps(summaries["candidate"]["macro_mean_recall"], indent=2))


if __name__ == "__main__":
    main()
