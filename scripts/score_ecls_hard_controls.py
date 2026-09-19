"""Explicit-seed ECLS controls and paired invariance checks on development data."""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ProteinMPNN"))
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize
from scripts.benchmark_ecls_pae_joint import digest, read, write
from scripts.evaluate_mpnn_likelihood_baseline import MPNN_ALPHABET, chain_residues, locate_h3, score_sequence

OUT = ROOT / "results/ecls_hard_controls_scored_v1"


def rigid_transform(x):
    rotation = x.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return x @ rotation.T + x.new_tensor([8, -4, 2])


def ecls_scores(complex_logp, apo_logp, indices, sequences):
    if any(len(s) != len(indices) for s in sequences):
        raise ValueError("H3 length mismatch")
    return np.array([score_sequence(apo_logp, indices, s) - score_sequence(complex_logp, indices, s) for s in sequences])


def make_batch(pdb, hc, ac, indices, apo=False):
    parsed = parse_PDB(str(pdb))[0]
    chains = sorted(k.removeprefix("seq_chain_") for k in parsed if k.startswith("seq_chain_"))
    if apo:
        parsed.pop(f"seq_chain_{ac}")
        parsed.pop(f"coords_chain_{ac}")
        chains.remove(ac)
        parsed["seq"] = "".join(parsed[f"seq_chain_{c}"] for c in chains)
        parsed["num_of_chains"] = len(chains)
    length = len(parsed[f"seq_chain_{hc}"])
    fixed = {parsed["name"]: {hc: [i + 1 for i in range(length) if i not in indices]}}
    roles = {parsed["name"]: ([hc], [c for c in chains if c != hc])}
    batch = tied_featurize([copy.deepcopy(parsed)], torch.device("cpu"), roles, fixed)
    x, s, mask, _, chain_m, chain_encoding = batch[:6]
    chain_m_pos, residue_idx = batch[10], batch[12]
    design = chain_m * chain_m_pos
    expected = parsed[f"seq_chain_{hc}"]
    decoded = "".join(MPNN_ALPHABET[int(i)] for i in s[0, :length])
    if decoded != expected or not torch.all(mask[0, indices] * design[0, indices] > 0):
        raise ValueError("ProteinMPNN chain or H3 mask alignment mismatch")
    order = [hc] + sorted(c for c in chains if c != hc)
    offset = 0
    antigen_indices = []
    for c in order:
        if c == ac:
            antigen_indices = list(range(offset, offset + len(parsed[f"seq_chain_{c}"])))
        offset += len(parsed[f"seq_chain_{c}"])
    return (x, s, mask, design, residue_idx, chain_encoding), antigen_indices


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest_path = ROOT / "results/ecls_pae_evidence_dev_v1/hard_controls_manifest.json"
    manifest = read(manifest_path)
    for path, expected in manifest["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Prepared control source changed: {path}")
    weights = ROOT / "ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    protocol = {"status": "development_only", "manifest_sha256": digest(manifest_path),
        "weights_sha256": digest(weights), "seeds": manifest["protocol"]["scoring_seeds"],
        "seed_semantics": "literal torch numpy python seeds including zero; bypass CLI random-zero behavior",
        "invariance_atol": 1e-5, "device": "cpu", "backbone_noise": 0,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in (
            Path(__file__).resolve(), ROOT / "ProteinMPNN/protein_mpnn_utils.py",
            ROOT / "scripts/evaluate_mpnn_likelihood_baseline.py")},
        "rigid_transform": "90_degree_z_rotation_plus_translation_8_minus4_2_in_memory",
        "bootstrap": "10000_component_resamples_seed_20260916_descriptive_only",
        "independent_confirmation": False}
    if (OUT / "protocol.json").exists() and read(OUT / "protocol.json") != protocol:
        raise ValueError("Protocol/source changed; use a new run version")
    write(OUT / "protocol.json", protocol)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, augment_eps=0, k_neighbors=checkpoint["num_edges"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    components = []
    for component in manifest["components"]:
        scaffold = component["scaffold"]
        pdb = ROOT / component["pdb"]
        hc, ac = component["heavy_chain"], component["antigen_chain"]
        indices = locate_h3(chain_residues(pdb, hc), component["native_h3"])
        sequences = [component["native_h3"]] + [c["h3_sequence"] for c in component["controls"]]
        if len(sequences) != 21 or len(set(sequences)) != 21:
            raise ValueError("Expected 20 unique nonnative controls")
        batches = [make_batch(pdb, hc, ac, indices, apo=a) for a in (False, True)]
        runs = []
        for seed in protocol["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            state_arrays, transformed_arrays = [], []
            shuffled_complex = None
            invariance = {}
            for state, (batch, antigen) in zip(("complex", "apo"), batches):
                x, s, mask, design, residue, chains = batch
                rand = torch.randn(design.shape)
                def forward(xx, ss):
                    with torch.no_grad():
                        result = model.conditional_probs(xx, ss, mask, design, residue, chains, rand, backbone_only=True)[0].numpy()
                    if not np.isfinite(result[indices]).all():
                        raise ValueError("Nonfinite H3 log probabilities")
                    return result
                base = forward(x, s)
                transformed = forward(rigid_transform(x), s)
                state_arrays.append(base)
                transformed_arrays.append(transformed)
                invariance[f"{state}_rigid_max_abs_logp"] = float(np.max(np.abs(base[indices] - transformed[indices])))
                arrays = {"baseline_logp": base, "rigid_logp": transformed}
                if antigen:
                    shuffled = s.clone()
                    original = s[0, antigen]
                    permutation = torch.randperm(len(antigen))
                    if torch.equal(original[permutation], original):
                        permutation = torch.roll(torch.arange(len(antigen)), 1)
                    if torch.equal(original[permutation], original):
                        raise ValueError("Antigen shuffle did not change identity")
                    shuffled[0, antigen] = original[permutation]
                    shuffled_complex = forward(x, shuffled)
                    invariance["antigen_shuffle_max_abs_logp"] = float(np.max(np.abs(base[indices] - shuffled_complex[indices])))
                    arrays["antigen_shuffle_logp"] = shuffled_complex
                    arrays["shuffled_antigen_S"] = shuffled[0, antigen].numpy()
                path = OUT / "arrays" / f"{scaffold}_{seed}_{state}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays, h3_indices=np.array(indices), S=s[0].numpy())
            values = ecls_scores(*state_arrays, indices, sequences)
            rigid_values = ecls_scores(*transformed_arrays, indices, sequences)
            shuffle_values = ecls_scores(shuffled_complex, state_arrays[1], indices, sequences)
            invariance["rigid_max_abs_ecls"] = float(np.max(np.abs(values - rigid_values)))
            invariance["antigen_shuffle_max_abs_ecls"] = float(np.max(np.abs(values - shuffle_values)))
            passed = all(v <= protocol["invariance_atol"] for v in invariance.values())
            legacy = []
            for state, base in zip(("complex", "apo"), state_arrays):
                suffix = scaffold if state == "complex" else scaffold + "_apo"
                path = ROOT / f"results/ecls_pae_joint_dev_v1/work/{scaffold}/{state}/conditional_probs_only/{suffix}.npz"
                with np.load(path) as z:
                    legacy.append({"state": state, "source": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                        "max_abs_H3_logp_difference": float(np.max(np.abs(base[indices] - z["log_p"][0, indices])))})
            runs.append({"seed": seed, "ecls": values.tolist(), "native_minus_control_mean": float(values[0] - values[1:].mean()),
                "native_beats_control_fraction": float(np.mean(values[0] > values[1:]) + .5 * np.mean(values[0] == values[1:])),
                "invariance": invariance, "invariance_pass": passed, "legacy_cache_comparison": legacy})
            print(f"{scaffold}, seed {seed}: delta={runs[-1]['native_minus_control_mean']:.6f}, invariance={passed}", flush=True)
        components.append({"scaffold": scaffold, "sequences": sequences, "runs": runs,
            "mean_native_advantage": float(np.mean([r["native_minus_control_mean"] for r in runs])),
            "advantage_seed_range": float(np.ptp([r["native_minus_control_mean"] for r in runs]))})
        write(OUT / "progress.json", {"components_completed": components})
    deltas = np.array([c["mean_native_advantage"] for c in components])
    boot = np.random.default_rng(20260916).choice(deltas, (10000, len(deltas)), replace=True).mean(1)
    arrays_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((OUT / "arrays").glob("*.npz"))}
    report = {"status": "completed_development_only", "protocol": protocol, "components": components,
        "component_count": len(components), "unique_control_count": 120,
        "mean_native_advantage": float(deltas.mean()), "positive_components": int((deltas > 0).sum()),
        "descriptive_component_bootstrap_95_interval": np.quantile(boot, [.025, .975]).tolist(),
        "all_invariance_checks_pass": all(r["invariance_pass"] for c in components for r in c["runs"]),
        "arrays_sha256": arrays_hashes,
        "limitations": ["Previously inspected six development components, no independent confirmation.",
            "Near-native controls have unknown binding status; no affinity claim.",
            "Three scoring seeds are sensitivity checks, not independent biological replicates.",
            "Legacy CLI seed zero selected an unrecorded random seed; cached outputs preserved, comparisons reported.",
            "Structural contact/remote perturbation experiment not run."]}
    write(OUT / "report.json", report)
    print(f"Mean advantage {deltas.mean():.6f}; positive {sum(deltas > 0)}/6; all invariance {report['all_invariance_checks_pass']}")


if __name__ == "__main__":
    main()
