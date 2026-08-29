#!/usr/bin/env python
"""Generate v3 untouched-panel H3 candidates with the prefix-conditioned model.

Each component uses one aggregated adapter per arm: the ensemble arm batches
every accepted pose in a single forward per H3 position (equal-weight
log-probability averaging, per the frozen correction contract); the
single-state arm uses only the component's own experimental pose. Candidates
fill substitution buckets [2, 4, 6, 8] exactly, one candidate per bucket per
arm, using the frozen per-bucket seeds, with deduplication inside each
component-arm and a native control recorded per arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTEIN_MPNN = ROOT / "ProteinMPNN"
for entry in (str(PROTEIN_MPNN), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from protein_mpnn_utils import StructureDatasetPDB, parse_PDB, tied_featurize  # noqa: E402

from scripts.autoregressive_multiconformer_sampler import sample_autoregressive  # noqa: E402
from scripts.protein_mpnn_prefix_adapter import build_model  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_ATTEMPTS = 200


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BatchedPoseAdapter:
    """Expose one autoregressive interface over a homogeneous pose batch.

    At each H3 position the model sees the already sampled prefix. Log
    probabilities are averaged across poses with equal frozen weights. This is
    a product-of-experts-like sequence objective, not an average of completed
    candidate scores after generation.
    """

    def __init__(self, model, tensors, h3_positions, native_indices, arm_name):
        self.model = model
        self.tensors = tensors
        self.h3_positions = tuple(int(value) for value in h3_positions)
        self.native_indices = np.asarray(native_indices, dtype=np.int64)
        self.arm_name = arm_name
        self.pose_count = int(tensors["X"].shape[0])
        self.call_count = 0

    @torch.no_grad()
    def next_log_probs(self, prefix, position):
        if len(prefix) != position:
            raise ValueError("Prefix length must equal the requested H3 position")
        # Every pose starts from its native sequence tensor. Only the sampled
        # prefix is replaced; future H3 residues remain unavailable to the
        # prefix-conditioned model.
        sequence = self.tensors["S"].clone()
        if prefix:
            letters = torch.as_tensor(
                [ALPHABET.index(aa) for aa in prefix], device=sequence.device
            )
            sequence[:, list(self.h3_positions[:position])] = letters.unsqueeze(0)
        # tied_featurize marks the full heavy chain as designable. The actual
        # frozen contract mutates H3 only, so expose framework sequence as fixed
        # context and keep only future H3 positions hidden by decoding order.
        design_mask = torch.zeros_like(self.tensors["chain_M"])
        design_mask[:, list(self.h3_positions)] = 1.0
        log_probs = self.model.prefix_next_log_probs(
            self.tensors["X"], sequence, self.tensors["mask"],
            design_mask, self.tensors["residue_idx"],
            self.tensors["chain_encoding_all"], self.h3_positions[:position],
            self.h3_positions[position],
        )
        self.call_count += 1
        return log_probs.mean(dim=0).detach().cpu().numpy()

    def native_sequence_score(self, native):
        score = 0.0
        for position, aa in enumerate(native):
            values = self.next_log_probs(native[:position], position)
            score += float(values[ALPHABET.index(aa)])
        return score


class HeterogeneousPoseAdapter:
    """Aggregate poses that cannot share one padded structural tensor.

    Missing residues or different chain layouts make joint featurization
    unsafe. Scoring each compatible layout separately preserves pose evidence
    instead of dropping the inconvenient structures after seeing results.
    """

    def __init__(self, adapters, arm_name):
        if not adapters:
            raise ValueError("Heterogeneous aggregation requires pose adapters")
        self.adapters = adapters
        self.arm_name = arm_name
        self.pose_count = sum(adapter.pose_count for adapter in adapters)
        self.call_count = 0

    def next_log_probs(self, prefix, position):
        values = np.stack([
            adapter.next_log_probs(prefix, position) for adapter in self.adapters
        ])
        self.call_count += len(self.adapters)
        return values.mean(axis=0)

    def native_sequence_score(self, native):
        score = 0.0
        for position, amino_acid in enumerate(native):
            values = self.next_log_probs(native[:position], position)
            score += float(values[ALPHABET.index(amino_acid)])
        return score


def canonicalize_chains(item, antibody_chains, antigen_chains):
    """Relabel chain IDs deterministically for ProteinMPNN featurization."""
    ordered = list(antibody_chains) + list(antigen_chains)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Antibody and antigen chain roles overlap")
    if len(ordered) > len(CHAIN_ALPHABET):
        raise ValueError("Too many chains for canonical relabeling")
    mapping = dict(zip(ordered, CHAIN_ALPHABET, strict=False))
    canonical = {
        key: value
        for key, value in item.items()
        if not key.startswith(("seq_chain_", "coords_chain_"))
    }
    for old, new in mapping.items():
        sequence_key = f"seq_chain_{old}"
        coordinates_key = f"coords_chain_{old}"
        if sequence_key not in item or coordinates_key not in item:
            raise ValueError(f"Configured chain is absent from pose: {old}")
        canonical[f"seq_chain_{new}"] = item[sequence_key]
        canonical[f"coords_chain_{new}"] = {
            coordinate_key.replace(f"chain_{old}", f"chain_{new}"): values
            for coordinate_key, values in item[coordinates_key].items()
        }
    canonical["seq"] = "".join(
        canonical[f"seq_chain_{mapping[chain]}"] for chain in ordered
    )
    return canonical, mapping


def find_observed_subsequence_positions(sequence, query):
    """Map an ungapped H3 sequence back to indices in a gapped parsed chain."""
    observed = [(index, amino_acid) for index, amino_acid in enumerate(sequence)
                if amino_acid != "-"]
    observed_sequence = "".join(amino_acid for _, amino_acid in observed)
    start = observed_sequence.find(query)
    if start < 0:
        return []
    return [
        observed[index][0] for index in range(start, start + len(query))
    ]


def featurize_arm(model, pose_rows, arm_name, native_h3, device):
    """Create a pose adapter and auditable H3/chain provenance for one arm."""
    items = []
    chain_dict = {}
    h3_mappings = {}
    layout_signatures = {}
    for row in pose_rows:
        parsed = parse_PDB(str(ROOT / row["path"]))
        dataset = StructureDatasetPDB(parsed, truncate=None, max_length=5000, verbose=False)
        if len(dataset) != 1:
            raise ValueError(f"{row['pose_id']}: pose could not be featurized")
        item = dataset[0]
        item, mapping = canonicalize_chains(
            item, row["antibody_chains"], row["antigen_chains"]
        )
        heavy = mapping[row["antibody_chains"][0]]
        sequence = item[f"seq_chain_{heavy}"]
        h3_positions = find_observed_subsequence_positions(sequence, native_h3)
        if not h3_positions:
            raise ValueError(f"{row['pose_id']}: native H3 not found in heavy chain")
        visible = [mapping[chain] for chain in (
            list(row["antibody_chains"])[1:] + list(row["antigen_chains"])
        )]
        items.append(item)
        chain_dict[item["name"]] = ([heavy], visible)
        h3_mappings[item["name"]] = (tuple(h3_positions), len(sequence))
        layout_signatures[item["name"]] = tuple(sorted(
            (key, len(value)) for key, value in item.items()
            if key.startswith("seq_chain_")
        ))
    batch_signatures = {
        (h3_mappings[name], layout_signatures[name]) for name in h3_mappings
    }
    # Batch only truly identical layouts. Padding unlike chain layouts changes
    # index semantics and can silently score the wrong H3 positions.
    if len(batch_signatures) == 1:
        (h3_positions, heavy_length), _layout = batch_signatures.pop()
        batch = tied_featurize(items, device, chain_dict, ca_only=False)
        X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, _, _, residue_idx = batch[:13]
        h3_positions = list(h3_positions)
        native_indices = S[0, h3_positions].detach().cpu().numpy()
        tensors = {
            "X": X, "S": S, "mask": mask, "chain_M": chain_M,
            "residue_idx": residue_idx, "chain_encoding_all": chain_encoding_all,
        }
        adapter = BatchedPoseAdapter(
            model, tensors, h3_positions, native_indices, arm_name
        )
        h3_provenance = h3_positions
    else:
        adapters = []
        heavy_lengths = []
        for item in items:
            positions, heavy_length = h3_mappings[item["name"]]
            batch = tied_featurize(
                [item], device, {item["name"]: chain_dict[item["name"]]},
                ca_only=False,
            )
            X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, _, _, residue_idx = batch[:13]
            positions = list(positions)
            tensors = {
                "X": X, "S": S, "mask": mask, "chain_M": chain_M,
                "residue_idx": residue_idx,
                "chain_encoding_all": chain_encoding_all,
            }
            adapters.append(BatchedPoseAdapter(
                model, tensors, positions,
                S[0, positions].detach().cpu().numpy(), arm_name,
            ))
            heavy_lengths.append(heavy_length)
        adapter = HeterogeneousPoseAdapter(adapters, arm_name)
        heavy_length = heavy_lengths
        h3_provenance = {
            name: list(mapping[0]) for name, mapping in h3_mappings.items()
        }
    return adapter, {
        "pose_ids": [row["pose_id"] for row in pose_rows],
        "pose_paths": [row["path"] for row in pose_rows],
        "h3_positions_zero_based": h3_provenance,
        "heavy_chain_length": heavy_length,
    }


def count_substitutions(candidate, native):
    if len(candidate) != len(native):
        raise ValueError("Candidate and native H3 lengths differ")
    return sum(left != right for left, right in zip(candidate, native, strict=True))


def generate_component(component, seeds, buckets, native_h3, pose_rows, model, device):
    """Generate matched ensemble and single-state candidates for one component."""
    arms = {}
    # The two arms differ only in structural context. Checkpoint, seed,
    # mutation buckets, and candidate budget remain matched by construction.
    arm_specs = {
        "ensemble": pose_rows,
        "single_state": pose_rows[:1],
    }
    for arm_name, rows in arm_specs.items():
        adapter, provenance = featurize_arm(model, rows, arm_name, native_h3, device)
        candidates = []
        attempts_log = []
        for bucket_index, bucket in enumerate(buckets):
            seed = seeds[bucket_index]
            rng = np.random.default_rng(seed)
            attempts = 0
            while attempts < MAX_ATTEMPTS:
                attempts += 1
                sequence = sample_autoregressive(
                    native_h3, [adapter], rng, max_substitutions=bucket,
                    alphabet=ALPHABET,
                )
                substitutions = count_substitutions(sequence, native_h3)
                # A bucket is exact, not an upper bound. This prevents the
                # ensemble arm from receiving an easier mutation budget.
                if substitutions == bucket and all(
                    row["sequence"] != sequence for row in candidates
                ):
                    candidates.append({
                        "seed": seed,
                        "substitution_bucket": bucket,
                        "substitutions": substitutions,
                        "sequence": sequence,
                    })
                    break
            attempts_log.append({"bucket": bucket, "seed": seed, "attempts": attempts})
        if len(candidates) != len(buckets):
            raise RuntimeError(
                f"{component}/{arm_name}: only {len(candidates)}/{len(buckets)} "
                "bucket candidates generated"
            )
        arms[arm_name] = {
            "pose_count": adapter.pose_count,
            "model_calls": adapter.call_count,
            "provenance": provenance,
            "candidates": candidates,
            "attempts": attempts_log,
            "native_control": {
                "sequence": native_h3,
                "substitutions": 0,
                "generation_score": adapter.native_sequence_score(native_h3),
            },
        }
    return arms


def generate(panel_path, admission_path, readiness_path, poses_path, checkpoint,
             output_dir, components_filter):
    panel = json.loads(panel_path.read_text(encoding="ascii"))
    admission = json.loads(admission_path.read_text(encoding="ascii"))
    readiness = json.loads(readiness_path.read_text(encoding="ascii"))
    poses = json.loads(poses_path.read_text(encoding="ascii"))
    seeds = panel["seeds"]
    buckets = panel["mutation_space"]["substitution_buckets"]
    config = yaml.safe_load(
        (ROOT / "configs/benchmarks/idp_ensemble_correction_v3.yml").read_text(
            encoding="ascii"
        )
    )
    if buckets != config["mutation_space"]["substitution_buckets"]:
        raise RuntimeError("Panel buckets diverge from the frozen contract")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device=device)
    readiness_rows = {row["component_id"]: row for row in readiness["components"]}
    poses_rows = {row["component_id"]: row for row in poses["components"]}
    admission_rows = {row["component_id"]: row for row in admission["components"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    for component in panel["components"]:
        component_id = component["component_id"]
        if components_filter and component_id not in components_filter:
            continue
        output_path = output_dir / f"{component_id}.json"
        if output_path.exists():
            print(f"skip {component_id}: already written")
            continue
        native_h3 = readiness_rows[component_id]["h3_sequence"]
        accepted = poses_rows[component_id]["poses"]
        arms = generate_component(
            component_id, seeds, buckets, native_h3, accepted, model, device
        )
        lineage = admission_rows[component_id]
        payload = {
            "schema_version": 1,
            "status": "component_candidates_complete",
            "component_id": component_id,
            "target": lineage["target"],
            "antibody_lineage": lineage.get(
                "antibody_lineage_cluster",
                lineage.get("lineage", lineage.get("lineage_proxy")),
            ),
            "native_h3": native_h3,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "seeds": seeds,
            "substitution_buckets": buckets,
            "arms": arms,
            "claim_boundary": (
                "computational H3 candidates on untouched non-IDP-specific "
                "targets; no binding or performance result"
            ),
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
        print(f"{component_id}: complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component", action="append", default=[])
    args = parser.parse_args()
    generate(
        args.panel, args.admission, args.readiness, args.poses, args.checkpoint,
        args.output_dir, set(args.component),
    )


if __name__ == "__main__":
    main()
