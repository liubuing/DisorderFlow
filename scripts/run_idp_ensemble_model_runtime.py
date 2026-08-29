#!/usr/bin/env python
"""Run a real two-pose ProteinMPNN prefix-conditioned runtime smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTEIN_MPNN = ROOT / "ProteinMPNN"
if str(PROTEIN_MPNN) not in sys.path:
    sys.path.insert(0, str(PROTEIN_MPNN))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protein_mpnn_utils import (  # noqa: E402
    StructureDatasetPDB,
    _S_to_seq,
    parse_PDB,
    tied_featurize,
)

from scripts.autoregressive_multiconformer_sampler import sample_autoregressive  # noqa: E402
from scripts.protein_mpnn_prefix_adapter import (  # noqa: E402
    ProteinMPNNPrefixAdapter,
    build_model,
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_pose_adapter(pdb_path, model, component, device):
    parsed = parse_PDB(str(pdb_path))
    dataset = StructureDatasetPDB(parsed, truncate=None, max_length=1000, verbose=False)
    item = dataset[0]
    name = item["name"]
    chain_dict = {name: ([component["heavy_chain"]], component["visible_chains"])}
    tensors = tied_featurize(
        [item], device, chain_dict, ca_only=False
    )
    X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, _, _, residue_idx = (
        tensors[:13]
    )
    sequence = item[f"seq_chain_{component['heavy_chain']}"]
    native_h3 = component["native_h3"]
    start = sequence.find(native_h3)
    if start < 0:
        raise ValueError(f"Native H3 not found in heavy chain for {pdb_path}")
    # tied_featurize orders masked/design chains before visible chains.
    prefix_length = 0
    h3_positions = [prefix_length + start + offset for offset in range(len(native_h3))]
    native_indices = S[0, h3_positions].detach().cpu().numpy()
    return ProteinMPNNPrefixAdapter(
        model,
        {
            "X": X,
            "S": S,
            "mask": mask,
            "chain_M": chain_M,
            "residue_idx": residue_idx,
            "chain_encoding_all": chain_encoding_all,
        },
        h3_positions,
        native_indices,
        pose_id=pdb_path.stem,
    ), _S_to_seq(S[0], mask[0]), h3_positions


def run(pose_paths, checkpoint, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite runtime trace: {output}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device=device)
    component = {
        "heavy_chain": "B",
        "visible_chains": ["A", "P"],
        "native_h3": "ASLYSLPVY",
    }
    adapters = []
    native = None
    positions = None
    for path in pose_paths:
        adapter, full_native, h3_positions = build_pose_adapter(
            path, model, component, device
        )
        if native is None:
            native = full_native
            positions = h3_positions
        elif full_native != native or h3_positions != positions:
            raise ValueError("Pose tensors do not share sequence or H3 index layout")
        adapters.append(adapter)
    sample_autoregressive(
        component["native_h3"], adapters, np.random.default_rng(12001),
        max_substitutions=0, alphabet="ACDEFGHIKLMNPQRSTVWYX",
    )
    events = []
    for adapter in adapters:
        events.extend(adapter.runtime_trace)
    payload = {
        "schema_version": 1,
        "status": "runtime_trace_complete",
        "classification": "historical_pose_model_runtime_validation_only",
        "checkpoint": str(checkpoint),
        "pose_paths": [str(path) for path in pose_paths],
        "component_id": "ABETA_3U0T",
        "h3_positions_zero_indexed": positions,
        "events": events,
        "claim_boundary": "runtime validation only; not untouched confirmation or binding evidence",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(
        [resolve(path) for path in args.pose],
        resolve(args.checkpoint),
        resolve(args.output),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "events"}, indent=2))


if __name__ == "__main__":
    main()
