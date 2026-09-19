"""Post-hoc double-precision diagnosis; does not revise original pass/fail."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_ecls_hard_controls import (
    OUT, ProteinMPNN, chain_residues, digest, ecls_scores, locate_h3,
    make_batch, read, rigid_transform, write,
)


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report_path = OUT / "report.json"
    report = read(report_path)
    manifest_path = ROOT / "results/ecls_pae_evidence_dev_v1/hard_controls_manifest.json"
    manifest = read(manifest_path)
    weights = ROOT / "ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    # Upstream hard-codes float32 casts. Use an in-memory diagnostic copy;
    # never edit the scoring implementation or its saved single-precision result.
    vendor_path = ROOT / "ProteinMPNN/protein_mpnn_utils.py"
    source = vendor_path.read_text(encoding="utf-8").replace(".float()", ".to(dtype=torch.float64)").replace("torch.float32", "torch.float64")
    namespace = {"__name__": "protein_mpnn_float64_diagnostic", "__file__": str(vendor_path)}
    exec(compile(source, str(vendor_path) + "[float64-diagnostic]", "exec"), namespace)
    torch.set_default_dtype(torch.float64)
    model = namespace["ProteinMPNN"](num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, augment_eps=0, k_neighbors=checkpoint["num_edges"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.double().eval()
    results = []
    for c in manifest["components"]:
        sc = c["scaffold"]
        prior = next(r for r in report["components"] if r["scaffold"] == sc)
        if all(r["invariance_pass"] for r in prior["runs"]):
            continue
        indices = locate_h3(chain_residues(ROOT / c["pdb"], c["heavy_chain"]), c["native_h3"])
        sequences = [c["native_h3"]] + [r["h3_sequence"] for r in c["controls"]]
        torch.manual_seed(0)
        arrays, rigid_arrays, maxima = [], [], {}
        for state, apo in (("complex", False), ("apo", True)):
            batch, _ = make_batch(ROOT / c["pdb"], c["heavy_chain"], c["antigen_chain"], indices, apo)
            x, s, mask, design, residue, chains = [v.double() if v.is_floating_point() else v for v in batch]
            rand = torch.randn(design.shape, dtype=torch.float64)
            def forward(coords):
                with torch.no_grad():
                    return model.conditional_probs(coords, s, mask, design, residue, chains, rand, backbone_only=True)[0].numpy()
            baseline, rigid = forward(x), forward(rigid_transform(x))
            arrays.append(baseline)
            rigid_arrays.append(rigid)
            maxima[state] = float(np.max(np.abs(baseline[indices] - rigid[indices])))
        ecls = ecls_scores(*arrays, indices, sequences)
        transformed = ecls_scores(*rigid_arrays, indices, sequences)
        results.append({"scaffold": sc, "float64_rigid_max_abs_H3_logp": maxima,
            "float64_rigid_max_abs_ecls": float(np.max(np.abs(ecls - transformed))),
            "float64_native_advantage": float(ecls[0] - ecls[1:].mean()),
            "difference_from_float32_native_advantage": float(ecls[0] - ecls[1:].mean() - prior["mean_native_advantage"])})
        print(results[-1], flush=True)
    write(OUT / "numeric_diagnosis.json", {"status": "posthoc_numeric_diagnosis_original_failures_retained",
        "diagnostic_precision_change": "in-memory vendor copy: float() and torch.float32 replaced by float64; default dtype float64; checkpoint values unchanged",
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in (report_path, manifest_path, weights, vendor_path, Path(__file__).resolve())},
        "components": results})


if __name__ == "__main__":
    main()
