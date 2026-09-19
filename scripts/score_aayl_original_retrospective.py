"""Descriptive ECLS evaluation on source-verified AlphaSeq H3 variants."""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_ecls_hard_controls import make_batch, ProteinMPNN
from scripts.evaluate_mpnn_likelihood_baseline import score_sequence
from scripts.build_aayl_original_endpoint_cohort import freeze

SOURCE = ROOT / "data/aayl_original_retrospective_v1"
OUT = ROOT / "results/aayl_original_retrospective_v1"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def correlation(x, y):
    if len(set(map(float, x))) < 2 or len(set(map(float, y))) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def main():
    cohort = json.loads((SOURCE / "cohort.json").read_text())
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    if not cohort["ready_for_descriptive_scoring"] or cohort["protocol_sha256"] != digest(SOURCE / "protocol.json"):
        raise ValueError("Cohort does not pass its prepared scoring gate")
    weights = ROOT / "ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    freeze(OUT / "protocol.json", {"cohort_sha256": digest(SOURCE / "cohort.json"), "rules_sha256": digest(SOURCE / "protocol.json"),
        "weights_sha256": digest(weights), "scorer_sha256": digest(Path(__file__)),
        "batch_helper_sha256": digest(ROOT / "scripts/score_ecls_hard_controls.py"),
        "vendor_sha256": digest(ROOT / "ProteinMPNN/protein_mpnn_utils.py"),
        "seeds": protocol["seeds"], "torch_version": str(torch.__version__), "backbone_noise": 0,
        "device": "cpu", "classification": "retrospective_single_antigen_descriptive_not_confirmatory"})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, augment_eps=0, k_neighbors=checkpoint["num_edges"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    summaries, scores = [], []
    for family, structure in cohort["structures"].items():
        selected = [r for r in cohort["eligible"] if r["family"] == family]
        pdb = ROOT / "data/abbibench_structural_audit_v1/structures" / Path(structure["file"]).name
        if digest(pdb) != structure["sha256"]:
            raise ValueError("Structure changed")
        indices = structure["h3_indices_zero_based_observed_heavy"]
        batches = [make_batch(pdb, structure["heavy_chain"], structure["antigen_chains"][0], indices, apo=a)[0] for a in (False, True)]
        values_by_seed = []
        for seed in protocol["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            arrays = []
            for batch in batches:
                x, s, mask, design, residue, chains = batch
                with torch.no_grad():
                    logp = model.conditional_probs(x, s, mask, design, residue, chains, torch.randn(design.shape), backbone_only=True)[0].numpy()
                if not np.isfinite(logp[indices]).all():
                    raise ValueError("Nonfinite log probabilities")
                arrays.append(logp)
            values = []
            for row in selected:
                complex_nll, apo_nll = [score_sequence(logp, indices, row["h3_sequence"]) for logp in arrays]
                values.append([apo_nll-complex_nll, -complex_nll, -apo_nll])
            values_by_seed.append(np.array(values))
            (OUT / "arrays").mkdir(exist_ok=True)
            np.savez_compressed(OUT / "arrays" / f"{family}_{seed}.npz", complex_logp=arrays[0], apo_logp=arrays[1], h3_indices=indices)
        matrix = np.mean(values_by_seed, axis=0)
        y = [r["endpoint"] for r in selected]
        methods = ["ECLS", "negative_complex_NLL", "negative_apo_NLL"]
        correlations = {name: correlation(matrix[:, i], y) for i, name in enumerate(methods)}
        summary = {"family": family, "n_unique_variants": len(selected), "spearman": correlations,
            "ECLS_minus_complex_spearman": correlations["ECLS"]-correlations["negative_complex_NLL"] if correlations["ECLS"] is not None and correlations["negative_complex_NLL"] is not None else None,
            "maximum_seed_score_difference": float(np.max(np.abs(np.stack(values_by_seed)-values_by_seed[0]))),
            "per_seed_spearman": [{name: correlation(v[:, i], y) for i, name in enumerate(methods)} for v in values_by_seed]}
        summaries.append(summary)
        for row, vals in zip(selected, matrix):
            scores.append({"family": family, "poi": row["poi"], "assay": row["assay"], "h3_sequence": row["h3_sequence"],
                "endpoint": row["endpoint"], **{name: float(v) for name, v in zip(methods, vals)}})
        print(json.dumps(summary, indent=2), flush=True)
    freeze(OUT / "report.json", {"classification": "retrospective_AlphaSeq_single_antigen_descriptive", "summary": summaries,
        "independent_antigens": 1, "structure_source": "AF3 predicted parent complexes", "source_endpoint": "9 - median(original log10 nM)",
        "independent_confirmation": False, "p_values_or_antigen_level_CI": None,
        "limitations": ["One antigen and two parent lineages", "Complete-readings subset excludes weak/no-reading observations", "Predicted rather than experimental complex coordinates", "Known historical sequence overlaps and prior label processing audit", "No direct physical Kd calibration or causal binding claim"]})
    freeze(OUT / "scores.json", {"records": scores})


if __name__ == "__main__":
    main()
