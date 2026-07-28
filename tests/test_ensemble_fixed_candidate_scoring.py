import sys
from pathlib import Path

import numpy as np
import torch

from modules.ensemble_scorer import EnsembleScorer


def test_ensemble_scorer_scores_each_exact_candidate_on_every_conformer(
        monkeypatch, tmp_path):
    conformers = [np.zeros((2, 3)), np.ones((2, 3))]
    paths = [tmp_path / "c0.pdb", tmp_path / "c1.pdb"]
    for path in paths:
        path.write_text("END\n", encoding="ascii")

    scorer = object.__new__(EnsembleScorer)
    monkeypatch.setattr(
        scorer, "find_conformations",
        lambda sequence: {"ca_positions": conformers, "rmsf": np.ones(2)},
    )
    monkeypatch.setattr(
        scorer, "build_conformation_pdbs", lambda positions, chain: (paths, str(tmp_path)))
    monkeypatch.setattr("modules.ensemble_scorer.shutil.rmtree", lambda *args, **kwargs: None)

    calls = []

    def score_candidate(pdb_path, region_spec, candidate_sequence, **kwargs):
        calls.append((Path(pdb_path).name, candidate_sequence))
        value = float(len(candidate_sequence) + int(Path(pdb_path).stem[-1]))
        return {
            "plddt": torch.tensor([[value]]),
            "iptm": torch.tensor([value]),
            "pae": torch.tensor([[[value]]]),
            "state_compatibility": torch.tensor([value]),
        }

    import modules.bfn_loader as bfn_loader
    monkeypatch.setattr(bfn_loader, "score_bfn_candidate", score_candidate)

    result = scorer.score_designs_bfn(
        [{"sequence": "AAAA"}, {"sequence": "CCCCC"}], "AG", object(), device="cpu")

    assert calls == [
        ("c0.pdb", "AAAA"), ("c1.pdb", "AAAA"),
        ("c0.pdb", "CCCCC"), ("c1.pdb", "CCCCC"),
    ]
    assert result[0]["bfn_fixed_candidate_across_conformers"] is True
    assert result[0]["bfn_state_compatibility_mean"] == 4.5
    assert result[1]["bfn_state_compatibility_mean"] == 5.5
