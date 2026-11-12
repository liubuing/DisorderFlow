"""End-to-end smoke test on a real SAbDab-derived complex (4HIX, chains H/L/A).

Requires torch in the running interpreter and the vendored ProteinMPNN script,
weights and demo PDB; skipped otherwise so the suite stays runnable anywhere.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from ecls_scorer import cli

REPO_ROOT = Path(__file__).resolve().parents[3]
MPNN_SCRIPT = REPO_ROOT / "ProteinMPNN" / "protein_mpnn_run.py"
MPNN_WEIGHTS = REPO_ROOT / "ProteinMPNN" / "vanilla_model_weights"
DEMO_PDB = REPO_ROOT / "data" / "anti_abeta_refs" / "4HIX.pdb"
H3_SEQUENCE = "VRYDHYSGSSDY"  # 4HIX native CDR-H3

torch_available = importlib.util.find_spec("torch") is not None
assets_available = MPNN_SCRIPT.exists() and MPNN_WEIGHTS.exists() and DEMO_PDB.exists()

pytestmark = pytest.mark.skipif(
    not (torch_available and assets_available),
    reason="requires torch and the vendored ProteinMPNN assets plus the 4HIX demo PDB")


def test_cli_scores_real_complex_and_reports_advantage(tmp_path):
    output = tmp_path / "result.json"
    exit_code = cli.main([
        "--pdb", str(DEMO_PDB),
        "--heavy-chain", "H", "--light-chain", "L", "--peptide-chain", "A",
        "--h3-sequence", H3_SEQUENCE,
        "--shuffles", "20", "--seed", "2309",
        "--mpnn-script", str(MPNN_SCRIPT),
        "--mpnn-weights", str(MPNN_WEIGHTS),
        "--work-dir", str(tmp_path / "work"),
        "--output", str(output),
    ])
    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["h3"]["sequence"] == H3_SEQUENCE
    indices = result["h3"]["indices_0based"]
    assert len(indices) == len(H3_SEQUENCE)
    assert indices == list(range(indices[0], indices[-1] + 1))

    native = result["sequences"][0]
    assert native["name"] == "native"
    assert native["sequence"] == H3_SEQUENCE
    assert native["ecls"] == round(native["complex_h3_nll"] - native["apo_h3_nll"], 6)

    shuffle_rows = [row for row in result["sequences"] if row["name"].startswith("shuffle_")]
    assert len(shuffle_rows) == 20
    for row in shuffle_rows:
        assert sorted(row["sequence"]) == sorted(H3_SEQUENCE)

    summary = result["advantage_summary"]
    assert summary["shuffle_count"] == 20
    assert summary["native"]["name"] == "native"
    assert result["provenance"]["model_weights_sha256"]
    assert "does not reproduce" in result["boundary"]
