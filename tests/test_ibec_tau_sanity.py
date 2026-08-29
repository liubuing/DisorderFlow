import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "run_ibec_tau_sanity.py"
    spec = importlib.util.spec_from_file_location("run_ibec_tau_sanity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tau_structure_has_frozen_full_h3_and_epitope():
    mod = load_script()
    path = ROOT / "data/non_abeta_idp/ensembles_v1/tau/5MP3_tau_core_complex.pdb"
    heavy = mod.chain_sequence(path, "A")
    assert heavy[94] == "C"
    assert heavy[95:108] == "ARDYYGTSFAMDY"
    assert heavy[108] == "W"
    assert mod.chain_sequence(path, "C") == "KHVPGGGSV"
    pose = ROOT / "data/non_abeta_idp/pose_panels_v1/tau/conformer0_pose.pdb"
    assert mod.chain_sequence(pose, "P") == "KHVPGGGSV"


def test_selected_tau_poses_are_exactly_five():
    mod = load_script()
    import json

    audit = json.loads((
        ROOT / "outputs/non_abeta_idp_pose_panels_v1/pose_panel_audit.json"
    ).read_text())
    config = {"target": {"selected_poses": 5}}
    rows = mod.selected_tau_poses(config, audit)
    assert len(rows) == 5
    assert [row["conformer"] for row in rows] == [0, 1, 2, 3, 4]
