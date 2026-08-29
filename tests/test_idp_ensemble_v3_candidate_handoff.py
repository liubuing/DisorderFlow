import json

from scripts.blind_idp_ensemble_v3_candidates import blind
from scripts.finalize_idp_ensemble_v3_candidates import finalize


def test_finalize_and_blind_v3_candidates(tmp_path):
    panel = {
        "components": [{"component_id": f"C{i:02d}"} for i in range(12)],
        "seeds": [12001, 12011, 12021, 12031],
        "mutation_space": {"substitution_buckets": [2, 4, 6, 8]},
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel), encoding="ascii")
    candidate_dir = tmp_path / "components"
    candidate_dir.mkdir()
    native = "AAAAAAAAAA"
    for component in panel["components"]:
        component_id = component["component_id"]
        arms = {}
        for arm in ("ensemble", "single_state"):
            rows = []
            for index, bucket in enumerate([2, 4, 6, 8]):
                sequence = "C" * bucket + "A" * (len(native) - bucket)
                rows.append({
                    "seed": panel["seeds"][index],
                    "substitution_bucket": bucket,
                    "substitutions": bucket,
                    "sequence": sequence,
                })
            arms[arm] = {
                "pose_count": 2 if arm == "ensemble" else 1,
                "candidates": rows,
                "native_control": {"sequence": native, "substitutions": 0},
            }
        (candidate_dir / f"{component_id}.json").write_text(json.dumps({
            "status": "component_candidates_complete",
            "component_id": component_id,
            "target": "target",
            "native_h3": native,
            "seeds": panel["seeds"],
            "substitution_buckets": [2, 4, 6, 8],
            "arms": arms,
        }), encoding="ascii")
    candidates_path = tmp_path / "candidates.json"
    result = finalize(panel_path, candidate_dir, candidates_path)
    assert result["candidate_count"] == 96
    manifest_path = tmp_path / "manifest.json"
    key_path = tmp_path / "key.json"
    manifest = blind(candidates_path, panel_path, manifest_path, key_path)
    assert manifest["construct_count"] == 108
    assert all("sequence" not in row and "arm" not in row for row in manifest["constructs"])
    assert len(json.loads(key_path.read_text(encoding="ascii"))["constructs"]) == 108
