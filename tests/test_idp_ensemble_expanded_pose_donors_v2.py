from scripts.materialize_idp_ensemble_expanded_pose_donors_v2 import donor_ids


def test_donor_ids_only_include_experimental_ready_components():
    audit = {
        "components": [
            {
                "experimental_pose_source_ready": True,
                "same_lineage_target_pdb_ids": ["2bbb", "1aaa"],
            },
            {
                "experimental_pose_source_ready": False,
                "same_lineage_target_pdb_ids": ["3ccc"],
            },
        ]
    }
    assert donor_ids(audit) == ["1aaa", "2bbb"]
