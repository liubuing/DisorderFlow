import numpy as np

from scripts.score_idp_ensemble_v3_source_sensitivity import panel_rows


def test_source_panels_split_experimental_and_sampled():
    rows = [
        {"pose_id": "experimental", "source": "own_entry"},
        {
            "pose_id": "sampled",
            "source": "restrained_local_interface_sampling",
        },
    ]
    panels = panel_rows(rows)
    assert [row["pose_id"] for row in panels["experimental_only"]] == [
        "experimental"
    ]
    assert [row["pose_id"] for row in panels["sampled_only"]] == ["sampled"]
    assert len(panels["mixed"]) == 2
