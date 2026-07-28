import pytest

from modules.idp_antibody_design import graft_cdrs, validate_cdr_candidate


def test_graft_cdrs_requires_exact_payload_and_preserves_framework():
    grafted, mutations = graft_cdrs("ABCDEFGHIJ", "XYUV", [(2, 3, 2), (7, 8, 2)])
    assert grafted == "AXYDEFUVIJ"
    assert [position for position, _, _ in mutations] == [2, 3, 7, 8]
    assert grafted[0] == "A" and grafted[3:6] == "DEF" and grafted[8:] == "IJ"

    with pytest.raises(ValueError, match="payload"):
        graft_cdrs("ABCDEFGHIJ", "XYZ", [(2, 3, 2), (7, 8, 2)])
    with pytest.raises(ValueError, match="exceeds scaffold"):
        graft_cdrs("ABCDE", "XY", [(5, 6, 2)])


def test_validate_cdr_candidate_enforces_budgets_identity_and_hotspots():
    native = {"H1": "ACDEF", "H3": "GHIKL"}
    candidate = {"H1": "ACKEF", "H3": "GHIML"}
    result = validate_cdr_candidate(
        native, candidate,
        max_total_mutations=2,
        max_mutations_per_cdr={"H1": 1, "H3": 1},
        min_native_identity=0.8,
        fixed_hotspots={"H1": [1, 2]},
    )
    assert result["total_mutations"] == 2
    assert result["native_identity"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="fixed hotspot"):
        validate_cdr_candidate(
            native, {**candidate, "H1": "KCKEF"}, max_total_mutations=3,
            fixed_hotspots={"H1": [1]})
    with pytest.raises(ValueError, match="limit"):
        validate_cdr_candidate(native, candidate, max_total_mutations=1)
