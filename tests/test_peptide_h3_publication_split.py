from scripts.build.build_peptide_h3_publication_split import (
    cluster_tokens,
    exact_alignment_map,
    exclude_conflicts,
    sequence_homology,
)


def _record(record_id, **axes):
    names = (
        "pdb_id", "official_ab_cluster", "official_cdrh3_cluster",
        "official_antigen_cluster", "vh_sequence_exact", "vl_sequence_exact",
        "cdr_h3_sequence_exact", "antigen_sequence_exact",
    )
    return {
        "id": record_id,
        "axis_values": {name: tuple(axes.get(name, ())) for name in names},
    }


def test_cluster_tokens_normalizes_official_encodings():
    assert cluster_tokens("['ag1', 'ag2']") == ("ag1", "ag2")
    assert cluster_tokens("ag2/ag1") == ("ag1", "ag2")
    assert cluster_tokens("") == ()


def test_later_split_conflict_is_excluded_on_any_axis():
    train = [_record("train", official_ab_cluster=("ab1",), antigen_sequence_exact=("AAAA",))]
    candidates = [
        _record("ab_overlap", official_ab_cluster=("ab1",)),
        _record("antigen_overlap", antigen_sequence_exact=("AAAA",)),
        _record("clean", official_ab_cluster=("ab2",), antigen_sequence_exact=("CCCC",)),
    ]
    retained, excluded = exclude_conflicts(candidates, train)
    assert [record["id"] for record in retained] == ["clean"]
    assert {record["id"] for record in excluded} == {"ab_overlap", "antigen_overlap"}


def test_sequence_homology_detects_covered_fragment_but_not_unrelated_sequence():
    related = sequence_homology("DAEFRHDS", "XXDAEFRHDSYY", 0.8, 0.8)
    unrelated = sequence_homology("DAEFRHDS", "KLVFFAED", 0.5, 0.8)
    assert related["match"] is True
    assert related["coverage"] == 1.0
    assert unrelated["match"] is False


def test_exact_alignment_map_preserves_inserted_official_h3_segment():
    official = "AAAACDYSGWGG"
    parsed = "AACDYSGWGG"
    mapping = exact_alignment_map(official, parsed)
    h3_indices = [mapping[index] for index in range(5, 8)]
    assert "".join(parsed[index] for index in h3_indices) == "DYS"
    assert h3_indices == list(range(h3_indices[0], h3_indices[0] + 3))
