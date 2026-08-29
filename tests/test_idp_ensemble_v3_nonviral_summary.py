from scripts.summarize_idp_ensemble_v3_nonviral import hamming, warning_count


def test_nonviral_summary_helpers():
    assert hamming("AAAA", "ACCA") == 2
    row = {
        "n_linked_glycosylation_motifs": [],
        "deamidation_motifs": ["NG"],
        "isomerization_motifs": [],
        "oxidation_residues": ["M"],
    }
    assert warning_count(row) == 2
