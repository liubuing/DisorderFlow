from scripts.analyze_idp_ensemble_v3_nonviral import sequence_metrics

CONTRACT = {
    "allowed_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
    "maximum_hydrophobic_run": 4,
    "maximum_absolute_charge_proxy": 8,
    "flag_deamidation_motifs": ["NG", "NS", "NN", "NQ"],
    "flag_isomerization_motifs": ["DG", "DS", "DT", "DD"],
    "flag_oxidation_residues": ["M", "W"],
}


def test_sequence_qc_flags_glycosylation_and_long_hydrophobic_run():
    glyco = sequence_metrics("ARNVTGY", CONTRACT)
    assert glyco["n_linked_glycosylation_motifs"] == ["NVT"]
    assert glyco["qc_pass"] is False
    hydrophobic = sequence_metrics("AFFFFFG", CONTRACT)
    assert hydrophobic["checks"]["hydrophobic_run_ok"] is False
