from datetime import date

from scripts.build.build_peptide_h3_temporal_split import (
    metadata_eligible,
    metadata_record,
)


def _row(**updates):
    row = {
        "INSTANCE": "pdb_00009abc_H_L",
        "PDB_ID": "9abc",
        "PDBdepo": "2024-02-01",
        "holo": "True",
        "agtypes": "PEPTIDE",
        "agresolvedseqs": "DAEFRHDS",
        "VH_numerable_seq": "A" * 100,
        "VL_numerable_seq": "C" * 90,
        "CDRH3": "ARGYFDY",
        "ab_cluster": "ab1",
        "cdrh3_cluster": "h31",
        "agclusters": "['ag1']",
    }
    row.update(updates)
    return row


def test_temporal_metadata_filter_requires_post_training_peptide():
    config = {
        "minimum_antigen_length": 5, "maximum_antigen_length": 50,
        "minimum_h3_length": 4, "maximum_h3_length": 30,
    }
    assert metadata_eligible(_row(), config, date(2021, 8, 2)) is True
    assert metadata_eligible(_row(PDBdepo="2021-08-02"), config, date(2021, 8, 2)) is False
    assert metadata_eligible(_row(agtypes="PROTEIN/PEPTIDE"), config, date(2021, 8, 2)) is False


def test_temporal_metadata_record_uses_official_axes():
    record = metadata_record(_row())
    assert record["axis_values"]["official_ab_cluster"] == ("ab1",)
    assert record["axis_values"]["official_antigen_cluster"] == ("ag1",)
    assert record["axis_values"]["cdr_h3_sequence_exact"] == ("ARGYFDY",)
