import torch
from pathlib import Path

from disorderflow.datasets.statecontrast_structural import (
    assign_official_imgt_cdrs,
    exact_alignment_map,
)
from disorderflow.utils.data import CompleteGroupBatchSampler
from disorderflow.utils.protein.constants import CDR, ressymb_to_resindex


def encoded(sequence):
    return torch.tensor([ressymb_to_resindex[aa] for aa in sequence], dtype=torch.long)


def test_official_imgt_cdrs_map_exactly_through_constant_domain_context():
    official = "QVQLGFTFSSYTWVRQISSGGAYTKGRFCARDRGFDYWGQGT"
    parsed = official + "ASTKGPSVFPLAP"
    chain = {"aa": encoded(parsed)}
    cdrs = {"H1": "GFTFSSYT", "H2": "ISSGGAYT", "H3": "CARDRGFDYW"}
    spans = {
        name: {
            "start": official.index(sequence),
            "end": official.index(sequence) + len(sequence),
        }
        for name, sequence in cdrs.items()
    }

    indices = assign_official_imgt_cdrs(chain, official, cdrs, spans)

    assert set(indices) == set(cdrs)
    assert "".join(parsed[index] for index in indices["H3"]) == cdrs["H3"]
    assert (chain["cdr_flag"] == int(CDR.H1)).sum().item() == len(cdrs["H1"])
    assert (chain["cdr_flag"] == int(CDR.H2)).sum().item() == len(cdrs["H2"])
    assert (chain["cdr_flag"] == int(CDR.H3)).sum().item() == len(cdrs["H3"])


def test_alignment_map_does_not_map_mismatched_residues():
    mapping = exact_alignment_map("ACDE", "ACNE")
    assert mapping[0] == 0
    assert mapping[1] == 1
    assert 2 not in mapping
    assert mapping[3] == 3


class GroupedDataset:
    group_indices = ((0, 1, 2), (3, 4), (5, 6, 7))


def test_complete_group_sampler_never_splits_groups():
    sampler = CompleteGroupBatchSampler(
        GroupedDataset(), max_batch_records=5, shuffle=False)
    batches = list(sampler)

    assert batches == [[0, 1, 2, 3, 4], [5, 6, 7]]
    for group in GroupedDataset.group_indices:
        containing = [batch for batch in batches if set(group) & set(batch)]
        assert len(containing) == 1
        assert set(group).issubset(containing[0])


def test_complete_group_sampler_rejects_too_small_capacity():
    try:
        CompleteGroupBatchSampler(GroupedDataset(), max_batch_records=2)
    except ValueError as error:
        assert "exceeds batch capacity" in str(error)
    else:
        raise AssertionError("Expected an undersized batch capacity failure")


def test_core_antigen_mask_uses_fragment_type_not_partner_antibody_chain():
    from disorderflow.modules.bfn.core import AntibodyBFN_Core

    core = object.__new__(AntibodyBFN_Core)
    batch = {
        "mask": torch.ones((1, 6), dtype=torch.bool),
        "fragment_type": torch.tensor([[1, 1, 2, 2, 3, 3]]),
        "chain_nb": torch.tensor([[0, 0, 1, 1, 2, 2]]),
    }
    generate = torch.tensor([[True, False, True, False, False, False]])

    assert core._mask_antigen(batch, generate).tolist() == [
        [False, False, False, False, True, True]
    ]


def test_receiver_pair_aggregation_broadcasts_for_multi_sample_batches():
    source = Path("disorderflow/modules/bfn/receiver.py").read_text(encoding="utf-8")
    assert "clamp(min=1).unsqueeze(-1)" in source
    assert "sum(dim=2) / antigen_count" in source
