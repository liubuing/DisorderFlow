import json
from pathlib import Path

import torch

from disorderflow.models.bfn_model import build_antigen_sequence_mismatch
from disorderflow.utils.protein.constants import Fragment


ROOT = Path(__file__).resolve().parents[1]


def test_antigen_mismatch_preserves_scaffold_and_target_geometry_layout():
    batch = {
        'aa': torch.tensor([[1, 2, 3, 4, 21], [5, 6, 7, 8, 9]]),
        'fragment_type': torch.tensor([
            [0, 0, int(Fragment.Antigen), int(Fragment.Antigen), 0],
            [0, 0, int(Fragment.Antigen), int(Fragment.Antigen), int(Fragment.Antigen)],
        ]),
        'mask': torch.tensor([
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]),
    }
    mismatch, valid = build_antigen_sequence_mismatch(batch)

    assert mismatch[:, :2].equal(batch['aa'][:, :2])
    assert mismatch[0, 2:4].tolist() == [7, 9]
    assert mismatch[1, 2:5].tolist() == [3, 3, 4]
    assert valid.tolist() == [True, True]


def test_singleton_batch_cannot_create_antigen_mismatch():
    batch = {
        'aa': torch.tensor([[1, 2]]),
        'fragment_type': torch.tensor([[0, int(Fragment.Antigen)]]),
        'mask': torch.ones(1, 2, dtype=torch.bool),
    }
    assert build_antigen_sequence_mismatch(batch) == (None, None)


def test_successor_manifest_is_development_only_and_excludes_frozen_sets():
    manifest = json.loads((
        ROOT / 'data/successor_v3_development/manifest.json'
    ).read_text(encoding='utf-8'))
    assert manifest['status'] == 'frozen_development_only; not confirmatory'
    assert manifest['counts']['components'] >= 12
    assert manifest['counts']['representatives'] == manifest['counts']['components']
    assert not any(manifest['prohibited_overlap'].values())


def test_successor_training_uses_real_sequence_specificity_loss():
    model_source = (
        ROOT / 'disorderflow/models/bfn_model.py').read_text(encoding='utf-8')
    core_source = (
        ROOT / 'disorderflow/modules/bfn/core.py').read_text(encoding='utf-8')
    config = (
        ROOT / 'configs/train/bfn_successor_v3_h3_specificity.yml'
    ).read_text(encoding='utf-8')
    assert "batch['antigen_mismatch_pair_feat']" in model_source
    assert "losses['antigen_mismatch_rank']" in core_source
    assert 'selection: H_CDR3' in config
    assert 'prohibited_initializer: stage_b' in config
    assert ('allowed_initializer_sha256: '
            '00ae022962bb35bd1ab65e4fea912715d3e99d09c0858ed6b8452ece32e9678f'
            in config)


def test_successor_passes_only_development_gates_and_preserves_claim_boundary():
    analysis = json.loads((
        ROOT / 'results/successor_v3_development/analysis.json'
    ).read_text(encoding='utf-8'))
    lineage = json.loads((
        ROOT / 'publication/successor_v3_checkpoint_lineage.json'
    ).read_text(encoding='utf-8'))
    assert analysis['classification'] == (
        'exposed development only; not confirmatory')
    assert analysis['successor']['valid_components'] == 52
    assert analysis['successor']['component_bootstrap_ci95'][1] < 0
    assert analysis['successor']['contact']['auroc'] >= 0.60
    assert analysis['all_development_gates_passed'] is True
    assert lineage['development_evaluation']['all_development_gates_passed'] is True
    assert 'does not reverse the v2 terminal negative' in lineage['claim_boundary']
