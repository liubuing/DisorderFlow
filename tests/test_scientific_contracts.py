import numpy as np
import pytest
import torch


def test_contrastive_augmentation_keeps_an_immutable_sequence_target():
    from pathlib import Path

    source = Path('disorderflow/modules/bfn/core.py').read_text(encoding='utf-8')
    target_copy = source.index("x_seq_target = batch['aa'].clone()")
    input_copy = source.index('x_seq = x_seq_target.clone()')
    shuffle = source.index('x_seq[b, cdr_idx] =')
    assert target_copy < input_copy < shuffle
    assert 'x_seq_target = x_seq.clone()' not in source
    assert 'if self.training and mask_aug_prob > 0' in source


def test_validation_fixes_time_and_sampling_seed():
    from pathlib import Path

    source = Path('train.py').read_text(encoding='utf-8-sig')
    assert "batch['fixed_t'] = config.train.get('val_fixed_t', 0.5)" in source
    assert 'torch.manual_seed(validation_seed)' in source


def test_pipeline_uses_fp32_and_training_fails_closed_on_nan():
    from pathlib import Path

    pipeline = Path('scripts/run_v5_1_training_pipeline.sh').read_text(
        encoding='utf-8')
    training = Path('train.py').read_text(encoding='utf-8-sig')
    assert '--no_amp' in pipeline
    assert 'max_consecutive_failed_steps' in training
    assert 'consecutive training steps had no finite micro-batches' in training


def test_phase3_binary_metrics_handle_perfect_predictions_and_ties():
    from scripts.evaluate_v5_1_phase3 import binary_metrics

    perfect = binary_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    tied = binary_metrics([0.5, 0.5], [1, 0])
    assert perfect['auroc'] == pytest.approx(1.0)
    assert perfect['auprc'] == pytest.approx(1.0)
    assert tied['auroc'] == pytest.approx(0.5)


def test_sampling_propagates_disorder_condition_to_receiver():
    from pathlib import Path

    source = Path('disorderflow/modules/bfn/core.py').read_text(encoding='utf-8')
    sample_source = source[source.index('def sample('):]
    assert "epi_disorder = batch.get('epitope_disorder_profile', None)" in sample_source
    assert 'epitope_disorder=epi_disorder' in sample_source


def test_disorder_alignment_uses_global_continuous_scale_not_batch_quantiles():
    from pathlib import Path

    source = Path('disorderflow/modules/bfn/core.py').read_text(encoding='utf-8')
    block = source[source.index("disorder_align_w ="):source.index("losses['avg_t']")]
    assert 'disorder_align_scale' in block
    assert 'target_entropy' in block
    assert 'torch.quantile' not in block


def test_corrective_stage_freezes_previously_validated_heads():
    from disorderflow.utils.misc import load_config

    config, _ = load_config('configs/train/bfn_v5_1_stage5_corrective.yml')
    trainable = set(config.train.trainable_modules)
    assert 'contrastive_cdr_conv' in trainable
    assert 'disorder_proj' in trainable
    assert 'contact_head' not in trainable
    assert 'head_disorder' not in trainable


def test_corrective_pipeline_is_fail_closed_on_evaluation_gates():
    from pathlib import Path

    source = Path('scripts/run_v5_1_corrective.sh').read_text(encoding='utf-8')
    assert '--require-gates' in source
    assert '--baseline-checkpoint "$BASE_CHECKPOINT"' in source
    assert source.index('--require-gates') < source.index('> "$STATE_DIR/final_checkpoint.txt"')


def test_cross_sample_negative_uses_a_real_donor_cdr():
    from disorderflow.datasets.disorder_augmented import ContrastiveNegativeDataset

    class Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                'aa': torch.tensor([1, 2, 3]) if index == 0 else torch.tensor([1, 8, 9]),
                'generate_flag': torch.tensor([False, True, True]),
            }

    item = ContrastiveNegativeDataset(Dataset())[0]
    assert torch.equal(item['aa'], torch.tensor([1, 2, 3]))
    assert torch.equal(item['contrastive_negative_aa'], torch.tensor([1, 8, 9]))


def test_default_v5_1_stages_do_not_claim_unsupported_contrastive_training():
    from disorderflow.utils.misc import load_config

    for name in ('stage2_interface', 'stage3_codesign', 'stage4_idp_conditioned'):
        config, _ = load_config(f'configs/train/bfn_v5_1_{name}.yml')
        assert config.model.loss_weight.get('contrastive', 0) == 0
        assert 'contrastive' not in config.train.loss_weights


def test_conformation_loader_does_not_encode_rmsf_in_atom_masks(tmp_path):
    from disorderflow.datasets.conformation_dataset import ConformationRegressionDataset

    batch = {
        'aa': torch.tensor([0, 1]),
        'pos_heavyatom': torch.ones((2, 4, 3)),
        'mask_heavyatom': torch.ones((2, 4), dtype=torch.bool),
    }
    entry = {
        'batch': batch,
        'n_conformations': 2,
        'ca_positions': [np.zeros((2, 3)), np.full((2, 3), 2.0)],
        'rmsf': np.array([0.1, 9.0]),
        'disorder_label': np.array([0.0, 1.0]),
    }
    import pickle
    with open(tmp_path / '00000000.pkl', 'wb') as handle:
        pickle.dump(entry, handle)

    dataset = object.__new__(ConformationRegressionDataset)
    dataset._valid_indices = [0]
    dataset._use_lmdb = False
    dataset._pkl_dir = str(tmp_path)
    dataset.fix_backbone = False

    item = dataset[0]
    assert torch.equal(item['mask_heavyatom'], batch['mask_heavyatom'])
    assert torch.all(item['pos_heavyatom'][:, 0] == 1)
    assert torch.all(item['pos_heavyatom'][:, 2] == 1)


def test_af2_extraction_never_substitutes_ptm_for_iptm():
    from modules.af2_validator import extract_confidence_from_af2_results

    result = extract_confidence_from_af2_results([
        {'sequence': 'AAA', 'success': True, 'plddt': 80.0, 'ptm': 0.7,
         'iptm': None, 'validation_mode': 'monomer'}
    ])[0]
    assert result['iptm'] is None
    assert result['ptm'] == 0.7
    assert result['validation_mode'] == 'monomer'


def test_af2_cascade_keeps_monomer_interface_metric_missing():
    from modules.cascade_filter import apply_cascade_af2

    result = {
        'sequence': 'AAA', 'plddt': 80.0, 'ptm': 0.7, 'iptm': None,
        'max_pae': 10.0, 'validation_mode': 'monomer',
    }
    filtered, _ = apply_cascade_af2([result])
    assert filtered
    assert filtered[0]['iptm'] is None
    assert 0.0 <= filtered[0]['af2_composite_score'] <= 1.0


def test_disorder_lookup_is_case_insensitive():
    from disorderflow.datasets.disorder_augmented import DisorderAugmentedDataset

    class Dataset:
        def __getitem__(self, index):
            return {
                'aa': torch.tensor([0, 1, 2]),
                'fragment_type': torch.tensor([1, 3, 3]),
            }

        def __len__(self):
            return 1

    wrapped = DisorderAugmentedDataset(
        Dataset(), {'1abc': np.array([0.25, 0.75])}, ['1ABC'])
    item = wrapped[0]
    assert torch.allclose(
        item['epitope_disorder_profile'], torch.tensor([0.0, 0.25, 0.75]))


def test_disorder_profile_is_attached_before_phase3_patch():
    from disorderflow.datasets.disorder_augmented import DisorderAugmentedDataset

    class Dataset:
        ids = ['sample']

        def get_structure(self, index):
            return {
                'heavy': {'aa': torch.tensor([1])},
                'light': None,
                'antigen': {'aa': torch.tensor([2, 3, 4])},
            }

        def transform(self, structure):
            profile = structure['antigen']['epitope_disorder_profile']
            return {
                'aa': torch.tensor([1, 4, 2]),
                'fragment_type': torch.tensor([1, 3, 3]),
                'epitope_disorder_profile': torch.stack([
                    torch.tensor(0.0), profile[2], profile[0]]),
            }

        def __len__(self):
            return 1

    wrapped = DisorderAugmentedDataset(
        Dataset(), {'sample': np.array([0.1, 0.2, 0.9])}, ['sample'])
    item = wrapped[0]
    assert torch.allclose(
        item['epitope_disorder_profile'], torch.tensor([0.0, 0.9, 0.1]))
    assert item['epitope_disorder'].item() == pytest.approx(0.5)


def test_ema_checkpoint_state_uses_shadow_parameters():
    from disorderflow.utils.train import EMAModel

    model = torch.nn.Linear(2, 1)
    ema = EMAModel(model, decay=0.5)
    initial = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    ema.update(model)
    state = ema.averaged_state_dict(model)
    assert torch.allclose(state['weight'], initial + 1.0)
    assert torch.allclose(model.weight, initial + 2.0)


def test_frozen_manifest_rejects_dataset_drift(tmp_path):
    import json
    import lmdb
    import pickle

    from scripts.validate_v5_1_training import raw_manifest_report

    db_path = tmp_path / 'raw.lmdb'
    env = lmdb.open(str(db_path), map_size=1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(1))
        txn.put(b'00000000', pickle.dumps({'pdb_id': 'x', 'sequence': 'AAA'}))
    env.close()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'count': 1,
        'unique_pdb_ids': 1,
        'unique_sequences': 1,
        'canonical_fingerprint': 'wrong',
    }), encoding='utf-8')
    with pytest.raises(RuntimeError, match='manifest mismatch'):
        raw_manifest_report(str(db_path), manifest)


def test_v5_training_rejects_rank_only_rmsf_calibration(tmp_path):
    import json

    from scripts.validate_v5_1_training import calibration_report

    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({
        'source_count': 1301,
        'summary': {'verdict': 'rank_only', 'n_proteins': 50},
    }), encoding='utf-8')
    with pytest.raises(RuntimeError, match='does not support physical_continuous'):
        calibration_report(path, 1301)


def test_v5_rank_supervision_accepts_matching_calibration(tmp_path):
    import json

    from scripts.validate_v5_1_training import calibration_report

    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({
        'source_count': 1289,
        'summary': {'verdict': 'rank_only', 'n_proteins': 121},
        'subgroups': {
            'annotated_idp': {'verdict': 'rank_only', 'n_proteins': 19},
        },
    }), encoding='utf-8')
    report = calibration_report(path, 1289, 'rank_only')
    assert report['all_proteins']['verdict'] == 'rank_only'


def test_v5_training_requires_idp_subset_calibration(tmp_path):
    import json

    from scripts.validate_v5_1_training import calibration_report

    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({
        'source_count': 1301,
        'summary': {'verdict': 'pass_physical_label', 'n_proteins': 50},
        'subgroups': {
            'annotated_idp': {'verdict': 'rank_only', 'n_proteins': 15},
        },
    }), encoding='utf-8')
    with pytest.raises(RuntimeError, match='annotated-IDP subset'):
        calibration_report(path, 1301)
