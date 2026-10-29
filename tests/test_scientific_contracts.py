"""v5.1 训练管线科学契约的守护测试：对比增广目标不可变性、验证期固定 t 与种子、fp32 与 NaN
fail-closed、二元指标平局口径、无序条件传播与全局连续对齐尺度、纠正阶段冻结已验证头、
跨样本负例真实性、AF2 置信指标不可替代（ptm≠iptm）、无序查找大小写不敏感与 patch 前挂载、
EMA 影子参数、冻结清单漂移拒绝，以及 RMSF 校准判定与标注 IDP 子集门槛。"""
import numpy as np
import pytest
import torch


# 源码契约：对比增广必须先克隆目标（x_seq_target）再复制输入再打乱——防止目标被增广污染，
# 且打乱只在训练态与概率开关下发生。
def test_contrastive_augmentation_keeps_an_immutable_sequence_target():
    from pathlib import Path

    source = Path('disorderflow/modules/bfn/core.py').read_text(encoding='utf-8')
    target_copy = source.index("x_seq_target = batch['aa'].clone()")
    input_copy = source.index('x_seq = x_seq_target.clone()')
    shuffle = source.index('x_seq[b, cdr_idx] =')
    assert target_copy < input_copy < shuffle
    assert 'x_seq_target = x_seq.clone()' not in source
    assert 'if self.training and mask_aug_prob > 0' in source


# 验证确定性：验证循环必须固定 t（默认 0.5）与采样种子，防止验证指标随机波动。
def test_validation_fixes_time_and_sampling_seed():
    from pathlib import Path

    source = Path('train.py').read_text(encoding='utf-8-sig')
    assert "batch['fixed_t'] = config.train.get('val_fixed_t', 0.5)" in source
    assert 'torch.manual_seed(validation_seed)' in source


# 数值纪律：管线禁用 AMP（fp32 训练），训练循环对连续非有限 micro-batch 必须累积失败并终止——
# 防止 NaN 静默传播毁掉整轮训练。
def test_pipeline_uses_fp32_and_training_fails_closed_on_nan():
    from pathlib import Path

    pipeline = Path('scripts/run_v5_1_training_pipeline.sh').read_text(
        encoding='utf-8')
    training = Path('train.py').read_text(encoding='utf-8-sig')
    assert '--no_amp' in pipeline
    assert 'max_consecutive_failed_steps' in training
    assert 'consecutive training steps had no finite micro-batches' in training


# 指标平局：完美预测 AUROC/AUPRC 均为 1，全平局 AUROC 必须为 0.5（机会水平）——
# 防止平局被误报为完美或零判别力。
def test_phase3_binary_metrics_handle_perfect_predictions_and_ties():
    from scripts.evaluate_v5_1_phase3 import binary_metrics

    perfect = binary_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    tied = binary_metrics([0.5, 0.5], [1, 0])
    assert perfect['auroc'] == pytest.approx(1.0)
    assert perfect['auprc'] == pytest.approx(1.0)
    assert tied['auroc'] == pytest.approx(0.5)


# 条件传播：sample() 必须从 batch 读取表位无序度 profile 并传给 receiver——
# 防止采样路径丢失条件输入导致生成与训练分布脱节。
def test_sampling_propagates_disorder_condition_to_receiver():
    from pathlib import Path

    source = Path('disorderflow/modules/bfn/core.py').read_text(encoding='utf-8')
    sample_source = source[source.index('def sample('):]
    assert "epi_disorder = batch.get('epitope_disorder_profile', None)" in sample_source
    assert 'epitope_disorder=epi_disorder' in sample_source


# 对齐尺度：无序对齐权重必须用全局连续 scale 与目标熵，禁止按 batch 分位数归一——
# 防止 batch 内归一化使权重随小批量成分抖动。
def test_disorder_alignment_uses_global_continuous_scale_not_batch_quantiles():
    from pathlib import Path

    source = Path('disorderflow/modules/bfn/core.py').read_text(encoding='utf-8')
    block = source[source.index("disorder_align_w ="):source.index("losses['avg_t']")]
    assert 'disorder_align_scale' in block
    assert 'target_entropy' in block
    assert 'torch.quantile' not in block


# 冻结纪律：纠正阶段只解冻对比/无序投影模块，已验证的 contact_head 与 head_disorder 必须保持冻结——
# 防止后续微调破坏先前验证过的预测头。
def test_corrective_stage_freezes_previously_validated_heads():
    from disorderflow.utils.misc import load_config

    config, _ = load_config('configs/train/bfn_v5_1_stage5_corrective.yml')
    trainable = set(config.train.trainable_modules)
    assert 'contrastive_cdr_conv' in trainable
    assert 'disorder_proj' in trainable
    assert 'contact_head' not in trainable
    assert 'head_disorder' not in trainable


# fail-closed 管线：纠正脚本必须带 --require-gates 与基线 checkpoint，且在写出最终 checkpoint 之前——
# 防止评估门未通过也产出正式权重。
def test_corrective_pipeline_is_fail_closed_on_evaluation_gates():
    from pathlib import Path

    source = Path('scripts/run_v5_1_corrective.sh').read_text(encoding='utf-8')
    assert '--require-gates' in source
    assert '--baseline-checkpoint "$BASE_CHECKPOINT"' in source
    assert source.index('--require-gates') < source.index('> "$STATE_DIR/final_checkpoint.txt"')


# 负例真实性：跨样本负例必须来自真实供体样本的生成区（而非合成打乱），
# 保证对比信号反映真实序列多样性。
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


# 声明诚实：stage2-4 默认配置的对比损失权重必须为 0 且不得在 train.loss_weights 中声称对比训练——
# 防止配置层宣称未被验证的训练方式。
def test_default_v5_1_stages_do_not_claim_unsupported_contrastive_training():
    from disorderflow.utils.misc import load_config

    for name in ('stage2_interface', 'stage3_codesign', 'stage4_idp_conditioned'):
        config, _ = load_config(f'configs/train/bfn_v5_1_{name}.yml')
        assert config.model.loss_weight.get('contrastive', 0) == 0
        assert 'contrastive' not in config.train.loss_weights


# 掩码纯净：构象加载器不得把 RMSF 信息编码进原子掩码或坐标（mask 与 pos 原样透传）——
# 防止监督信号从输入特征泄漏。
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


# 指标不可替代：monomer 模式下 iptm 缺失必须保持 None，不得用 ptm 冒充——
# 防止界面置信分被单体分污染。
def test_af2_extraction_never_substitutes_ptm_for_iptm():
    from modules.af2_validator import extract_confidence_from_af2_results

    result = extract_confidence_from_af2_results([
        {'sequence': 'AAA', 'success': True, 'plddt': 80.0, 'ptm': 0.7,
         'iptm': None, 'validation_mode': 'monomer'}
    ])[0]
    assert result['iptm'] is None
    assert result['ptm'] == 0.7
    assert result['validation_mode'] == 'monomer'


# 级联一致性：monomer 结果通过级联后 iptm 仍为 None，复合分归一到 [0,1]——防止缺失指标被填默认值。
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


# 大小写不敏感：PDB ID 大小写差异不得导致无序 profile 匹配失败（1abc vs 1ABC）——防止静默缺档。
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


# 挂载时序：无序 profile 必须在 phase3 patch（重排）之前挂载并随之重排，且聚合量按 patch 后计算——
# 防止 profile 与残基错位。
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


# EMA 语义：averaged_state_dict 必须返回影子（衰减后）参数而非当前参数——防止导出的"平均权重"实为瞬时权重。
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


# 漂移拒绝：LMDB 实际内容与冻结清单的 canonical_fingerprint 不符必须 RuntimeError——
# 防止训练数据被静默替换后仍沿用旧清单。
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


# 校准匹配：要求 physical_continuous 监督时，rank_only 判定必须被拒绝——防止把序数校准当连续校准。
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


# 一致放行：监督要求为 rank_only 且校准判定一致（含子组）时必须通过，防止误拒。
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


# 子组门槛：标注 IDP 子集判定弱于总体（rank_only vs pass_physical_label）时必须报错——
# 防止用总体达标掩盖关键子集监督不足。
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
