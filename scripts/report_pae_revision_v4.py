"""Build revision documents directly from experiment artifacts, not literals."""
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.harden_pae_surrogate_v4 import OUT, read, write, digest, correlation, SEEDS


def main():
    report = read(OUT/'experiment_report.json')
    manifest = read(OUT/'data_manifest.json')
    timing = read(OUT/'timing.json')
    external = ROOT/'data/pae_surrogate_external_v4'
    audit = read(external/'isolation_audit.json')
    discovery = read(external/'discovery.json')
    corrected = {}
    predictions = read(OUT/'predictions.json')
    permutation_deltas = {}
    for seed in SEEDS:
        score = dict(zip(predictions['entities'],predictions['models'][f'full_s{seed}']))
        permutation_deltas[str(seed)] = {f'EXT{i:03d}':abs(
            score[f'EXT{i:03d}|control|native']-score[f'EXT{i:03d}|control|composition_shuffle']) for i in range(1,7)}
    write(OUT/'permutation_diagnostic.json',{
        'classification':'posthoc_architecture_diagnostic_not_binding_test',
        'native_shuffle_absolute_score_differences':permutation_deltas,
        'interpretation':'On a fixed design backbone, this pooled scalar branch has no H3 sequence-order information beyond composition. Small nonzero differences can be floating-point reduction noise.'})
    historical_sources = {}
    for seed in SEEDS:
        path = ROOT/f'results/candidate_interface_per_entity_records/transfer_s{seed}_entity_records.json'
        data = read(path)
        historical_sources[str(path.relative_to(ROOT))] = digest(path)
        by = defaultdict(list)
        for r in data['records']:
            by[r['scaffold_family']].append(r)
        corrected[str(seed)] = {s:correlation([r['pred_pae'] for r in rs],[r['target_pae'] for r in rs]) for s,rs in by.items()}
    corrections = {
        'classification':'superseding_correction_ledger; frozen_v1_v3_artifacts_unchanged',
        'historical_source_hashes':historical_sources,
        'corrected_historical_per_scaffold_spearman':corrected,
        'issues':[
            {'id':'counts','old':'84 entities per scaffold','correct':'14 per external scaffold, 84 total, 504 teacher replicates. Historical GP2 test: 15+16+17=48 entities, 288 replicates.'},
            {'id':'direction','old':'negative NLL/PAE rho is favorable','correct':'Both lower-is-better; positive rho aligns rankings. Signed rho must be retained.'},
            {'id':'target','old':'MPNN comparison uses identical PAE target','correct':'Old baseline uses runner interface_pae; v3 model uses localized patch PAE. v4 recomputes one full H3-to-antigen directional target for ALL methods.'},
            {'id':'input_provenance','old':'Historical v3 establishes screening before AF2','correct':'v3 build_entry constructs input from result[pdb], an AF2 output. v4 features use generation_work/structures only.'},
            {'id':'threshold','old':'every seed and every scaffold above 0.5','correct':'Median-per-scaffold gate passed; a GP2 per-scaffold rho is 0.456. These are different statements.'},
            {'id':'per_scaffold_table','old':'hand-entered Table 3 and supplement values','correct':'Recomputed historical per-scaffold values are recorded above; do not mix them with the new v4 target.'},
            {'id':'ablation','old':'different prediction targets prove interface conditioning','correct':'Only matched same-target module ablations address this. v4 no-antigen ablation does not support necessity of antigen context.'},
            {'id':'cost','old':'2900-4900x and 1/6 cost both stated','correct':'Neither is retained as a validated end-to-end speedup. v4 reports measured PDB-to-score time and budget/recall, with historical teacher timing separately scoped.'},
            {'id':'test_status','old':'test never evaluated in README/reproducibility','correct':'Historical GP2 test was evaluated; v4 excludes it entirely. No new blind external test completed.'},
            {'id':'abstention','old':'deployment abstention policy','correct':'Historical gates require labels and are retrospective evaluability gates, not a validated label-free per-candidate abstention detector.'},
        ]}
    write(OUT/'corrections.json',corrections)
    feasibility = {'classification':'model_free_external_feasibility','counts':audit['counts'],
        'snapshot_entries':discovery['counts']['total_entries'],
        'decision':'No new independent cohort: all newly eligible complexes fail reference homology isolation.',
        'new_blind_validation_complete':False,'thresholds_relaxed':False,
        'old_test_reused':False,'source_audit_sha256':digest(external/'isolation_audit.json')}
    write(OUT/'external_feasibility.json',feasibility)
    models = report['models']
    labels = [('ridge_alpha_10','Ridge'),('extra_trees_300_min_leaf_3','Extra Trees'),
              ('mpnn_nll_signed_same_target','ProteinMPNN NLL')]
    labels += [(f'{variant}_s{seed}',f'{variant}, seed {seed}') for variant in ('full','no_antigen_context','no_noise_weight','no_grouped_difference') for seed in SEEDS]
    lines = ['| Method | Median scaffold rho | Top-20% recall at 3/14 budget | Top-20% recall at 7/14 budget |',
             '|---|---:|---:|---:|']
    for key,label in labels:
        m = models[key]
        lines.append(f"| {label} | {m['median_scaffold_spearman']:.3f} | {m['macro_top20_recall']['0.2']:.3f} | {m['macro_top20_recall']['0.5']:.3f} |")
    table = '\n'.join(lines)
    latency = [r['seconds_per_entity_amortized'] for r in timing['scaffolds']]
    main_text = f'''# Pre-AF2 interface-PAE ranking: a controlled revision of DisorderFlow

Status: **PAE branch development draft; not ready for journal submission**. This v4 document supersedes the v3 surrogate manuscript's scientific interpretation. It is separate from the primary ECLS manuscript (`publication/MANUSCRIPT_DRAFT.md`) and does not replace its frozen result. Publication routing is defined in `docs/PUBLICATION_MAP.md`. Historical artifacts remain unchanged. All new results below are generated by `scripts/report_pae_revision_v4.py` from versioned JSON artifacts.

## Abstract

We audited an antibody-peptide confidence surrogate and found that its historical evaluation consumed AlphaFold2 (AF2) output coordinates, while its inverse-folding comparator used a different PAE aggregation. We constructed a separate pre-AF2 experiment using only fixed design backbones and candidate sequences as inputs, and one directional CDR-H3-to-antigen teacher target for every method. We trained on 240 candidate entities from 15 components, selected scalar-head checkpoints on 32 entities from two calibration components, and descriptively evaluated 84 entities on six previously exposed external components. Simple supervised baselines matched or exceeded the three-seed full model, and removing antigen context did not degrade the median rank correlation. Screening the best three of 14 candidates retained 61.1-77.8% of the teacher's best three for the full model; these are AF2-relative recall values, not experimental binding rates. A new temporal PDB search yielded no independent eligible component after homology isolation. The results support a narrowly scoped development benchmark and computational screening signal, but do not establish an advantage of the complex architecture or new blind generalization.

## Methods

### Data, provenance and separation

The original 15/2 component training/calibration split was retained. Its previously opened three-component GP2 test was excluded from data extraction, training, selection and evaluation. The six old external components are explicitly labeled exposed descriptive transfer, not a new confirmatory test. Each external component has 12 generated candidates and two controls. All 356 retained entities have six AF2 label replicates, yielding 2,136 teacher records. Independent biological units are components, not candidates or AF2 replicates. Existing encoder pretraining provenance remains a limitation for any stronger independence claim.

Inputs come exclusively from fixed `generation_work/structures` PDBs with candidate H3 sequence injection. AF2 coordinates, confidence fields and labels are not features. The scalar label is the mean PAE over the fixed H3 positions and **all** antigen positions in the heavy-to-antigen direction, divided by 31, then averaged over two AF2 model protocols and three seeds. This is a new v4 estimand; it is not silently substituted into historical v3 claims. Matrix checksums and sizes are verified. Sequence indices, not model-dependent geometry or teacher scores, determine the target mask.

### Models and controls

The pre-existing frozen encoder and sequence embedding are reused. Fresh scalar heads implement the pooled v2 PAE branch: linear projections of mean H3 sequence embedding, mean antigen sequence embedding and mean H3-antigen pair embedding are summed, then passed through LayerNorm, ReLU, dropout and a sigmoid scalar readout. This cached computation is algebraically equivalent to pooling the branch's linear projections. These are new scalar-only models, not a reproduction of v3 joint pLDDT/ipTM/PAE training.

All three seeds (2041, 2053, 2069) use 1,000 full-batch AdamW updates, learning rate 0.001, weight decay 0.0001, gradient clipping 1, and checkpoint selection by calibration MSE every 50 steps. Training uses a noise-weighted squared error plus 0.5 times a within-component squared difference loss. Noise weight is clipped 1/(1+(replicate SD/0.1)^2), in [0.1,1]. Matched ablations remove antigen/pair context, noise weights, or grouped differences separately. Their heads use the same seed-specific initialization and selection rule. They establish facts about v4 only.

Ridge (alpha 10) and Extra Trees (300 trees, minimum leaf size 3) use identical labels and sequence-composition/length plus design-backbone distance features. Standardization is fit on training only. ProteinMPNN NLL is reused as a sequence score and compared against the **same v4 target**. NLL and PAE both favor smaller values, so positive correlation aligns rankings; negative correlation does not. No sign is selected using external results and no absolute correlation is used.

### Statistics and screening

We report per-component signed Spearman, its median, reliable-pair accuracy at joint-SEM multipliers 0, 0.5, 1 and 2, and top-20% target recall at nominal budgets 10%, 20% and 50%. Actual counts are rounded up: 2, 3 and 7 of 14. Ties use expected inclusion under uniform random tie breaking; constant predictions therefore score at chance. No gate is described as a label-free deployment abstention detector. Paired component-bootstrap differences use 10,000 resamples and are descriptive, without multiple-comparison significance claims. Full per-entity predictions, sensitivity results and intervals are in `results/pae_surrogate_revision_v4/`.

## Results

{table}

The full model does not consistently outperform Ridge. Removing antigen context yields median component correlations 0.850-0.867, exceeding each matched full model on this summary. Noise weighting and grouped differences also show no consistent benefit across seeds. We therefore withdraw claims that these modules have demonstrated necessity or superiority. The ProteinMPNN score is an unfavorable ranking direction on this AF2-relative task; its absolute correlation must not be presented as competitive performance.

At a 3/14 budget, the full models retain 61.1-77.8% of the teacher's top three candidates, compared with 72.2% for Ridge and random expected recall 21.4%. This reduces candidate-level AF2 evaluations by 78.6%, conditional on accepting the observed recall loss. At a 7/14 budget, recall is 88.9-97.2% with 50% fewer candidate evaluations. These are small-cohort retrospective estimates; no wet-lab enrichment or binding claim follows.

Measured design-PDB-to-score latency on {timing['device_name']} ranges from {min(latency):.4f} to {max(latency):.4f} seconds per candidate, amortized over 14 candidates on a scaffold. Measurement includes PDB parsing, patch construction, frozen pair features, candidate injection and scalar scoring, using five repetitions after one warmup. Model loading ({timing['model_load_seconds']:.2f} s) is reported separately. Historical AF2 timing was not remeasured under identical conditions, so no hardware-normalized thousands-fold speedup is claimed. Timing predictions were checked against the stored experiment predictions.

### New external cohort feasibility

A query frozen before retrieval searched experimental antibody structures released from 2026-09-02 through 2026-09-16. It returned 45 entries, including five metadata candidates when the viral pool was included. Structural processing retained two antibody-peptide complexes. Both failed the unchanged five-axis reference-homology screen; zero independent components remained. No confidence labels or performance scores were generated for this pool. The prespecified minimum of six components and three antigen families was not met. This is a completed **feasibility audit**, not completed external validation.

## Limitations and publication decision

All labels are AF2-Multimer single-sequence outputs, not experimental binding or structural accuracy. The six external components were already exposed, encoder provenance is not newly independent, and there is no fresh confirmation cohort. The benchmark has only three selected candidates per generator arm per external scaffold, limiting conclusions about within-generator selection and possible generator-distribution confounding. The head ablations evaluate a revised scalar-only design, not the historical joint model. Six-component uncertainty is large. We do not infer antigen-specific recognition merely from agreement with an AF2 teacher.

An additional architecture diagnostic found that the pooled scalar branch does not encode H3 sequence order on a fixed backbone: the candidate positions are masked in pair encoding and residue embeddings are averaged. Native and composition-shuffled controls therefore produce effectively identical scores (maximum observed absolute difference {max(v for d in permutation_deltas.values() for v in d.values()):.2g} across 18 seed/component contrasts, consistent with floating-point reduction noise). The diagnostic is recorded separately as post hoc. This limits any sequence-specific interpretation and explains why composition baselines are essential.

The present evidence does **not** establish a sufficiently strong new-method advantage for a confident Bioinformatics Original Paper submission. The practical pre-AF2 screening signal is worth retaining, but model complexity is not justified by these results. A future methodological claim requires an independent multi-family cohort and a reproducible advantage over the simple supervised baseline; a binding claim additionally requires appropriate experimental measurements. Failed or null comparisons must remain in the manuscript.

## Reproducibility

Run `python scripts/harden_pae_surrogate_v4.py`, `python scripts/benchmark_pae_revision_v4.py`, then `python scripts/report_pae_revision_v4.py`. The protocol predates the new model results. Do not retune based on the exposed external set or reopen the historical GP2 test. `python -m pytest tests/test_pae_surrogate_revision_v4.py -q` checks target direction, tie handling, correlation sign, ablation isolation and split provenance. Data-source and checkpoint hashes accompany the experiment. No historical frozen artifact or uploaded release was replaced.

Journal scope consulted: [Bioinformatics scope guidelines](https://academic.oup.com/bioinformatics/pages/scope_guidelines). This document is a corrected research draft, not a claim of journal acceptance.
'''
    pub = ROOT/'publication/af2_interface_pae_surrogate_manuscript_v4.md'
    pub.write_text(main_text,encoding='utf-8')
    cn = f'''# DisorderFlow PAE 分支四项整改结果

本文仅评估 PAE 代理排序分支。当前主论文和发布主线是 ECLS，入口为 `publication/MANUSCRIPT_DRAFT.md`；PAE 的修订结论不替代 ECLS 已冻结的最终评估。统一定位见 `docs/PUBLICATION_MAP.md`。

结论：已完成证据修正、同目标基线与三种模块消融、筛选收益实测，以及新外部数据的可行性审计。**尚未补成可支撑正刊创新性的新证据：完整模型没有稳定优于简单基线，新外部独立队列为 0。不能据此宣称已经达到 Bioinformatics 投稿标准。**

| 工作项 | 实际完成 | 结果与边界 |
|---|---|---|
| 创新性/基线 | Ridge、Extra Trees、同目标 ProteinMPNN；完整模型及三种消融各 3 种子，共 12 次训练 | 完整模型 rho 0.638–0.824；Ridge 0.839；去抗原上下文 0.850–0.867。复杂模块优势未证实 |
| 外部覆盖 | 检索 45 个新发布结构，处理 5 个候选，2 个结构合格，执行五轴同源隔离 | 2 个均与参考集同源，新增独立组件 0；未伪装为盲测，也未降低隔离门槛 |
| 筛选收益 | 10%/20%/50% 预算，实际取 2/3/7 个候选；并实测原始 PDB 到评分耗时 | 取 3/14 时保留最优 3 个的 61.1–77.8%，少运行 78.6% 的候选 AF2 评估；这是计算代理召回率 |
| 论文纠错 | 新 v4 稿、逐项纠错账本、自动生成结果表 | 修正数量、方向、目标不一致、输入来源、阈值、手填表、成本、测试集状态与拒判含义 |

额外发现的关键问题：v3 数据构造读取 AF2 输出 PDB，因此它的结果不能直接支持运行 AF2 前的筛选。v4 已另建只使用原始设计骨架和候选序列的流程；AF2 矩阵只作为监督标签。

{table}

新训练使用 15 个训练组件的 240 个实体、2 个校准组件的 32 个实体；描述性外部评估为 6 个组件的 84 个实体。已有 GP2 最终测试集完全没有在 v4 中重新使用。外部旧数据已被查看，不能称为新的确认性验证。

完整模型的原始 PDB 到评分实测为每候选 {min(latency):.4f}–{max(latency):.4f} 秒（每骨架 14 个候选分摊；包含解析、特征与评分，不含模型载入）。未对旧 AF2 耗时做同硬件复测，因此撤回“2900–4900 倍”端到端加速主张。

另一个结构性限制：新标量分支对 H3 残基嵌入取均值，固定骨架上的组成相同、顺序不同序列得到几乎相同的分数。18 个原生/打乱对照的最大评分差为 {max(v for d in permutation_deltas.values() for v in d.values()):.2g}。这不能被解释为具备序列顺序特异性的抗原识别能力；该项为事后结构诊断。

配对组件 bootstrap 区间和噪声阈值敏感性见 `experiment_report.json`；这些只作描述，不作多重比较后的显著性声明。消融针对新 v4 标量预测头，不冒充原 v3 联合训练的消融。

下一步科研条件很明确：取得真正独立、跨抗原家族的数据，并证明方法稳定优于 Ridge 等简单监督基线；否则应按基准/方法边界研究重新定位，而不是继续强化复杂模型优越性的叙述。此次没有制造正结果，也没有发布或替换上传包。

产物：

- `publication/af2_interface_pae_surrogate_manuscript_v4.md`：当前修订稿。
- `results/pae_surrogate_revision_v4/corrections.json`：原稿错误与更正依据。
- `results/pae_surrogate_revision_v4/experiment_report.json`：基线、消融、区间和筛选收益。
- `results/pae_surrogate_revision_v4/timing.json`：实测耗时。
- `data/pae_surrogate_external_v4/isolation_audit.json`：新增外部队列隔离结果。
'''
    (ROOT/'docs/PAE_SURROGATE_REVISION_V4.md').write_text(cn,encoding='utf-8')
    write(OUT/'revision_status.json',{
        'scope':'pae_surrogate_branch_only_not_primary_ecls_manuscript',
        'manuscript':str(pub.relative_to(ROOT)), 'manuscript_sha256':digest(pub),
        'four_workstreams':{'integrity':'corrected_in_superseding_v4',
            'baselines_and_ablations':'completed_no_complex_model_advantage',
            'screening':'completed_descriptive_computational_benefit',
            'external_validation':'feasibility_completed_zero_independent_components_validation_incomplete'},
        'submission_ready':False,
        'script_sha256':{p.name:digest(p) for p in [ROOT/'scripts/harden_pae_surrogate_v4.py',ROOT/'scripts/benchmark_pae_revision_v4.py',Path(__file__)]}})
    print('Wrote v4 manuscript, Chinese report, correction ledger and feasibility decision.')


if __name__=='__main__':
    main()
