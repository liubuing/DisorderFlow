"""Post-hoc teacher sensitivity with frozen predictions; no model selection."""
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import read,freeze,digest
from scripts.run_pae_public_pilot_v11 import OUT,verify,screening_metrics


def main():
    verify()
    freeze(OUT/'sensitivity_protocol.json',{
        'classification':'post_hoc_descriptive_teacher_sensitivity',
        'teacher_sha256':digest(OUT/'teacher_labels.json'),
        'prediction_sha256':digest(OUT/'pre_teacher_predictions.json'),
        'scenarios':'full6, model1/2 averaging3seeds, eachseed averaging2models, leave1seedout, eachof6individualteacher slots',
        'no_refit_or_threshold_selection':True,
        'boundary':'repeats share candidates and are not independent biological validation; scenario ranges are not confidence intervals',
        'script_sha256':digest(Path(__file__))})
    rows=read(OUT/'teacher_labels.json')['rows']
    pred=read(OUT/'pre_teacher_predictions.json')['predictions']
    groups=read(OUT/'holdout.json')['components']
    scenarios={'full6':lambda o:True}
    for m in (1,2): scenarios[f'model{m}']=lambda o,m=m:o['model']==m
    for s in (7103,7111,7121):
        scenarios[f'seed{s}']=lambda o,s=s:o['seed']==s
        scenarios[f'leave_seed{s}_out']=lambda o,s=s:o['seed']!=s
        for m in (1,2): scenarios[f'model{m}_seed{s}']=lambda o,m=m,s=s:o['model']==m and o['seed']==s
    results={}
    for name,include in scenarios.items():
        cells={}; families={}
        for g in groups:
            selected=[r for r in rows if r['component_id']==g['component_id'] and r['entity_type']=='candidate']
            target=[float(np.mean([o['target'] for o in r['observations'] if include(o)])) for r in selected]
            cell=screening_metrics([r['entity_id'] for r in selected],[pred[r['entity_id']] for r in selected],target)
            cells[g['component_id']]=cell
            families.setdefault(g['family_stratum'],[]).append(cell['recall'])
        results[name]={'components':cells,'stratum_equal_recall':float(np.mean([np.mean(v) for v in families.values()])),
            'median_component_rho':float(np.median([v['spearman'] for v in cells.values() if v['spearman'] is not None]))}
    original=read(OUT/'evaluation.json')
    assert results['full6']['components']==original['components']
    assert results['full6']['stratum_equal_recall']==original['stratum_equal_recall']
    controls=[]
    for g in groups:
        pair={r['arm']:r for r in rows if r['component_id']==g['component_id'] and r['entity_type']=='control'}
        native,shuffle=pair['native'],pair['composition_shuffle']
        controls.append({'component':g['component_id'],
            'prediction_shuffle_minus_native':pred[shuffle['entity_id']]-pred[native['entity_id']],
            'teacher_shuffle_minus_native':shuffle['target']-native['target'],
            'interpretation':'composition-only model cannot distinguish sequence-order shuffle; not a binding-specificity result'})
    values=[v['stratum_equal_recall'] for k,v in results.items() if k!='full6']
    payload={'classification':'post_hoc_descriptive_not_confirmatory','scenarios':results,'controls':controls,
        'scenario_recall_range':[min(values),max(values)],'all_scenario_macro_recalls_above_random_0_2':all(v>.2 for v in values),
        'range_is_not_confidence_interval':True}
    freeze(OUT/'sensitivity.json',payload)
    lines=['# 公共回顾性PAE补测 v11：结果与稳定性', '',
        '完成时间：2026-09-19 13:36。768次候选生成、120条候选及12条对照、792次AF2评分全部完成。结果文件已核对哈希和完整六重复，主指标重新计算一致。', '',
        '## 冻结主结果', '',
        '保留每组20条候选中的4条，按4类抗原等权汇总，对AF2真实前20%候选的召回率为 **60.4%**，随机期望为 **20.0%**。候选排序相关性按6组取中位数为 **0.772**。60.4%/20%=3.02倍是描述性召回比，不是统计显著性、计算加速倍数或亲和力倍数。', '',
        '| 组件 | 结构 | 抗原类别 | 前20%召回率 | Spearmanρ |', '|---|---|---|---:|---:|']
    for g in groups:
        c=original['components'][g['component_id']]
        lines.append(f"| {g['component_id']} | {g['representative']['pdb_id']} | {g['family_stratum']} | {c['recall']:.1%} | {c['spearman']:.3f} |")
    lines += ['', 'CSP的3组先内部平均，再与其他3类等权。6组均高于本组20%的随机召回期望，但每组真值优选候选仅4条，召回率以25个百分点跳变，不能将单组百分比视为精确总体估计。', '',
        '## 事后教师敏感性', '',
        '保持Ridge预测和候选池不变，改变用于计算目标的AF2重复子集。所有结果均保留，没有据此改参数或选择最有利场景。', '',
        '| 教师子集 | 4类等权召回率 | 6组ρ中位数 |', '|---|---:|---:|']
    for name,v in results.items():
        lines.append(f"| {name} | {v['stratum_equal_recall']:.1%} | {v['median_component_rho']:.3f} |")
    lines += ['',f"各非完整教师场景的等权召回率范围为 **{min(values):.1%}–{max(values):.1%}**。这只是教师设置敏感性范围，不是置信区间；重复共享同一批候选，不能增加生物学独立样本量。", '',
        '## 对照和限制', '', '| 组件 | 打乱−原生：Ridge预测 | 打乱−原生：教师PAE |', '|---|---:|---:|']
    for c in controls:
        lines.append(f"| {c['component']} | {c['prediction_shuffle_minus_native']:.6f} | {c['teacher_shuffle_minus_native']:.4f} |")
    lines += ['', '归一化PAE越低越好。组成相同的打乱对照在Ridge中无法区分，这是模型结构限制；本结果不能支持序列顺序特异性的识别能力。', '',
        '- 主评分器仍是原Ridge，没有新方法胜过Ridge的证据。',
        '- 本轮是单一ProteinMPNN生成器、温度0.3的回顾性评估；该温度是在v10候选多样性失败后、未查看本轮标签前另行固定的。不能把结果推广至所有生成器和采样分布。',
        '- 已知CSP/融合肽家族、历史V3结构接触和上游训练不确定性均保留；不是严格独立抗原验证。',
        '- 目前衡量的是AF2置信度排序，不是实验结合、治疗效果或临床有效性。',
        '- 此次给全部候选运行AF2以建立真值，未做端到端加速对照。', '',
        '## 本阶段判断', '',
        '这批数据提供了“冻结简单模型在另一个公开回顾性候选池中具有筛选价值”的描述性支持；论文可使用此结果作为补充证据，不能改写为新算法优越性或临床证据。PUB003整体排序较弱，后续若研究失效原因应作为新开发实验单列，不能在本池调参后继续称为外部验证。', '',
        '产物：`evaluation.json`为冻结主评估；`verification.json`为完整性复核；`sensitivity_protocol.json`和`sensitivity.json`为事后诊断。均位于 `results/pae_public_pilot_v11/`。无新GPU任务。', '']
    (ROOT/'docs/PAE_PUBLIC_PILOT_V11_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'main_recall':original['stratum_equal_recall'],'sensitivity_range':payload['scenario_recall_range'],
        'all_above_random':payload['all_scenario_macro_recalls_above_random_0_2'],'controls':controls},flush=True)


if __name__=='__main__':main()
