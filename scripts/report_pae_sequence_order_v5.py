"""Verify and render PAE generator/order diagnostics without winner switching."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import pae_sequence_order_v5 as v


def main():
    report=v.read(v.OUT/'report.json'); predictions=v.read(v.OUT/'predictions.json'); fits=v.read(v.OUT/'fits.json')
    rows=[r for r in v.read(v.OLD/'data_manifest.json')['rows'] if r['split']=='transfer']
    assert predictions['entities']==[r['id'] for r in rows]
    checked=0
    for name,p in predictions['models'].items():
        generated={'all_entities':v.summarize(v.by_scaffold(rows,p,False)),
            'generated_only':v.summarize(v.by_scaffold(rows,p)), 'within_generator':v.within_arm(rows,p)}
        for group,cells in generated.items():
            for key,cell in cells['cells'].items():
                expected=report['models'][name][group]['cells'][key]
                for metric,value in cell.items():
                    assert expected[metric] is None if value is None else np.isclose(expected[metric],value)
                checked+=1
    # Representation sensitivity is necessary, but does not prove better ranking.
    controls=[]
    for scaffold in sorted({r['scaffold'] for r in rows}):
        a=next(r for r in rows if r['scaffold']==scaffold and r['arm']=='native')
        b=next(r for r in rows if r['scaffold']==scaffold and r['arm']=='composition_shuffle')
        assert sorted(a['h3_sequence'])==sorted(b['h3_sequence'])
        controls.append({'scaffold':scaffold,'position_feature_L2':float(np.linalg.norm(v.order_features(a['h3_sequence'])-v.order_features(b['h3_sequence']))),
            'bigram_feature_L2':float(np.linalg.norm(v.bigrams(a['h3_sequence'])-v.bigrams(b['h3_sequence'])))})
    v.freeze(v.OUT/'verification.json',{'metric_cells_recomputed':checked,'transfer_entities':len(rows),
        'generated_only_entities':sum(r['entity_type']=='candidate' for r in rows),
        'order_sensitivity':controls,'artifacts':{name:v.digest(v.OUT/name) for name in ['protocol.json','fits.json','predictions.json','report.json']}})
    def f(value): return '未定义' if value is None else f'{value:.3f}'
    lines=['# PAE 路线首轮：生成来源与序列顺序诊断','','日期：2026-09-17。已完成；旧外部开发数据，未产生新的独立确认性评估。','',
      '## 结论','','PAE 代理排序确有开发信号，但部分信号来自生成方法之间的差异。保留序列位置的主要新基线没有通过预设改进门槛。二肽特征的前列召回出现值得后续验证的迹象，但它是次要探索结果，不能替代失败的主要模型结论。现阶段保留 Ridge 作为筛选参照，不启用复杂模型优越性或亲和力主张。','',
      '## 设计与隔离','','沿用 v4 的 15 个训练组件/240 个实体、2 个校准组件/32 个实体和6个已暴露外部组件/84个实体。新模型只用原始设计骨架特征和序列；训练只看训练集，选参只看校准集，历史 GP2 不读取或重用。','',
      '主要评估排除 native 和 shuffle 人工对照，每个外部骨架剩12个真实生成候选，分别来自4种生成方法，每种3个。除骨架内整体排序外，单独比较骨架×生成方法内的排序。所有 PAE 分数均为越低越好。','',
      '三个新 Ridge 使用相同校准选参规则，网格为0.1至10000。position增加按H3两端对齐的32位独热编码；bigram增加相邻氨基酸对频率。主要候选事先固定为position。生成来源对照仅使用训练来源均值，不读取外部来源标签均值。','',
      '## 结果','','表中ρ为骨架内相关性中位数；候选召回是在每组12个中选3个并找回实测AF2最低PAE前三个，随机期望为0.25。组内正确率使用标签差大于合并一个SEM的候选对。这里的“可靠”只是计算噪声阈值，不是统计或实验真值保证。','',
      '| 方法 | 全14实体ρ | 仅生成候选ρ | 候选top20召回 | 同来源组内正确率 |','|---|---:|---:|---:|---:|']
    for name,m in report['models'].items():
        lines.append(f"| {name} | {f(m['all_entities']['median_rho'])} | {f(m['generated_only']['median_rho'])} | {f(m['generated_only']['macro_top20_recall'])} | {f(m['within_generator']['macro_reliable_pair_accuracy'])} |")
    lines+=['','组内共有24组、72个候选对；只有11组中的25对超过预设噪声阈值。原Ridge的1.000正确率只对应这25对，不能推广为普遍准确率。每组仅3条序列，Spearman取值非常粗，不适宜据此作强统计主张。','',
      '## 生成来源是否解释全部信号','','仅看生成来源的对照在生成候选上的ρ中位数为0.561，而原Ridge为0.829。这个比较及组内置换均提示Ridge还有同来源内排序信息，不能说已有信号全是来源混杂。','',
      '| 方法 | 实际平均骨架ρ | 保留来源、打乱组内对应后的平均ρ | 置换分布95%范围 |','|---|---:|---:|---|']
    for name,p in report['within_arm_permutation'].items():
        if p['actual_mean_scaffold_rho'] is None: continue
        interval=p['shuffle_95_percentile_range']
        lines.append(f"| {name} | {p['actual_mean_scaffold_rho']:.3f} | {p['within_arm_shuffle_mean_rho']:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] |")
    lines+=['','每种方法进行2000次组内置换；这是已暴露数据的描述性诊断，不是新盲测的显著性证明，也不能将相关系数差异直接转换为“来源贡献百分比”。','',
      '## 顺序修正是否有效','','位置表示已能区分全部六组原生/同组成打乱对照，但“能区分输入”不等于“预测更准”。位置模型候选ρ为0.720，低于相同选参条件下的组成模型0.769；组内正确率和召回也未改善，预设门槛未通过。','',
      '二肽模型候选召回为0.889，组成模型为0.778、原固定alpha Ridge为0.722；但二肽模型ρ为0.801，仍低于原Ridge的0.829。二肽表示亦不是完整的序列顺序编码，应作为固定的次要候选在新数据验证，不能事后更改主要假设。','',
      '三个新模型均选择alpha=10000，位于本轮网格上界。只有两个校准组件，选参稳定性有限；后续可在训练组件内部建立分组验证并预先扩展网格，不能继续依据这六个外部组件追逐最优结果。','',
      '## 下一步及停止条件','','1. 建立候选来源均衡、更密集的开发面板，建议每骨架每生成方法至少20条唯一候选；这是工作量建议，尚未生成或跑AF2。记录筛选与失败，避免只保留成功结构造成偏差。',
      '2. 在获取新标签前固定表示、选参规则、筛选预算及成功条件；保留原Ridge、组成、位置与二肽对照。旧骨架新增序列仍是开发数据，不能当作新增独立抗原。',
      '3. 继续独立抗原家族队列的来源及同源隔离审计；当前新增独立队列仍为0，GP2不重新启封。独立样本无法获得时，应维持工具/基准定位而不是声称泛化新方法。',
      '4. 只有筛选效果在更大面板稳定后，才启动同硬件完整AF2与预筛流程计时。此轮AF2新增任务为0，没有实测新端到端加速，也没有湿实验结合结论。','',
      '## 复现与核查','',
      f'运行 `python scripts/pae_sequence_order_v5.py --prepare`、`--run`，再运行 `python scripts/report_pae_sequence_order_v5.py`。已核对{checked}个指标单元，重建原Ridge预测一致；产物位于 `results/pae_sequence_order_v5/`。','',
      '本轮改变研究优先级，不覆盖ECLS冻结稿、旧PAE结果或发布包。','']
    path=ROOT/'docs/PAE_SEQUENCE_ORDER_V5_RESULTS.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)


if __name__=='__main__': main()
