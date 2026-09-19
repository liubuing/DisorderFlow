"""Verify saved candidate predictions and write the complete v2 development report."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_aayl_low_order import OUT, base, freeze


def main():
    d,rows,changes=base.load()
    report=json.loads((OUT/'report.json').read_text())
    splits=json.loads((OUT/'splits.json').read_text())['splits']
    split_map={(s['family'],s['budget'],s['seed']):s for s in splits}
    preds=json.loads((OUT/'predictions.json').read_text())['records']
    groups={}
    for p in preds:
        key=(p['family'],p['budget'],p['seed'],p['model'])
        g=groups.setdefault(key,{})
        assert p['index'] not in g
        g[p['index']]=p['prediction']
    for r in report['metrics']:
        s=split_map[(r['family'],r['budget'],r['seed'])]
        g=groups[(r['family'],r['budget'],r['seed'],r['model'])]
        assert set(g)==set(s['test'])
        if r['stratum']=='nonoverlapping_replicate_ranges':
            correct=[]
            # Independent loop implementation, rather than the vectorized scorer.
            for j,i in enumerate(s['test']):
                a=[o['original_log10_nM'] for o in rows[i]['observations']]
                for k in s['test'][j+1:]:
                    b=[o['original_log10_nM'] for o in rows[k]['observations']]
                    if max(a)<min(b) or max(b)<min(a):
                        truth=np.sign(rows[i]['endpoint']-rows[k]['endpoint'])
                        prediction=np.sign(g[i]-g[k])
                        correct.append(.5 if prediction==0 else float(prediction==truth))
            assert len(correct)==r['n_pairs']
            assert np.isclose(np.mean(correct),r['pair_accuracy'])
        else:
            idx=[i for i in s['test'] if r['stratum']=='all' or ((i in s['covered'])==(r['stratum']=='covered'))]
            recalculated=base.metrics([rows[i]['endpoint'] for i in idx],[g[i] for i in idx])
            for key,v in recalculated.items():
                assert r[key] is None if v is None else np.isclose(r[key],v)
    freeze(OUT/'verification.json',{'metric_cells_verified':len(report['metrics']),'prediction_records':len(preds),
           'artifacts':{f:base.digest(OUT/f) for f in ['protocol.json','splits.json','predictions.json','fits.json','report.json']},
           'verifier_sha256':base.digest(Path(__file__))})
    lines=['# 低阶突变训练到三突变排序：第二轮开发结果','','日期：2026-09-16。此前已暴露的 AAYL 开发数据；不是独立盲测。','',
      '## 结论与决策','','加入双突变后，可用训练量和替换覆盖明显增加，但三种监督基线均未获得稳定的三突变排序优势。ESM2 相比 Ridge 的预设投入门槛再次未通过。现阶段不进入昂贵结构融合、大模型训练或投稿包装。', '',
      '完整读数内重复测量排序一致性较高，不能简单把模型失败解释为标签随机噪声。另一方面，完整读数筛选损失大量候选，因此结果只适用于完整读数子集；仍不能证明失效来源必然是非加性效应。','',
      '## 固定设计','','每个谱系独立：单＋双突变训练，全部三突变测试。预算 20/50/100 各五个固定种子，全量只运行一次；按训练池单/双比例抽样。同一划分比较位置 Ridge、RBF 核 Ridge、冻结 ESM2 t6 8M＋Ridge 和三种零样本分数。监督模型的信息条件高于零样本对照。', '',
      '端点为 9 − median(三重复原始 log10 estimated Kd[nM])，越高越好；不是直接物理结合测量。沿用第一轮模型网格，所有参数选择、中心化和特征缩放仅在训练内五折进行。测试标签不传入拟合函数。覆盖要求每个具体位点＋替换氨基酸均在训练出现，不等于该替换组合已经出现。', '',
      'ESM 投入门槛：两个谱系全预算 Spearman 均严格优于位置 Ridge，且 top20 recall 均不下降。该条件为预设资源决策规则，不是统计显著性。', '',
      '## 主要结果：全部训练预算','','| 谱系 | 方法 | Spearman | top20 recall | 富集倍数 |','|---|---|---:|---:|---:|']
    for r in report['primary']:
        lines.append(f"| {r['family']} | {r['model']} | {r['spearman']:.3f} | {r['top20_recall']:.3f} | {r['enrichment']:.2f} |")
    lines+=['','top20 使用 ceil(0.2×n) 的筛选数量，并列边界按分数权重处理。前列富集不等于整体排序可靠，也未证明监督模型优于原有简单分数。','',
       '## 替换覆盖与上一轮比较','','| 谱系 | 本轮训练量 | 三突变测试量 | 单突变训练时覆盖 | 本轮覆盖 |','|---|---:|---:|---:|---:|']
    old=json.loads((base.OUT/'splits.json').read_text())['splits']
    for s in splits:
        if s['budget']=='all':
            previous=next(o for o in old if o['family']==s['family'] and o['budget']=='all')
            prev=sum(len(changes[i])==3 for i in previous['covered'])
            lines.append(f"| {s['family']} | {len(s['train'])} | {len(s['test'])} | {prev} | {len(s['covered'])} |")
    lines+=['','| 谱系 | 分层 | n | 方法 | Spearman | top20 recall |','|---|---|---:|---|---:|---:|']
    for r in report['metrics']:
        if r['budget']=='all' and r['stratum'] in ['covered','uncovered']:
            lines.append(f"| {r['family']} | {r['stratum']} | {r['n']} | {r['model']} | {r['spearman']:.3f} | {r['top20_recall']:.3f} |")
    lines+=['','上轮与本轮全量训练的标签数不同，不能把差异归因于训练阶数本身。同预算曲线用于描述，不能事后挑选最佳种子或覆盖子集作为主要结果。','',
      '## 学习曲线','','Spearman 为五个子集的均值 [最小, 最大]，不是置信区间。','',
      '| 谱系 | 预算 | 方法 | Spearman 均值 [范围] |','|---|---|---|---|']
    for family in sorted(d['structures']):
        for budget in [20,50,100]:
            for model in ['position_ridge','position_rbf','esm2_ridge']:
                v=[r['spearman'] for r in report['metrics'] if r['family']==family and r['budget']==budget and r['model']==model and r['stratum']=='all']
                lines.append(f'| {family} | {budget} | {model} | {np.mean(v):.3f} [{min(v):.3f}, {max(v):.3f}] |')
    lines+=['','## 重复测量与无读数审计','','| 谱系 | 突变数 | 完整/全部候选 | 完整率 | 三对重复 Spearman 范围 | 重复极差中位数(log10 nM) |','|---|---:|---:|---:|---:|---:|']
    for a in report['label_audit']:
        v=list(a['replicate_spearman'].values())
        lines.append(f"| {a['family']} | {a['degree']} | {a['included']}/{a['included']+a['excluded']} | {a['completion_fraction']:.1%} | {min(v):.3f}–{max(v):.3f} | {a['median_log10_range']:.3f} |")
    lines+=['','上述一致性仅针对完整候选，不能外推至无读数候选，也不能排除共享系统误差。无读数没有填充为精确亲和力或统一检测上限。缺失随阶数的比例变化说明完整读数条件改变了评估人群，但本诊断不单独识别缺失的因果机制。','',
      '为检查结果是否主要由近似并列标签造成，另评估三重复观测范围互不重叠的候选对。观测范围不是置信区间，这些成对样本共享候选，不能当作独立样本做显著性推断。','',
      '| 谱系 | 方法 | 范围不重叠对数/全部对数 | 排序正确率 |','|---|---|---:|---:|']
    for r in report['metrics']:
        if r['budget']=='all' and r['stratum']=='nonoverlapping_replicate_ranges':
            lines.append(f"| {r['family']} | {r['model']} | {r['n_pairs']}/{r['total_pairs']} | {r['pair_accuracy']:.3f} |")
    lines+=['','## 下一步的研究判断','','当前证据不支持仅靠增加双突变训练量或本轮 ESM 表示改善排序。优先做数据与任务诊断：预先固定低阶池内的候选留出，区分同分布可学习性与跨阶推广失败；检查训练/测试端点分布和被筛掉候选的观测过程；随后补充其他实验系统。上述工作尚未执行。','',
      '只有同分布可学习、跨阶失效的证据得到支持后，才值得检验显式突变交互或结构特征的增量贡献。可靠性加权亦需同预算消融证明；高重复一致性不支持预先把它作为能解决问题的结论。任何新路线仍须多个靶标与独立验证支持，当前没有一区/二区接收证据。','',
      '## 复现与验证','','运行 `python scripts/benchmark_aayl_low_order.py --prepare`，然后 `--run`；最后 `python scripts/report_aayl_low_order.py`。协议保护输入、缓存表示、上一轮结果和脚本哈希，保留旧版结果。','',
      f'本轮 32 个训练划分、96 次监督模型网格搜索；保存 {len(preds)} 条候选预测，重算核对 {len(report["metrics"])} 个指标单元。包括拆分无泄漏、预算、替换覆盖、测试特征不影响训练缩放/选参、缺失计数闭合的测试。','']
    path=ROOT/'docs/AAYL_LOW_ORDER_STAGE2_RESULTS.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)


if __name__=='__main__': main()
