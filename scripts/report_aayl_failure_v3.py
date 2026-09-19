"""Report confirmed findings separately from hypotheses; retain all ablations."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_aayl_failure_v3 import OUT,base,freeze


def main():
    report=json.loads((OUT/'report.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    fits=json.loads((OUT/'fits.json').read_text())['records']
    predictions=json.loads((OUT/'predictions.json').read_text())['records']
    splits=json.loads((OUT/'splits.json').read_text())['splits']
    panel=json.loads((OUT/'measurement_repair_panel.json').read_text())
    d,rows,changes=base.load()
    grouped=defaultdict(list); folded=defaultdict(list)
    for p in predictions:
        key=tuple(p[k] for k in ['family','task','representation','policy','control'])+(len(changes[p['index']]),)
        grouped[key].append(p); folded[key+(p['fold'],)].append(p)
        s=next(s for s in splits if all(s[k]==p[k] for k in ['family','task','fold']))
        assert p['index'] in s['test'] and p['index'] not in s['train']
        if p['control']=='real': assert p['target']==rows[p['index']]['endpoint']
    for r in report['metrics']:
        key=tuple(r[k] for k in ['family','task','representation','policy','control','degree'])
        ps=grouped[key]
        assert len(ps)==len({p['index'] for p in ps})
        m=base.metrics([p['target'] for p in ps],[p['prediction'] for p in ps])
        for k,v in m.items(): assert r[k] is None if v is None else np.isclose(r[k],v)
    foldmetrics=[]
    for key,ps in folded.items():
        if key[1]=='ood': continue
        foldmetrics.append(dict(zip(['family','task','representation','policy','control','degree','fold'],key)) |
          base.metrics([p['target'] for p in ps],[p['prediction'] for p in ps]))
    foldmeans=[]
    fg=defaultdict(list)
    for r in foldmetrics:
        fg[tuple(r[k] for k in ['family','task','representation','policy','control','degree'])].append(r)
    for key,rr in fg.items():
        vals=[r['spearman'] for r in rr if r['spearman'] is not None]
        foldmeans.append(dict(zip(['family','task','representation','policy','control','degree'],key)) |
            {'mean_fold_rho':float(np.mean(vals)),'min_fold_rho':min(vals),'max_fold_rho':max(vals),'n_folds':len(vals)})
    freeze(OUT/'foldwise_diagnostics.json',{'status':'posthoc_reporting_sensitivity_no_refitting_or_selection',
        'reason':'OOF models can have different intercepts/calibration; inspect within-fold, within-degree ranks too; do not select the nicer aggregation',
        'folds':foldmetrics,'summary':foldmeans})
    previous=json.loads((ROOT/'results/aayl_low_order_v2/report.json').read_text())['primary']
    decisions=[]
    for representation in ['position','esm']:
        cells=[]
        for family in d['structures']:
            r=next(r for r in report['metrics'] if r['family']==family and r['task']=='ood' and r['policy']=='shift_rank' and r['representation']==representation and r['control']=='real')
            b=next(b for b in previous if b['family']==family and b['model']=='position_ridge')
            cells.append({'family':family,'rho_gain':r['spearman']-b['spearman'],'recall_gain':r['top20_recall']-b['top20_recall']})
        decisions.append({'representation':representation,'primary_correction_passed':all(c['rho_gain']>0 and c['recall_gain']>=0 for c in cells),'cells':cells})
    freeze(OUT/'decision.json',{'decisions':decisions,'validated_affinity_ranker_available':False,
        'confirmed_repair':'expanded regularization reduces ESM squared error in both lineages; does not establish useful ranking',
        'not_promoted':['shift-aware validation as guaranteed performance fix','pairwise biological epistasis as proven cause','missing values as exact weak-binding labels'],
        'next_action':'source lookup and paired single/double measurement coverage; obtain independent new systems before method claims'})
    freeze(OUT/'verification.json',{'candidate_predictions_verified':len(predictions),'metric_cells_recomputed':len(report['metrics']),
        'model_searches':len(fits),'artifacts':{f:base.digest(OUT/f) for f in ['protocol.json','splits.json','audit.json','fits.json','predictions.json','report.json','foldwise_diagnostics.json','measurement_repair_panel.json','decision.json']}})
    lines=['# AAYL 排序失效：漏洞审计、修正与验证','','日期：2026-09-16。顺序开发诊断，数据已暴露；不能当作独立验证。','',
      '## 结论','','没有发现原始序列/端点对应、H3 编码或抽查的 ESM 行排列错误。确认的可修正问题是正则化搜索过窄；扩大搜索后两个谱系的 ESM 均方误差均下降。排序目标与选模目标、随机验证与跨阶任务不一致也是设计风险，但对照实测表明：改用排序选模、模拟跨阶验证不能保证恢复排序。','',
      '现有数据对三突变的具体突变对几乎无训练覆盖；同分布留出也仅见弱且不一致的信号。不能把失败归结为一个已经证实的生物学原因，更不能保证换模型便可修复。已执行计算修正与对照，实验补测尚未执行。','',
      '## 漏洞与原因：证据分级','','| 项目 | 核查结果 | 改正及执行结果 |','|---|---|---|',
      '| 数据/端点/链/H3 错位假设 | 1276 个候选原始序列及重复记录匹配；508 个端点重算通过；全体编码可还原原始突变 | 未发现该错误，不改写标签 |',
      '| ESM 缓存错行假设 | 4 个预设位置独立重算与缓存完全一致 | 权重、输入、缓存哈希已固定；这是抽查，非全量重算 |',
      '| 正则化搜索不足 | 旧 ESM 两谱系均选到 alpha=100 上界 | 扩展至 1e6；MSE 选模均选 1e4，测试误差下降 |',
      '| 排序任务却只用 MSE 选模 | 确认目标不一致，但不能直接证明为根因 | 同划分加入 Spearman 选模消融；无稳定有效排序 |',
      '| 随机 CV 未模拟跨阶 | 确认验证任务不匹配 | 训练内部低阶训练/最高阶验证；主要修正门槛未通过，不提升为默认方法 |',
      '| 单个替换覆盖不代表交互覆盖 | 99/102 与 95/95 三突变的三个具体突变对均未在训练双突变出现 | 生成亲本、单突变和双突变配套补测清单，尚无新增实验值 |',
      '| 原因只在跨阶数或标签随机噪声 | 不受支持：同分布信号也弱；完整子集重复一致性高 | 已补嵌套同分布 OOF、合成正对照、标签打乱负对照 |',
      '| 混合不同折 OOF 排序可能受校准影响 | 不同模型的均值偏移可影响拼接排序，尤其强收缩模型 | 保留原 OOF 汇总，并补逐折/逐阶平均，禁止选择较好口径掩盖弱结果 |','',
      '## 跨阶测试：所有预设消融','','两个谱系均用全部单＋双突变训练，同一三突变测试集。legacy_mse 在本轮统一的分层 CV 下重建，不是字节级复用旧模型；宽网格/MSE/排序消融之间使用相同候选划分。','',
      '| 谱系 | 表示 | 选模策略 | alpha | Spearman | top20 recall | MSE | 训练均值 MSE |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in report['metrics']:
        if r['task']=='ood' and r['control']=='real':
            f=next(f for f in fits if all(f[k]==r[k] for k in ['family','task','representation','policy','control']))
            lines.append(f"| {r['family']} | {r['representation']} | {r['policy']} | {f['alpha']:g} | {r['spearman']:.3f} | {r['top20_recall']:.3f} | {r['mse']:.3f} | {r['mean_control_mse']:.3f} |")
    lines+=['','wide_mse 只扩正则化；wide_rank 再改排序选模；shift_rank 再改为训练内低阶→最高阶验证。本轮预设主要修正是 shift_rank，不能事后改选外层结果最好的策略。其两种表示均未通过两个谱系同时改善的门槛。','',
      'ESM 的 MSE 从 1.314→0.995、3.893→2.745，有实际数值改善；相应训练均值基线为 0.974、2.774，故改善主要是减少过拟合，未证明获得实用预测能力。个别极强收缩配置接近常数预测，小幅正相关不等于可靠筛选。','',
      '## 同分布可学习性','','下表保留低阶训练池的五折嵌套留出结果，以阶数分组，展示平均折内 Spearman [最小, 最大]。这些折共享数据来源，范围不是置信区间。完整拼接 OOF 与混合阶数实验另存机器结果，不隐藏二者差异。','',
      '| 谱系 | 表示 | 策略 | 测试阶数 | 平均折内 ρ [范围] |','|---|---|---|---:|---|']
    for r in foldmeans:
        if r['task']=='low_oof' and r['control']=='real':
            lines.append(f"| {r['family']} | {r['representation']} | {r['policy']} | {r['degree']} | {r['mean_fold_rho']:.3f} [{r['min_fold_rho']:.3f}, {r['max_fold_rho']:.3f}] |")
    lines+=['','AAYL49 位置基线有一定低阶信号，例如 MSE 选模单/双突变平均折内相关约 0.147/0.258；不能概括为完全不可学习。但这一信号未在 AAYL51 和跨阶场景稳定复现。','',
      '## 正负对照','','相同序列设计赋予固定随机加性效应的合成正对照，跨阶相关为 0.931/0.865，说明流程能恢复这种已知可学习规律。该数值不是实测亲和力性能，也不能排除真实数据中的其他隐藏问题。一次分阶标签打乱只作程序 sanity check，不产生置换检验 p 值。','',
      '## 已生成的数据改正清单','','| 谱系 | 亲本锚点 | 单突变锚点 | 双突变锚点 | 合计构建体 | 已有不完整读数 |','|---|---:|---:|---:|---:|---:|']
    for s in panel['summary']:
        lines.append(f"| {s['family']} | {s['parent']} | {s['singles']} | {s['doubles']} | {s['proposed_constructs']} | {s['incomplete_existing']} |")
    lines+=['','清单包含重/轻链序列、H3、明确突变位置、关联三突变候选和优先级。它是当前全部三突变的完整配套覆盖方案，不是必须一次性订购的最小实验方案；两个谱系合计 748 个待核查/补测构建体。118 个存在不完整来源读数，应先复核/复测。其余先查更广来源，亲本控制也先核查原始控制表，不能断言数据库均不存在。','',
      '严格测量方案：同靶标、轻链、测量体系及批次对照，记录独立重复与无读数/检测限；每个拟研究三突变需配套亲本、三个单突变及三个双突变。如此才能检查加性和成对效应是否可识别。不要用缺失值冒充统一低亲和力，也不要用当前已暴露三突变声称新盲测。','',
      '## 产物、验证与边界','',
      f'已完成 {len(fits)} 次模型搜索；保存 {len(predictions)} 条逐候选预测并核对 {len(report["metrics"])} 个指标单元。结果、选参曲线、固定拆分、原始来源审计及补测清单位于 `results/aayl_failure_v3/`。', '',
      '复现：`python scripts/diagnose_aayl_failure_v3.py --prepare` → `--run` → `python scripts/build_aayl_repair_panel.py` → `python scripts/report_aayl_failure_v3.py`。旧版协议和结果未覆盖。报告中的折内汇总是事后解释敏感性，不触发重训练或选模。','',
      '计算修正已执行；湿实验和新来源标签尚未获得。目前没有经独立验证的亲和力排序新方法，不应将本轮修正包装成一区/二区方法成果。','',
      '## 方法依据','',
      '正则化强度和验证评分是不同选择，参见 [scikit-learn 官方 RidgeCV 文档](https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.RidgeCV.html)。原始 AlphaSeq 数据及测量体系见 [Scientific Data 原始论文](https://www.nature.com/articles/s41597-022-01779-4)。本报告的具体失效判断来自上述本地对照结果，而非由文献推定。','']
    path=ROOT/'docs/AAYL_FAILURE_DIAGNOSIS_AND_REPAIR.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)


if __name__=='__main__': main()
