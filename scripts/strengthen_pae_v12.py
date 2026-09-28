"""Post-hoc fixed baseline comparison; no selection or fitting on v11 labels."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from Bio.PDB import PDBParser
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.evaluate_mpnn_likelihood_baseline import run_mpnn, MPNN_ALPHABET

SOURCE = ROOT/'results/pae_public_pilot_v11'
OUT = ROOT/'results/pae_strengthening_v12'
AA = 'ACDEFGHIKLMNPQRSTVWY'


def recall(ids, scores, target, fraction):
    """Fixed top20% target, variable screening budget; integrate cutoff ties."""
    n = len(ids)
    k = int(np.ceil(n*fraction))
    q = int(np.ceil(n*.2))
    truth = set(sorted(range(n), key=lambda i:(target[i], ids[i]))[:q])
    chosen = sorted(range(n), key=lambda i:(scores[i], ids[i]))[:k]
    cutoff = sorted(scores)[k-1]
    below = {i for i,s in enumerate(scores) if s < cutoff}
    tied = {i for i,s in enumerate(scores) if s == cutoff}
    expected = (len(truth & below) + (k-len(below))*len(truth & tied)/len(tied))/q
    rho = float(spearmanr(scores, target).statistic) if np.ptp(scores)>1e-12 else None
    return dict(n=n,budget=k,target_count=q,recall=len(truth & set(chosen))/q,
                tie_averaged_recall=expected,random_expected_recall=k/n,spearman=rho,
                selected_ids=[ids[i] for i in chosen],cutoff_tie_count=len(tied))


def feature_matrix(groups, entities):
    features = {}
    for c in groups:
        assert digest(ROOT/c['canonical_pdb']) == c['canonical_sha256']
        m = next(PDBParser(QUIET=True).get_structure('s', ROOT/c['canonical_pdb']).get_models())
        h, ag = list(m['H']), list(m['P'])
        x = torch.tensor(np.array([h[i]['CA'].coord for i in c['representative']['h3_heavy_indices_zero_based']]))
        y = torch.tensor(np.array([a['CA'].coord for a in ag]))
        d = torch.cdist(x,y).min(1).values
        geom = [len(x),len(y),float(d.mean()),float(d.std(unbiased=False)),float(d.min()),float((d<8).float().mean())]
        for e in entities:
            if e['component_id'] != c['component_id']: continue
            comp = lambda seq:[seq.count(a)/len(seq) for a in AA]
            features[e['entity_id']] = comp(e['h3_sequence'])+comp(e['antigen_sequence'])+geom
    return np.array([features[e['entity_id']] for e in entities])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths = ['holdout.json','entities.json','selection.json','pre_teacher_predictions.json','teacher_labels.json']
    freeze(OUT/'protocol.json',{
        'classification':'post_hoc_descriptive_comparison_on_exposed_v11; not new validation',
        'sources':{p:digest(SOURCE/p) for p in paths},
        'training_sha256':digest(ROOT/'results/pae_surrogate_revision_v4/features.pt'),
        'script_sha256':digest(Path(__file__)),
        'budgets':[.1,.2,.3,.5], 'target':'fixed lowest20percent mean six-replicate normalized PAE',
        'ties':'entity_id operational tie break plus exact expected recall under random cutoff ties',
        'models':'frozen Ridge; original-train-only alpha10 standardized Ridge on composition40 or geometry6; no tuning',
        'heuristics':'lower mean Kyte-Doolittle hydropathy, lower absolute (K+R-D-E)/length, higher Shannon entropy; exploratory fixed directions, no label-based reversal',
        'mpnn':'saved first-source-attempt H3 NLL and global NLL; separate backbone-only conditional H3 NLL seed0 v_48_020; lower is better',
        'aggregation':'four strata equal after averaging components within stratum; no candidate-level CI',
        'boundary':'no binding claims; input geometry is constant within each scaffold',
    })
    groups = read(SOURCE/'holdout.json')['components']
    entities = [e for e in read(SOURCE/'entities.json')['entities'] if e['entity_type']=='candidate']
    ids = [e['entity_id'] for e in entities]
    x = feature_matrix(groups,entities)
    w = np.load(ROOT/'results/pae_screening_v7/ridge_fixed10.npz')
    frozen = ((x-w['mean'])/w['scale'])@w['coef']+w['intercept']
    reference = read(SOURCE/'pre_teacher_predictions.json')['predictions']
    np.testing.assert_allclose(frozen,[reference[i] for i in ids],atol=1e-7,rtol=0)
    scores = {'ridge_frozen':frozen,'ridge_h3_contribution':((x[:,:20]-w['mean'][:20])/w['scale'][:20])@w['coef'][:20]}
    train = torch.load(ROOT/'results/pae_surrogate_revision_v4/features.pt',weights_only=False)
    mask = np.array([r['split']=='train' for r in train['rows']])
    target = np.array([r['target'] for r in train['rows']],dtype=np.float32)
    for name, columns in [('composition_only',slice(0,40)),('geometry_only',slice(40,46))]:
        model = make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(train['simple'][mask,columns],target[mask])
        scores[name] = model.predict(x[:,columns])
        file = OUT/(name+'.npz')
        if not file.exists(): np.savez(file,mean=model[0].mean_,scale=model[0].scale_,coef=model[1].coef_,intercept=model[1].intercept_)
    kd = dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
    seqs = [e['h3_sequence'] for e in entities]
    scores['hydropathy_low'] = [np.mean([kd[a] for a in s]) for s in seqs]
    scores['absolute_charge_low'] = [abs(sum(s.count(a) for a in 'KR')-sum(s.count(a) for a in 'DE'))/len(s) for s in seqs]
    scores['entropy_high'] = [sum(p*np.log(p) for p in row[:20] if p>0) for row in x]
    sampling, global_nll, artifacts = [],[],{}
    for e in entities:
        c, seed, index = e['source_attempt_id'].split('|')
        folder = SOURCE/'generation_work'/c/seed/'mpnn'
        path = folder/'scores'/(c+'.npz')
        fasta = (folder/'seqs'/(c+'.fa')).read_text().splitlines()
        assert fasta[2*(int(index)+1)+1] == e['heavy_sequence']
        z = np.load(path)
        sampling.append(float(z['score'][int(index)])); global_nll.append(float(z['global_score'][int(index)]))
        artifacts[str(path.relative_to(ROOT))] = digest(path)
    scores['mpnn_sampling_h3_nll'] = sampling
    scores['mpnn_sampling_global_nll'] = global_nll
    conditional = {}
    for c in groups:
        r = c['representative']; h3 = r['h3_heavy_indices_zero_based']
        fixed = [i+1 for i in range(len(r['heavy_sequence'])) if i not in h3]
        work = OUT/'mpnn_backbone'/c['component_id']
        begin = time.perf_counter()
        run_mpnn(ROOT/c['canonical_pdb'],'H',fixed,work,seed=0)
        zpath = work/'conditional_probs_only'/(c['component_id']+'.npz')
        z = np.load(zpath)
        positions = np.flatnonzero(z['design_mask'])
        assert len(positions)==len(h3)
        assert ''.join(MPNN_ALPHABET[int(z['S'][i])] for i in positions)==r['cdr_h3_sequence']
        logp = z['log_p'].mean(axis=0)
        for e in entities:
            if e['component_id']==c['component_id']:
                conditional[e['entity_id']] = -float(np.mean([logp[i,MPNN_ALPHABET.index(a)] for i,a in zip(positions,e['h3_sequence'],strict=True)]))
        artifacts[str(zpath.relative_to(ROOT))]=digest(zpath)
        print('MPNN baseline',c['component_id'],round(time.perf_counter()-begin,2),flush=True)
    scores['mpnn_backbone_h3_nll'] = [conditional[i] for i in ids]
    freeze(OUT/'predictions.json',{'ids':ids,'scores':{k:np.asarray(v).tolist() for k,v in scores.items()},'artifacts':artifacts})
    labels = {r['entity_id']:r['target'] for r in read(SOURCE/'teacher_labels.json')['rows']}
    results = {}
    for name, values in scores.items():
        results[name] = {}
        for budget in (.1,.2,.3,.5):
            cells, strata = {},{}
            for c in groups:
                idx=[i for i,e in enumerate(entities) if e['component_id']==c['component_id']]
                cell=recall([ids[i] for i in idx],[float(values[i]) for i in idx],[labels[ids[i]] for i in idx],budget)
                cells[c['component_id']]=cell
                strata.setdefault(c['family_stratum'],[]).append(cell['tie_averaged_recall'])
            results[name][str(budget)]={'components':cells,'stratum_equal_tie_averaged_recall':float(np.mean([np.mean(v) for v in strata.values()]))}
    for budget in (.1,.2,.3,.5):
        for c in groups:
            cid=c['component_id']
            assert results['ridge_frozen'][str(budget)]['components'][cid]['selected_ids']==results['ridge_h3_contribution'][str(budget)]['components'][cid]['selected_ids']
    original=read(SOURCE/'evaluation.json')
    for c in groups:
        assert results['ridge_frozen']['0.2']['components'][c['component_id']]['recall']==original['components'][c['component_id']]['recall']
    freeze(OUT/'evaluation.json',{'classification':'post_hoc','results':results,'geometry_cannot_rank_within_scaffold':True})
    lines=['# PAE v12：补充基线比较','','已暴露的v11候选池，事后分析；无本池调参。目标始终是教师前20%，改变筛选预算。表中对分数相同的截止候选平均处理，避免ID碰巧打破平局造成虚假优势。','','| 方法 | 10%预算 | 20%预算 | 30%预算 | 50%预算 |','|---|---:|---:|---:|---:|']
    for name,data in results.items():
        lines.append('| '+name+' | '+' | '.join(f"{data[str(b)]['stratum_equal_tie_averaged_recall']:.1%}" for b in (.1,.2,.3,.5))+' |')
    lines+=['| 随机期望 | 10.0% | 20.0% | 30.0% | 50.0% |','','## 结构性发现','','冻结Ridge去除所有非H3组成项后，六组、四种预算下选择完全相同。几何和抗原组成在组内恒定，不能解释候选排序优势。独立重训的composition_only仅使用原240条训练记录，alpha=10不调参；并非本池训练。','','采样评分对应每条序列最初被选中的生成记录，不选重复记录的最优分数；backbone评分另行统一计算。两者均来自候选生成器，不能推广至其他生成器。简单理化指标方向固定，仅为探索性对照。','','所有分组结果、平局大小、实际选中ID见evaluation.json。不从120条候选构建伪独立置信区间。没有新增独立抗原证据。','']
    (ROOT/'docs/PAE_STRENGTHENING_V12.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v['0.2']['stratum_equal_tie_averaged_recall'] for k,v in results.items()},indent=2),flush=True)


if __name__=='__main__': main()
