"""Frozen development diagnostics and controlled model-selection repairs.

No new biological claims or independent-test status; old results are retained.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import benchmark_aayl_learnability as base
from scripts.build_aayl_original_endpoint_cohort import freeze
from scripts.crosswalk_abbibench_aayl_endpoints import original_rows, finite_value

OUT=ROOT/'results/aayl_failure_v3'
SEED=20260921
POLICIES=['legacy_mse','wide_mse','wide_rank','shift_rank']
ALPHAS=[.01,.1,1.,10.,100.,1000.,10000.,100000.,1000000.]


def rho(y,p):
    return float(spearmanr(y,p).statistic) if len(np.unique(y))>1 and len(np.unique(p))>1 else 0.


def features(d,rows,family):
    parent=d['structures'][family]['cdr_h3_sequence']
    x=np.zeros((len(rows),len(parent)*20))
    for i,r in enumerate(rows):
        if r['family']!=family: continue
        assert len(r['h3_sequence'])==len(parent)
        for j,(a,b) in enumerate(zip(r['h3_sequence'],parent)):
            x[i,j*20+base.AA.index(a)]+=1
            x[i,j*20+base.AA.index(b)]-=1
    return x


def inner_splits(degrees,shift=False):
    degrees=np.asarray(degrees)
    if shift:
        top=degrees.max()
        a,b=np.flatnonzero(degrees<top),np.flatnonzero(degrees==top)
        if len(a)<5 or len(b)<5: raise ValueError('Insufficient lower-to-higher validation data')
        return [(a,b)]
    return list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(degrees)),degrees))


def fit_model(x_train,y_train,degrees,x_test,representation,policy):
    model=make_pipeline(StandardScaler(),Ridge()) if representation=='esm' else make_pipeline(Ridge())
    alphas=ALPHAS[:5] if policy=='legacy_mse' else ALPHAS
    cv=inner_splits(degrees,policy=='shift_rank')
    search=GridSearchCV(model,{'ridge__alpha':alphas},cv=cv,
        scoring='neg_mean_squared_error' if policy.endswith('mse') else make_scorer(rho),
        n_jobs=1,error_score='raise')
    search.fit(x_train,y_train)
    return search.predict(x_test), {'alpha':search.best_params_['ridge__alpha'],
        'validation_score':float(search.best_score_), 'at_upper_boundary':search.best_params_['ridge__alpha']==alphas[-1],
        'curve':[{'alpha':float(a),'mean':float(m)} for a,m in zip(search.cv_results_['param_ridge__alpha'],search.cv_results_['mean_test_score'])]}


def prepare():
    d,rows,changes=base.load()
    inputs=[base.COHORT,base.SCORES,base.OUT/'esm2_h3_embeddings.npz',base.OUT/'embedding_receipt.json',
            ROOT/'results/aayl_low_order_v2/report.json',Path(__file__),Path(base.__file__)]
    freeze(OUT/'protocol.json',{'classification':'post_failure_exposed_development_diagnostics_not_confirmatory',
        'inputs':{str(p.resolve().relative_to(ROOT)):base.digest(p) for p in inputs},
        'seed':SEED,'representations':['position','esm'],'policies':POLICIES,'alphas':ALPHAS,
        'tasks':['low_order_5fold_nested_OOF','mixed_degree_5fold_nested_OOF','single_double_to_triple'],
        'selection':'5-fold degree-stratified within training; shift_rank trains lower degree and validates highest TRAINING degree only; outer triples excluded from OOD selection',
        'primary_correction':'shift_rank; other policies isolate regularization and scoring changes; do not choose best policy from final triples',
        'metrics':'Spearman/top20/pair accuracy per degree for OOF; MSE vs training-mean control; no degree-pooled Spearman for mixed tasks',
        'audits':['all cohort raw source sequence/replicate/endpoint match','H3 mapping and feature roundtrip','four ESM row sentinels','degree endpoint and exact/pairwise substitution coverage','label-shuffle negative control and additive synthetic positive control'],
        'controls':'position wide_rank only; synthetic effects frozen RNG; shuffle labels within degree and lineage once, descriptive sanity check NOT permutation significance',
        'missing_data':'retain prior missingness audit; do not replace missing labels or treat absence as fixed Kd',
        'success_rule':'primary correction improves full OOD Spearman and does not lower top20 recall versus v2 position Ridge in BOTH lineages; complex/new method claim still requires external data',
        'caveat':'This is sequential development on exposed data; neither nested CV nor a new seed restores independent confirmation'})
    splits=[]
    for family in sorted(d['structures']):
        low=[i for i,r in enumerate(rows) if r['family']==family and len(changes[i])<=2]
        triple=[i for i,r in enumerate(rows) if r['family']==family and len(changes[i])==3]
        splits.append({'family':family,'task':'ood','fold':0,'train':low,'test':triple})
        for task,pool in [('low_oof',low),('mixed_oof',sorted(low+triple))]:
            deg=[len(changes[i]) for i in pool]
            for fold,(a,b) in enumerate(inner_splits(deg)):
                splits.append({'family':family,'task':task,'fold':fold,'train':[pool[i] for i in a],'test':[pool[i] for i in b]})
    freeze(OUT/'splits.json',{'rows':[r['poi'] for r in rows],'splits':splits})
    print('Frozen 22 outer splits and diagnostic controls',flush=True)


def raw_audit(d,rows):
    source=ROOT/'data/abbibench_sequence_mapping_v1/original_alpha_seq/MITLL_AAlphaBio_Ab_Binding_dataset.csv.zip'
    source_protocol=json.loads((base.COHORT.parent/'protocol.json').read_text())
    assert base.digest(source)==source_protocol['source_sha256']
    all_rows=rows+d['excluded']
    index={(r['poi'],r['assay']):r for r in all_rows}
    assert len(index)==len(all_rows)
    actual=defaultdict(list)
    for raw in original_rows(source):
        key=(raw['POI'],raw['Assay'])
        if raw['Target']=='MIT_Target' and key in index:
            r=index[key]
            assert raw['HC']==r['heavy_sequence'] and raw['LC']==r['light_sequence']
            actual[key].append({'replicate':raw['Replicate'],'original_log10_nM':finite_value(raw['Pred_affinity'])})
    for key,r in index.items():
        assert sorted(actual[key],key=lambda x:x['replicate'])==sorted(r['observations'],key=lambda x:x['replicate'])
        if 'endpoint' in r:
            assert np.isclose(r['endpoint'],9-np.median([o['original_log10_nM'] for o in actual[key]]))
    return {'candidate_groups_matched':len(index),'eligible_endpoints_matched':len(rows),'source_sha256':base.digest(source)}


def sentinel_audit(d,rows,emb):
    import torch,esm
    torch.serialization.add_safe_globals([argparse.Namespace])
    torch.set_num_threads(2)
    p=json.loads((base.OUT/'protocol.json').read_text())
    assert base.digest(Path(p['embedding_weights']))==p['embedding_weights_sha256']
    model,alphabet=esm.pretrained.load_model_and_alphabet_local(p['embedding_weights'])
    model.eval().cpu()
    converter=alphabet.get_batch_converter()
    idx=[0,100,300,len(rows)-1]
    errors=[]
    for i in idx:
        _,_,tokens=converter([(rows[i]['poi'],rows[i]['heavy_sequence'])])
        with torch.inference_mode(): rep=model(tokens,repr_layers=[6],return_contacts=False)['representations'][6]
        h3=np.array(d['structures'][rows[i]['family']]['h3_indices_zero_based_observed_heavy'])+1
        v=rep[0,h3].mean(0).numpy()
        err=float(np.max(np.abs(v-emb[i])))
        assert np.allclose(v,emb[i],atol=2e-5,rtol=2e-5)
        errors.append({'index':i,'max_abs_error':err})
    return errors


def coverage_and_distribution(d,rows,changes):
    from itertools import combinations
    records=[]
    for family,s in d['structures'].items():
        parent=s['sequences'][s['heavy_chain']]
        assert ''.join(parent[i] for i in s['h3_indices_zero_based_observed_heavy'])==s['cdr_h3_sequence']
        low=[i for i,r in enumerate(rows) if r['family']==family and len(changes[i])<=2]
        seen=Counter(m for i in low for m in changes[i])
        pairs=set(p for i in low for p in combinations(sorted(changes[i]),2))
        triples=[i for i,r in enumerate(rows) if r['family']==family and len(changes[i])==3]
        records.append({'family':family,'per_degree':[{'degree':deg,'n':len(v:=[r['endpoint'] for i,r in enumerate(rows) if r['family']==family and len(changes[i])==deg]),
             'endpoint_quantiles':np.quantile(v,[0,.25,.5,.75,1]).tolist()} for deg in [1,2,3]],
             'triple_substitution_min_training_count_histogram':dict(Counter(min(seen[m] for m in changes[i]) for i in triples)),
             'triple_pair_coverage_histogram':dict(Counter(sum(p in pairs for p in combinations(sorted(changes[i]),2)) for i in triples))})
    return records


def run():
    threadpool_limits(2)
    protocol=json.loads((OUT/'protocol.json').read_text())
    for p,h in protocol['inputs'].items(): assert base.digest(ROOT/p)==h
    d,rows,changes=base.load()
    e=np.load(base.OUT/'esm2_h3_embeddings.npz')['features']
    audit={'raw_source':raw_audit(d,rows),'embedding_sentinels':sentinel_audit(d,rows,e),
           'coverage_distribution':coverage_and_distribution(d,rows,changes)}
    freeze(OUT/'audit.json',audit)
    print('Source, feature mapping and ESM alignment audits passed',flush=True)
    y=np.array([r['endpoint'] for r in rows])
    deg=np.array([len(m) for m in changes])
    predictions=[]; fits=[]
    splits=json.loads((OUT/'splits.json').read_text())['splits']
    x_by_family={f:features(d,rows,f) for f in d['structures']}
    rng=np.random.default_rng(SEED)
    shuffled=y.copy()
    synthetic=np.zeros(len(y))
    for family,x in x_by_family.items():
        idx=np.array([i for i,r in enumerate(rows) if r['family']==family])
        synthetic[idx]=(x@rng.normal(size=x.shape[1]))[idx]
        for k in [1,2,3]:
            sub=idx[deg[idx]==k]; shuffled[sub]=rng.permutation(y[sub])
    for s in splits:
        tr,te=np.array(s['train']),np.array(s['test'])
        identity={k:s[k] for k in ['family','task','fold']}
        models=[(rep,policy,'real') for rep in ['position','esm'] for policy in POLICIES]
        models += [('position','wide_rank',control) for control in ['shuffled','synthetic']]
        for rep,policy,control in models:
            x=e if rep=='esm' else x_by_family[s['family']]
            target={'real':y,'shuffled':shuffled,'synthetic':synthetic}[control]
            p,details=fit_model(x[tr],target[tr],deg[tr],x[te],rep,policy)
            fits.append({**identity,'representation':rep,'policy':policy,'control':control,**details})
            for i,v in zip(te,p):
                predictions.append({**identity,'representation':rep,'policy':policy,'control':control,'index':int(i),
                    'prediction':float(v),'target':float(target[i]),'train_mean':float(target[tr].mean())})
        print(f"Completed {s['family']} {s['task']} fold={s['fold']}",flush=True)
    groups=defaultdict(list)
    for p in predictions:
        groups[(p['family'],p['task'],p['representation'],p['policy'],p['control'],int(deg[p['index']]))].append(p)
    records=[]
    for key,ps in groups.items():
        assert len({p['index'] for p in ps})==len(ps)
        yy=np.array([p['target'] for p in ps]); pp=np.array([p['prediction'] for p in ps]); cc=np.array([p['train_mean'] for p in ps])
        records.append(dict(zip(['family','task','representation','policy','control','degree'],key)) |
            base.metrics(yy,pp) | {'mse':float(np.mean((yy-pp)**2)),'mean_control_mse':float(np.mean((yy-cc)**2))})
    freeze(OUT/'predictions.json',{'records':predictions})
    freeze(OUT/'fits.json',{'records':fits})
    freeze(OUT/'report.json',{'protocol_sha256':base.digest(OUT/'protocol.json'),'metrics':records,
        'note':'OOF per-degree pooled predictions from different fits; not independent folds or confidence intervals'})
    print('Done; all prespecified policies and sanity controls retained',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true'); parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.run: run()
