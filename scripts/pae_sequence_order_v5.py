"""Pre-AF2 exposed-development audit of generator confounding and sequence order.

Only v4 retained rows are read; historical GP2 is not loaded or evaluated.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze
from scripts.benchmark_aayl_learnability import digest,top_weights

OLD=ROOT/'results/pae_surrogate_revision_v4'
OUT=ROOT/'results/pae_sequence_order_v5'
AA='ACDEFGHIKLMNPQRSTVWY'
ALPHAS=[.1,1.,10.,100.,1000.,10000.]


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def correlation(y,p):
    return float(spearmanr(y,p).statistic) if len(y)>=3 and np.ptp(y)>0 and np.ptp(p)>1e-12 else None


def metrics(rows,p):
    y=np.array([r['target'] for r in rows]); p=np.array(p)
    n=len(y); k=int(np.ceil(.2*n))
    sem=np.array([r['sem'] for r in rows]); i,j=np.triu_indices(n,1)
    delta=y[i]-y[j]; dp=p[i]-p[j]
    meaningful=np.abs(delta)>np.hypot(sem[i],sem[j])
    scores=(np.sign(delta)==np.sign(dp)).astype(float); scores[np.abs(dp)<1e-12]=.5
    return {'n':n,'spearman':correlation(y,p),'mse':float(np.mean((p-y)**2)),
        'top20_recall':float(top_weights(-p,k)@top_weights(-y,k)/k),'selected':k,'random_recall':k/n,
        'all_pairs':len(i),'reliable_pairs':int(meaningful.sum()),
        'reliable_pair_accuracy':float(scores[meaningful].mean()) if meaningful.any() else None}


def order_features(sequence,slots=32):
    if not 1<=len(sequence)<=slots: raise ValueError('H3 length outside frozen position range')
    out=np.zeros((2,slots,20))
    for i,a in enumerate(sequence):
        out[0,i,AA.index(a)]=1
        out[1,slots-len(sequence)+i,AA.index(a)]=1
    return out.ravel()


def bigrams(sequence):
    out=np.zeros((20,20))
    for a,b in zip(sequence,sequence[1:]): out[AA.index(a),AA.index(b)]+=1/max(1,len(sequence)-1)
    return out.ravel()


def by_scaffold(rows,p,candidates_only=True):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if not candidates_only or r['entity_type']=='candidate': groups[r['scaffold']].append(i)
    return {s:metrics([rows[i] for i in ids],np.asarray(p)[ids]) for s,ids in sorted(groups.items())}


def summarize(cells):
    rhos=[m['spearman'] for m in cells.values() if m['spearman'] is not None]
    acc=[m['reliable_pair_accuracy'] for m in cells.values() if m['reliable_pair_accuracy'] is not None]
    return {'groups':len(cells),'defined_rho':len(rhos),'median_rho':float(np.median(rhos)) if rhos else None,
        'macro_top20_recall':float(np.mean([m['top20_recall'] for m in cells.values()])),
        'macro_reliable_pair_accuracy':float(np.mean(acc)) if acc else None,'cells':cells}


def within_arm(rows,p):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['entity_type']=='candidate': groups[(r['scaffold'],r['arm'])].append(i)
    cells={s+'|'+a:metrics([rows[i] for i in ids],np.asarray(p)[ids]) for (s,a),ids in sorted(groups.items())}
    return summarize(cells)


def prepare():
    sources=[OLD/'features.pt',OLD/'data_manifest.json',OLD/'predictions.json',OLD/'protocol.json',Path(__file__)]
    freeze(OUT/'protocol.json',{'classification':'previously_exposed_development_not_new_confirmation',
        'inputs':{str(p.resolve().relative_to(ROOT)):digest(p) for p in sources},
        'input_contract':'design-backbone simple features and sequence only; teacher target/SEM not input; v4 excluded GP2 remains excluded',
        'splits':'unchanged 15 train / 2 calibration / 6 exposed transfer; fit train only, select calibration only',
        'models':['v4_ridge_fixed10_rebuilt','composition_tuned','composition_position_tuned','composition_bigram_tuned','generator_mean_train_only','backbone_context_only'],
        'features':'composition_tuned uses v4 46 simple features; position adds dual-end 32x20 one-hot; bigram adds normalized 20x20 adjacent counts; no encoder pretraining in these new Ridge baselines',
        'alphas':ALPHAS,'selection':'max mean per-scaffold Spearman on calibration generated candidates, undefined=0; tie lower MSE then larger alpha; train-only scaling',
        'primary':'generated candidates only: per-scaffold and within scaffold+generator; exclude native/shuffle controls from primary evaluation',
        'secondary':'v4 full 14-candidate comparison; prediction permutation within scaffold+generator 2000 times, preserves generator distribution; descriptive not new-test significance',
        'main_candidate':'composition_position_tuned; bigram exploratory secondary; no external winner switching',
        'gate':'position median candidate rho > tuned composition AND macro within-arm reliable pair accuracy higher, with no lower candidate top20 recall; resource gate only',
        'limits':'3 transfer candidates per generator per scaffold, 6 scaffolds; ranks coarse; SEM teacher repeats are not independent biological units',
        'random_seed':20260917,'new_AF2_jobs':0})
    print('v5 protocol frozen before fitting',flush=True)


def choose(x,rows,train,cal):
    y=np.array([r['target'] for r in rows],dtype=np.float32)
    candidates=[]
    for alpha in ALPHAS:
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(x[train],y[train])
        pred=model.predict(x[cal]); cells=by_scaffold([rows[i] for i in cal],pred)
        score=float(np.mean([m['spearman'] if m['spearman'] is not None else 0 for m in cells.values()]))
        mse=float(np.mean((pred-y[cal])**2))
        candidates.append((score,-mse,alpha,model))
    best=max(candidates,key=lambda z:z[:3])
    return best[3],{'alpha':best[2],'calibration_rank_score':best[0],
       'grid':[{'alpha':a,'rank_score':s,'mse':-m} for s,m,a,_ in candidates]}


def permutation_diagnostic(rows,p,repeats=2000):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['entity_type']=='candidate': groups[(r['scaffold'],r['arm'])].append(i)
    by=defaultdict(list)
    for (s,a),ids in groups.items(): by[s].extend(ids)
    rng=np.random.default_rng(20260917)
    # Pre-rank y and p within scaffold: subgroup permutations preserve ranks/ties.
    from scipy.stats import rankdata
    null=np.zeros(repeats); actual=[]
    for s,ids in by.items():
        y=np.array([rows[i]['target'] for i in ids]); pr=np.array(p)[ids]
        yr=rankdata(y); yr-=yr.mean(); pr=rankdata(pr); pr-=pr.mean()
        denominator=np.linalg.norm(yr)*np.linalg.norm(pr)
        if denominator==0: continue
        actual.append(float(yr@pr/denominator))
        local={old:new for new,old in enumerate(ids)}
        samples=np.tile(pr,(repeats,1))
        for (ss,a),members in groups.items():
            if ss==s:
                jj=np.array([local[i] for i in members])
                for k in range(repeats): samples[k,jj]=rng.permutation(pr[jj])
        null+=samples@yr/denominator
    if not actual: return {'actual_mean_scaffold_rho':None}
    null/=len(actual)
    return {'actual_mean_scaffold_rho':float(np.mean(actual)),
        'within_arm_shuffle_mean_rho':float(null.mean()),'shuffle_95_percentile_range':np.quantile(null,[.025,.975]).tolist(),
        'repeats':repeats,'meaning':'Only within-generator assignments are destroyed; not causal attribution or independent significance'}


def run():
    threadpool_limits(2)
    protocol=read(OUT/'protocol.json')
    for relative,sha in protocol['inputs'].items(): assert digest(ROOT/relative)==sha
    data=torch.load(OLD/'features.pt',map_location='cpu',weights_only=False)
    rows=data['rows']; assert rows==read(OLD/'data_manifest.json')['rows']
    assert not {'V2C001','V2C002','V2C003'}&{r['scaffold'] for r in rows}
    for p in data['provenance']: assert '/generation_work/structures/' in p['pdb'] and digest(ROOT/p['pdb'])==p['sha256']
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','calibration','transfer']}
    tr,cal,te=[indices[s] for s in ['train','calibration','transfer']]
    external=[rows[i] for i in te]; y=np.array([r['target'] for r in rows],dtype=np.float32)
    simple=data['simple']
    xmap={'composition_tuned':simple,
          'composition_position_tuned':np.concatenate([simple,np.array([order_features(r['h3_sequence']) for r in rows])],axis=1),
          'composition_bigram_tuned':np.concatenate([simple,np.array([bigrams(r['h3_sequence']) for r in rows])],axis=1)}
    pred={}; fits={}
    for name,x in xmap.items():
        model,fit=choose(x,rows,tr,cal); pred[name]=model.predict(x[te]); fits[name]=fit
        scaler=model.named_steps['standardscaler']; ridge=model.named_steps['ridge']
        np.savez_compressed(OUT/(name+'.npz'),mean=scaler.mean_,scale=scaler.scale_,coef=ridge.coef_,intercept=ridge.intercept_)
    fixed=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(simple[tr],y[tr])
    pred['v4_ridge_fixed10_rebuilt']=fixed.predict(simple[te])
    context=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(simple[tr,20:],y[tr])
    pred['backbone_context_only']=context.predict(simple[te,20:])
    means={a:float(np.mean([r['target'] for r in rows if r['split']=='train' and r['arm']==a])) for a in {rows[i]['arm'] for i in tr}}
    pred['generator_mean_train_only']=np.array([means.get(r['arm'],float(y[tr].mean())) for r in external])
    old=read(OLD/'predictions.json'); assert old['entities']==[r['id'] for r in external]
    np.testing.assert_allclose(pred['v4_ridge_fixed10_rebuilt'],old['models']['ridge_alpha_10'],atol=1e-7)
    for name in ['full_s2041','full_s2053','full_s2069','no_antigen_context_s2041']:
        pred['v4_'+name]=np.array(old['models'][name])
    models={}; permutations={}
    for name,p in pred.items():
        models[name]={'all_entities':summarize(by_scaffold(external,p,False)),
          'generated_only':summarize(by_scaffold(external,p)), 'within_generator':within_arm(external,p)}
        permutations[name]=permutation_diagnostic(external,p)
    # Label-only descriptive decomposition, never fed to new models.
    decomposition=[]
    for s in sorted({r['scaffold'] for r in external}):
        rr=[r for r in external if r['scaffold']==s and r['entity_type']=='candidate']
        values=np.array([r['target'] for r in rr]); total=float(np.sum((values-values.mean())**2))
        within=sum(sum((r['target']-np.mean([v['target'] for v in rr if v['arm']==a]))**2 for r in rr if r['arm']==a) for a in {r['arm'] for r in rr})
        decomposition.append({'scaffold':s,'fraction_variance_between_generators':1-within/total if total else None})
    a,b=models['composition_position_tuned'],models['composition_tuned']
    gate=(a['generated_only']['median_rho']>b['generated_only']['median_rho'] and
          a['within_generator']['macro_reliable_pair_accuracy']>b['within_generator']['macro_reliable_pair_accuracy'] and
          a['generated_only']['macro_top20_recall']>=b['generated_only']['macro_top20_recall'])
    freeze(OUT/'fits.json',fits)
    freeze(OUT/'predictions.json',{'entities':[r['id'] for r in external],'models':{k:v.tolist() for k,v in pred.items()}})
    freeze(OUT/'report.json',{'protocol_sha256':digest(OUT/'protocol.json'),'models':models,'within_arm_permutation':permutations,
        'target_generator_decomposition':decomposition,'position_gate_passed':bool(gate),
        'counts':{k:len(v) for k,v in indices.items()},'new_independent_validation':False,'new_AF2_jobs':0})
    print(json.dumps({'position_gate_passed':bool(gate),'models':{k:{'all_rho':v['all_entities']['median_rho'],
        'candidate_rho':v['generated_only']['median_rho'],'within_generator_rho':v['within_generator']['median_rho'],
        'within_generator_reliable_pair_acc':v['within_generator']['macro_reliable_pair_accuracy']} for k,v in models.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.run: run()
