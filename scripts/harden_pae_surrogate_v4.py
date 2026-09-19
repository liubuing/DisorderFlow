"""Pre-AF2 development experiment; never modifies frozen v1-v3 evidence.

Inputs are design-time PDB coordinates and candidate sequences. AF2 matrices
are read ONLY for labels. The previously opened GP2 test is not evaluated.
All model choices are fixed in protocol.json before features/labels are built.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from disorderflow.models import get_model
from disorderflow.utils.protein.constants import Fragment
from modules.bfn_loader import build_region_batch, inject_candidate_sequence
from scripts.build_candidate_interface_multiscaffold_v1 import hierarchical_sem
from scripts.evaluate_mpnn_likelihood_baseline import chain_residues, locate_h3

OUT = ROOT / 'results/pae_surrogate_revision_v4'
AA = 'ACDEFGHIKLMNPQRSTVWY'
SEEDS = [2041, 2053, 2069]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def directional_target(pae, heavy_sequence, light_sequence, h3_sequence, antigen_sequence):
    """Only teacher-label path; never used to construct model features."""
    hstart = heavy_sequence.find(h3_sequence)
    if hstart < 0 or heavy_sequence.count(h3_sequence) != 1:
        raise ValueError('Ambiguous label H3 positions')
    agstart = len(heavy_sequence)+len(light_sequence)
    if pae.shape != (agstart+len(antigen_sequence),)*2:
        raise ValueError('Teacher array shape mismatch')
    return float(pae[hstart:hstart+len(h3_sequence),agstart:].mean()/31)


def screening(pred, target, fraction, top_fraction=0.2):
    """Expected recall under uniformly random tie breaking; lower is better.

    Soft target membership treats ties at the true top-k boundary fairly.
    This avoids arbitrary ID-order advantage for constant/mean baselines.
    """
    n = len(target)
    k, m = max(1, math.ceil(n*fraction)), max(1, math.ceil(n*top_fraction))
    def inclusion(values, slots):
        values = np.asarray(values)
        cutoff = np.sort(values)[slots-1]
        below, equal = values < cutoff, values == cutoff
        return below.astype(float) + equal * ((slots-below.sum())/equal.sum())
    selected = inclusion(pred, k)
    optimal = inclusion(target, m)
    return {'n': n, 'selected': k, 'top_count': m,
            'realized_fraction': k/n,
            'top20_recall': float(selected @ optimal / m),
            'random_expected_recall': k/n,
            'af2_call_reduction_fraction': 1-k/n}


def evaluate(rows, predictions):
    by = defaultdict(list)
    for i, row in enumerate(rows):
        by[row['scaffold']].append(i)
    result = {}
    for scaffold, indices in sorted(by.items()):
        p = np.asarray(predictions)[indices]
        y = np.asarray([rows[i]['target'] for i in indices])
        noise = np.asarray([rows[i]['sem'] for i in indices])
        pairs = {}
        for multiplier in (0, 0.5, 1, 2):
            dy, dp = y[:, None]-y, p[:, None]-p
            mask = np.triu(np.abs(dy) > multiplier*np.hypot(noise[:, None], noise), 1)
            # Tied predictions receive chance credit, never silently disappear.
            scores = (np.sign(dp) == np.sign(dy)).astype(float)
            scores[dp == 0] = 0.5
            pairs[str(multiplier)] = {'count': int(mask.sum()),
                'accuracy': float(scores[mask].mean()) if mask.any() else None}
        result[scaffold] = {'n': len(indices), 'spearman': correlation(p,y),
            'mae': float(np.abs(p-y).mean()), 'pair_sensitivity': pairs,
            'screening': {str(f): screening(p,y,f) for f in (0.1,0.2,0.5)}}
    rho = [r['spearman'] for r in result.values() if r['spearman'] is not None]
    return {'scaffolds': result, 'median_scaffold_spearman': float(np.median(rho)) if rho else None,
            'defined_spearman_scaffolds': len(rho),
            'macro_top20_recall': {str(f): float(np.mean([
                r['screening'][str(f)]['top20_recall'] for r in result.values()])) for f in (0.1,0.2,0.5)},
            'macro_random_recall': {str(f): float(np.mean([
                r['screening'][str(f)]['random_expected_recall'] for r in result.values()])) for f in (0.1,0.2,0.5)}}


def freeze_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    p = {'schema_version': 'pae_surrogate_revision_v4',
         'classification': 'posthoc_development_not_new_confirmation',
         'input_contract': 'Fixed design-input backbone and candidate H3 sequence; no AF2 coordinates, confidence or PAE in features.',
         'target_contract': 'Mean normalized directional H3-to-ALL-antigen PAE from six matrices; fixed sequence indices, no AF2-dependent patch selection.',
         'split': 'Original 15 train / 2 calibration. Six previously exposed external components used as descriptive transfer. GP2 test excluded.',
         'initializer': 'Frozen encoder from deployment seed 2041; fresh scalar heads, not a reproduction of joint v3 training.',
         'seeds': SEEDS, 'steps': 1000, 'lr': 0.001,
         'variants': ['full', 'no_antigen_context', 'no_noise_weight', 'no_grouped_difference'],
         'baselines': ['train_mean', 'ridge_alpha_10', 'extra_trees_300_min_leaf_3'],
         'selection': 'Minimum original-calibration MSE at steps 50,100,...,1000; no external selection.',
         'endpoints': ['signed scaffold Spearman', 'pair accuracy at SEM multipliers 0,0.5,1,2', 'top20% recall at budgets 10%,20%,50%'],
         'limitations': ['External cohort was already exposed; not a new blind test.',
                         'Cached-head ablations target a new scalar-only pre-AF2 model, not causal attribution for historical v3.',
                         'Existing frozen encoder pretraining provenance must be audited before claiming a fully independent test.']}
    path = OUT/'protocol.json'
    if path.exists():
        if read(path) != p:
            raise ValueError('Refusing to change frozen protocol')
    else:
        write(path,p)
    return p


def sources():
    split = read(ROOT/'data/candidate_interface_multiscaffold_v1/split_manifest.json')
    hold = read(ROOT/'data/multiscaffold_confirmatory_v2/holdout_manifest.json')
    ext = read(ROOT/'data/candidate_interface_external_calibration_v1/extension_manifest_v1.json')
    metadata = {c['component_id']:c['representative'] for c in hold['components']}
    metadata.update({c['component_id']:c for c in ext['components']})
    groups = {c:s for s in ('train','calibration') for c in split[s]}
    groups.update({c['component_id']:'transfer' for c in ext['components']})
    docs = []
    for prefix in ('candidate_interface_multiscaffold_v1','candidate_interface_multiscaffold_calibration_ext'):
        for sub in ('af2','af2_model2'):
            path = ROOT/f'results/{prefix}/{sub}/results.json'
            docs.append((path,read(path)))
    return metadata, groups, docs


def build_data():
    cache = OUT/'features.pt'
    if cache.exists():
        data = torch.load(cache, weights_only=False, map_location='cpu')
        for path, expected in data['source_manifests'].items():
            if digest(ROOT/path) != expected:
                raise ValueError(f'Cached data source changed: {path}')
        for source in data['provenance']:
            if digest(ROOT/source['pdb']) != source['sha256']:
                raise ValueError(f"Cached design backbone changed: {source['pdb']}")
        return data
    metadata, groups, docs = sources()
    entities, results = {}, defaultdict(list)
    for source_path, doc in docs:
        for e in doc['entities']:
            if e['component_id'] in groups:
                entities[e['entity_id']] = e
        for r in doc['results']:
            if r['component_id'] in groups and r['status'] == 'success':
                results[r['entity_id']].append({**r,'protocol_id':source_path.parent.name})
    deployment = read(ROOT/'publication/candidate_interface_pae_deployment_v1.json')
    # Old artifacts contain absolute D: paths; relocate the known repository suffix.
    ckpath = deployment['seeds'][0]['checkpoint'].replace('\\','/')
    ckpath = ROOT/ckpath.split('/DisorderFlow/',1)[-1]
    if digest(ckpath) != deployment['seeds'][0]['checkpoint_sha256']:
        raise ValueError('Initializer checkpoint hash mismatch')
    ck = torch.load(ckpath,map_location='cpu',weights_only=False)
    model = get_model(ck['config'].model)
    model.load_state_dict(ck['model'],strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    rows, features, simple, input_provenance = [], [], [], []
    begin = time.perf_counter()
    with torch.no_grad():
        for scaffold in sorted(groups):
            meta = metadata[scaffold]
            pdb = ROOT/f'data/multiscaffold_confirmatory_v2/generation_work/structures/{scaffold}.pdb'
            chain_ids = sorted({line[21] for line in pdb.read_text().splitlines() if line.startswith('ATOM')})
            residues = {c:chain_residues(pdb,c) for c in chain_ids}
            seqs = {c:''.join(r[2] for r in rs) for c,rs in residues.items()}
            hc = next(c for c,s in seqs.items() if meta['vh_sequence'] in s)
            lc = next(c for c,s in seqs.items() if meta['vl_sequence'] in s)
            ac = next(c for c,s in seqs.items() if c not in (hc,lc) and s == meta['antigen_sequence'])
            indices = locate_h3(residues[hc],meta['cdr_h3_sequence'],meta.get('h3_heavy_indices_zero_based'))
            if any(residues[hc][i][1] for i in indices):
                raise ValueError('Insertion-coded design residues need explicit mapping')
            region = hc+':'+','.join(str(residues[hc][i][0]) for i in indices)
            batch = build_region_batch(str(pdb),region,context_chains=[lc,ac],
                antigen_chains=[ac],antigen_context_cap=50,preserve_context_chain_order=True,device=device)
            cm = batch['generate_flag'][0].bool() & batch['mask'][0].bool()
            am = (batch['fragment_type'][0] == int(Fragment.Antigen)) & batch['mask'][0].bool()
            if int(am.sum()) != len(meta['antigen_sequence']):
                raise ValueError('Design patch must retain every antigen residue')
            # Candidate identity is masked by the frozen encoder, so context pair
            # features can be cached once per fixed design backbone.
            _, pair = model.encode(batch,remove_structure=True,remove_sequence=True)
            pair_mean = pair[0][cm][:,am].mean((0,1))
            d = torch.cdist(batch['pos_heavyatom'][0][cm,1],batch['pos_heavyatom'][0][am,1]).min(1).values
            geom = [len(indices),int(am.sum()),float(d.mean()),float(d.std(unbiased=False)),float(d.min()),float((d<8).float().mean())]
            input_provenance.append({'scaffold':scaffold,'pdb':pdb.relative_to(ROOT).as_posix(),'sha256':digest(pdb),'chains':[hc,lc,ac]})
            for eid,e in sorted(entities.items()):
                if e['component_id'] != scaffold:
                    continue
                if len(results[eid]) != 6:
                    raise ValueError(f'{eid}: incomplete teacher grid')
                inject_candidate_sequence(batch,e['h3_sequence'])
                probs = torch.nn.functional.one_hot(batch['aa'].clamp(0,model.bfn.num_classes-1),model.bfn.num_classes).float()
                seqemb = model.bfn.receiver.v12_seq_emb(probs)[0]
                features.append(torch.cat([seqemb[cm].mean(0),seqemb[am].mean(0),pair_mean]).cpu())
                composition = lambda s: [s.count(a)/len(s) for a in AA]
                simple.append(composition(e['h3_sequence'])+composition(e['antigen_sequence'])+geom)
                grid, elapsed = [], 0.0
                for r in results[eid]:
                    path = ROOT/r['pae_npz']
                    if digest(path) != r['pae_sha256']:
                        raise ValueError(f'PAE checksum mismatch: {path}')
                    with np.load(path) as z:
                        pae = z['pae']
                        value = directional_target(pae,e['heavy_sequence'],e['light_sequence'],e['h3_sequence'],e['antigen_sequence'])
                    grid.append({'protocol_id':r['protocol_id'],'af2_seed':r['af2_seed'],'value':value})
                    elapsed += r.get('elapsed',0)
                values = [r['value'] for r in grid]
                sd = float(np.std(values,ddof=1))
                rows.append({'id':eid,'scaffold':scaffold,'split':groups[scaffold],
                    'arm':e['arm'],'entity_type':e['entity_type'],'h3_sequence':e['h3_sequence'],
                    'antigen_sequence':e['antigen_sequence'],'target':float(np.mean(values)),
                    'sem':hierarchical_sem(grid,'value'),'weight':float(np.clip(1/(1+(sd/0.1)**2),0.1,1)),
                    'af2_six_replicate_recorded_seconds':elapsed})
            print(f'features: {scaffold} complete ({len(rows)} entities)',flush=True)
    data = {'rows':rows,'features':torch.stack(features),'simple':np.asarray(simple),
            'provenance':input_provenance,'checkpoint_sha256':digest(ckpath),
            'source_manifests':{p.relative_to(ROOT).as_posix():digest(p) for p,_ in docs},
            'feature_and_label_extraction_seconds':time.perf_counter()-begin}
    torch.save(data,cache)
    write(OUT/'data_manifest.json',{k:v for k,v in data.items() if k not in ('features','simple')})
    return data


class ScalarHead(torch.nn.Module):
    """Algebraic scalar PAE branch of v2, with fresh parameters."""
    def __init__(self,no_antigen=False):
        super().__init__()
        self.no_antigen = no_antigen
        self.i = torch.nn.Linear(64,32)
        self.j = torch.nn.Linear(64,32)
        self.pair = torch.nn.Linear(128,32)
        self.out = torch.nn.Sequential(torch.nn.LayerNorm(32),torch.nn.ReLU(),
                                       torch.nn.Dropout(0.05),torch.nn.Linear(32,1))

    def forward(self,x):
        hidden = self.i(x[:,:64])
        if not self.no_antigen:
            hidden = hidden+self.j(x[:,64:128])+self.pair(x[:,128:])
        return torch.sigmoid(self.out(hidden)).squeeze(-1)


def fit_head(x,y,w,groups,train,val,variant,seed,steps):
    torch.manual_seed(seed)
    model = ScalarHead(variant=='no_antigen_context')
    opt = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    xt,yt,wt = x[train],y[train],w[train]
    if variant == 'no_noise_weight':
        wt = torch.ones_like(wt)
    gt = np.asarray(groups)[train]
    pair_mask = torch.tensor(np.triu(gt[:,None]==gt[None,:],1))
    weights_pair = torch.sqrt(wt[:,None]*wt[None,:])[pair_mask]
    truth_delta = (yt[:,None]-yt[None,:])[pair_mask]
    best_loss,best,best_step = float('inf'),None,None
    for step in range(1,steps+1):
        model.train()
        pred = model(xt)
        loss = ((pred-yt).square()*wt).sum()/wt.sum()
        if variant != 'no_grouped_difference':
            delta = (pred[:,None]-pred[None,:])[pair_mask]
            loss = loss+0.5*((delta-truth_delta).square()*weights_pair).sum()/weights_pair.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vl = float((model(x[val])-y[val]).square().mean())
            if vl < best_loss:
                best_loss,best,best_step = vl,copy.deepcopy(model.state_dict()),step
    model.load_state_dict(best)
    model.eval()
    return model,best_step,best_loss


def paired_bootstrap(full, other, repetitions=10000):
    shared = sorted(set(full['scaffolds']) & set(other['scaffolds']))
    differences = [full['scaffolds'][s]['spearman']-other['scaffolds'][s]['spearman'] for s in shared
                   if full['scaffolds'][s]['spearman'] is not None and other['scaffolds'][s]['spearman'] is not None]
    if not differences:
        return None
    values = np.asarray(differences)
    rng = np.random.default_rng(41027)
    means = rng.choice(values,(repetitions,len(values)),replace=True).mean(1)
    return {'mean_scaffold_rho_difference':float(values.mean()),
            'percentile_95_ci':np.quantile(means,[0.025,0.975]).tolist(),
            'units':len(values),'unit':'scaffold; descriptive exposed external cohort'}


def experiment(data,protocol):
    rows,x,simple = data['rows'],data['features'].float(),data['simple']
    y = torch.tensor([r['target'] for r in rows],dtype=torch.float32)
    w = torch.tensor([r['weight'] for r in rows],dtype=torch.float32)
    groups = [r['scaffold'] for r in rows]
    masks = {s:np.asarray([r['split']==s for r in rows]) for s in ('train','calibration','transfer')}
    external = [r for r in rows if r['split']=='transfer']
    report = {'protocol_sha256':digest(OUT/'protocol.json'),'features_sha256':digest(OUT/'features.pt'),
              'classification':protocol['classification'],'counts':{s:int(m.sum()) for s,m in masks.items()},
              'models':{},'comparisons':{},'new_blind_external_validation_complete':False}
    output_predictions = {'entities':[r['id'] for r in external],'target':[r['target'] for r in external],'models':{}}
    estimators = {'ridge_alpha_10':make_pipeline(StandardScaler(),Ridge(alpha=10)),
                  'extra_trees_300_min_leaf_3':ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,random_state=2041,n_jobs=2)}
    for name in ['train_mean',*estimators]:
        if name=='train_mean':
            pred = np.repeat(float(y[masks['train']].mean()),len(external))
        else:
            est = estimators[name].fit(simple[masks['train']],y[masks['train']].numpy())
            pred = est.predict(simple[masks['transfer']])
        report['models'][name] = evaluate(external,pred)
        output_predictions['models'][name] = pred.tolist()
    # Reuse ONLY baseline sequence NLL, discard the historical mismatched PAE.
    old = read(ROOT/'results/candidate_interface_mpnn_baseline_v1/mpnn_likelihood_baseline.json')
    nll = {eid:r['h3_nll'] for s in old['results'].values() for eid,r in s['per_entity'].items()}
    pred = np.asarray([nll[r['id']] for r in external])
    report['models']['mpnn_nll_signed_same_target'] = evaluate(external,pred)
    output_predictions['models']['mpnn_nll_signed_same_target'] = pred.tolist()
    for variant in protocol['variants']:
        for seed in SEEDS:
            model,step,vl = fit_head(x,y,w,groups,masks['train'],masks['calibration'],variant,seed,protocol['steps'])
            with torch.no_grad():
                pred = model(x[masks['transfer']]).numpy()
            name = f'{variant}_s{seed}'
            report['models'][name] = {**evaluate(external,pred),'selected_step':step,'calibration_mse':vl}
            torch.save({'state_dict':model.state_dict(),'variant':variant,'seed':seed,
                        'protocol_sha256':report['protocol_sha256']},OUT/f'{name}.pt')
            output_predictions['models'][name] = pred.tolist()
            print(name,report['models'][name]['median_scaffold_spearman'],flush=True)
    for name in report['models']:
        if name.startswith('full_'):
            continue
        for seed in SEEDS:
            if '_s' in name and not name.endswith(f'_s{seed}'):
                continue
            report['comparisons'][f'full_s{seed}_minus_{name}'] = paired_bootstrap(report['models'][f'full_s{seed}'],report['models'][name])
    report['cost_scope'] = {'af2_cost':'recorded sum of six elapsed fields; excludes startup/MSA/preprocessing unless included by original runner',
        'speedup_claim':'not measured end-to-end; no 2900-4900x claim permitted',
        'screening':'budget reduction is a count of avoided six-replicate teacher evaluations, conditional on recall; not measured experimental hit enrichment'}
    write(OUT/'predictions.json',output_predictions)
    write(OUT/'experiment_report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = freeze_protocol()
    data = build_data()
    if not args.prepare_only:
        experiment(data,protocol)


if __name__=='__main__':
    main()
