"""Bounded retrospective PAE screening pilot; never an independent validation."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from Bio.PDB import MMCIFParser, PDBIO, Structure, Model, Chain
from Bio.SeqUtils import seq1
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.run_multiscaffold_v2_af2 import atomic_json, windows_to_wsl, build_entities
from scripts.audit_pae_public_provenance_v9 import STANDARD
from scripts import resume_pae_dense_batched as transport

OUT = ROOT / 'results/pae_public_pilot_v11'
SOURCE = ROOT / 'data/pae_public_provenance_v9/qualified_candidates.json'
WEIGHTS = ROOT / 'results/pae_screening_v7/ridge_fixed10.npz'
SEEDS = [20260919, 20261919, 20262919, 20263919]
AA = 'ACDEFGHIKLMNPQRSTVWY'


def status(stage, **kwargs):
    atomic_json(OUT / 'pipeline_status.json', {'stage':stage, 'pid':os.getpid(), 'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'), **kwargs})


def selected_sequences(component, pool):
    return sorted(pool, key=lambda s:hashlib.sha256(f'20260919|{component}|{s}'.encode()).hexdigest())[:20]


def canonicalize(r, path):
    assert digest(ROOT/r['source_cif']) == r['source_cif_sha256']
    original = next(MMCIFParser(QUIET=True).get_structure(r['instance'],str(ROOT/r['source_cif'])).get_models())
    structure = Structure.Structure(r['instance'])
    model = Model.Model(0)
    structure.add(model)
    for role, canonical, field in [('heavy','H','heavy_sequence'),('light','L','light_sequence'),('antigen','P','antigen_sequence')]:
        source = original[r['original_chain_ids'][role]]
        target = Chain.Chain(canonical)
        observed = []
        for residue in source:
            if 'CA' not in residue:
                continue
            if residue.resname not in STANDARD or residue.id[0] != ' ':
                raise ValueError(f'Unsupported chemistry {r["pdb_id"]} {role} {residue.resname}')
            if not all(a in residue for a in ('N','CA','C')):
                raise ValueError('Incomplete backbone')
            clone = copy.deepcopy(residue, {id(source):target})
            clone.detach_parent()
            clone.id = (' ',len(observed)+1,' ')
            target.add(clone)
            observed.append(seq1(residue.resname))
        if ''.join(observed) != r[field]:
            raise ValueError('Sequence differs from chemical qualification')
        model.add(target)
    path.parent.mkdir(parents=True,exist_ok=True)
    writer = PDBIO()
    writer.set_structure(structure)
    temporary = path.with_suffix('.tmp.pdb')
    writer.save(str(temporary))
    if path.exists():
        assert digest(path)==digest(temporary)
        temporary.unlink()
    else:
        temporary.replace(path)


def prepare():
    OUT.mkdir(exist_ok=True)
    groups = read(SOURCE)['components']
    components = []
    for group in groups:
        r = group['representative']
        path = OUT / 'design_structures' / (group['component_id']+'.pdb')
        canonicalize(r,path)
        components.append({**group, 'representative_id':r['instance'], 'canonical_pdb':str(path.relative_to(ROOT)), 'canonical_sha256':digest(path)})
    freeze(OUT / 'holdout.json', {'classification':'retrospective_overlap_disclosed', 'components':components, 'source_sha256':digest(SOURCE)})
    cfg = yaml.safe_load((ROOT / 'configs/benchmarks/multiscaffold_confirmatory_v2_calibration_ext_generation.yml').read_text())
    mpnn = cfg['generation']['arms']['proteinmpnn']
    mpnn['temperature'] = 0.3
    freeze(OUT / 'protocol.json', {
        'classification':'retrospective_descriptive_pilot_not_independent_confirmation',
        'parent_protocol_sha256':digest(ROOT/'results/pae_public_pilot_v10/protocol.json'),
        'parent_inventory_sha256':digest(ROOT/'results/pae_public_pilot_v10/inventory.json'),
        'amendment':'New pilot after v10 pre-label diversity failure: all six components temperature0.3 and fresh seeds; v10 unchanged, no pools mixed, min10 gate unchanged; not an independent confirmatory study',
        'holdout_sha256':digest(OUT/'holdout.json'), 'ridge_sha256':digest(WEIGHTS),
        'generator':mpnn, 'generation_seeds':SEEDS, 'draws_per_seed':32,
        'scope':'single ProteinMPNN generator; no claims comparing generators, no new training',
        'candidate_rule':'all128 draws percomponent; remove native and exact duplicates; SHA256(20260919|component|sequence) first20; require at least10 for EVERY component; otherwise stop before labels',
        'teacher':'AF2-Multimer v3 models1/2 x seeds7103/7111/7121 recycles3 single-sequence; all candidates and native/shuffle controls',
        'feature_contract':'H3 composition20 + antigen composition20 + H3length,AGlength,nearestCA distance mean/populationSD/min/fraction<8 from designPDB',
        'screening_budget':.2, 'rounding':'ceil on observed unique candidate count; tie entity_id',
        'primary':'within-component recall of true lowestPAE20percent at predicted20percent budget; random expected recall k/n',
        'aggregation':'report everycomponent; average CSP3components first, then equalweight4biological strata; four strata are descriptive not independent sampling units',
        'secondary':'Spearman percomponent; no significance or CI from candidate pseudoreplication; no claims of binding affinity',
        'failures':'3consecutiveAF2 failures stop; incomplete6replicate entities stop final evaluation; no silent droppedcomponents',
        'controls':'native and composition shuffle; excluded from screening and primary counts',
        'overlap':'knownCSP and fusionpeptide; V3 previously structurally examined; historicalmultimer/MPNN membership unresolved',
        'cost':'recordslotelapsed and localwalltime separately, no end-to-end speedup claim',
        'worker_batch_size':96})
    files = [Path(__file__),ROOT/'scripts/resume_pae_dense_batched.py',ROOT/'scripts/utils/af2_wsl_batch.py',
        ROOT/'scripts/benchmark_h3_candidate_reranking.py',ROOT/'scripts/run_multiscaffold_v2_af2.py',
        ROOT/'ProteinMPNN/protein_mpnn_run.py',ROOT/'ProteinMPNN/protein_mpnn_utils.py',
        ROOT/'scripts/run_pae_dense_v6.py',
        ROOT/'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2.yml',
        ROOT/'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2_model2.yml']
    freeze(OUT/'execution_code_manifest.json',{'files':{str(p.relative_to(ROOT)):digest(p) for p in files}})
    print('Prepared6components,4antigenstrata,768ProteinMPNN draws; no labels yet.',flush=True)


def verify():
    for path, sha in read(OUT/'execution_code_manifest.json')['files'].items():
        assert digest(ROOT/path)==sha
    protocol = read(OUT/'protocol.json')
    assert digest(OUT/'holdout.json')==protocol['holdout_sha256']
    assert digest(WEIGHTS)==protocol['ridge_sha256']
    for c in read(OUT/'holdout.json')['components']:
        assert digest(ROOT/c['canonical_pdb'])==c['canonical_sha256']
    return protocol


def generate():
    from scripts.benchmark_h3_candidate_reranking import generate_mpnn, validate_candidate
    protocol = verify()
    settings = protocol['generator']
    assert digest(ROOT/settings['weights']/(settings['model_name']+'.pt'))==settings['model_weights_sha256']
    path = OUT/'generation.json'
    result = read(path) if path.exists() else {'attempts':[], 'completed_groups':[]}
    for c in read(OUT/'holdout.json')['components']:
        r = c['representative']
        for seed in SEEDS:
            group_id = f'{c["component_id"]}|{seed}'
            if group_id in result['completed_groups']:
                continue
            begin = time.perf_counter()
            rows, command = generate_mpnn({'generation':{'candidates_per_seed':32,'proteinmpnn':settings}},r,
                r['h3_heavy_indices_zero_based'],ROOT/c['canonical_pdb'],{'H':{'length':len(r['heavy_sequence'])}},seed,
                OUT/'generation_work'/c['component_id']/str(seed))
            if len(rows)!=32:
                raise ValueError(f'{group_id}: expected32 draws, found{len(rows)}')
            for i,row in enumerate(rows):
                seq = validate_candidate(row['sequence'],len(r['cdr_h3_sequence']))
                heavy = list(r['heavy_sequence'])
                for index,aa in zip(r['h3_heavy_indices_zero_based'],seq,strict=True):
                    heavy[index]=aa
                result['attempts'].append({'id':f'{group_id}|{i:02d}','component_id':c['component_id'], 'seed':seed,
                    'sequence':seq,'full_heavy_sequence':''.join(heavy),'native':seq==r['cdr_h3_sequence']})
            result['completed_groups'].append(group_id)
            result['last_group_seconds']=time.perf_counter()-begin
            result['complete']=len(result['completed_groups'])==24
            atomic_json(path,result)
            status('generation',completed_draws=len(result['attempts']),expected_draws=768)
            print('Generation',group_id,len(result['attempts']),'/768',flush=True)


def select_and_predict():
    verify()
    generation = read(OUT/'generation.json')
    assert generation['complete'] and len(generation['attempts'])==768
    hold = read(OUT/'holdout.json')
    selections, inventory = [], []
    for c in hold['components']:
        pool = {}
        for row in generation['attempts']:
            if row['component_id']==c['component_id'] and not row['native']:
                pool.setdefault(row['sequence'],row)
        inventory.append({'component_id':c['component_id'],'unique_non_native':len(pool),'selected':min(20,len(pool))})
        for i,seq in enumerate(selected_sequences(c['component_id'],pool)):
            row=pool[seq]
            selections.append({'status':'selected','selection_id':f'{c["component_id"]}|proteinmpnn|{i:02d}',
                'component_id':c['component_id'],'arm':'proteinmpnn','selection_slot':i,
                'source_attempt_id':row['id'],'sequence':seq,'full_heavy_sequence':row['full_heavy_sequence']})
    freeze(OUT/'inventory.json',{'rows':inventory})
    if any(r['unique_non_native']<10 for r in inventory):
        raise ValueError('Insufficient diversity in at least one component; frozen min10 gate; no AF2 launched')
    freeze(OUT/'selection.json',{'selections':selections,'generation_sha256':digest(OUT/'generation.json')})
    entities = build_entities({'selections':selections},hold,20260919)
    freeze(OUT/'entities.json',{'entities':entities})
    with np.load(WEIGHTS) as z:
        w={k:z[k].copy() for k in z.files}
    predictions = {}
    for c in hold['components']:
        # Canonical PDB parser, not AF2 coordinates.
        from Bio.PDB import PDBParser
        model=next(PDBParser(QUIET=True).get_structure(c['component_id'],str(ROOT/c['canonical_pdb'])).get_models())
        r=c['representative']; h=list(model['H']); ag=list(model['P'])
        x=torch.tensor(np.array([h[i]['CA'].coord for i in r['h3_heavy_indices_zero_based']]))
        y=torch.tensor(np.array([a['CA'].coord for a in ag]))
        d=torch.cdist(x,y).min(1).values
        geom=[len(x),len(y),float(d.mean()),float(d.std(unbiased=False)),float(d.min()),float((d<8).float().mean())]
        comp=lambda s:[s.count(a)/len(s) for a in AA]
        for e in entities:
            if e['component_id']!=c['component_id']: continue
            features=np.array(comp(e['h3_sequence'])+comp(e['antigen_sequence'])+geom)
            predictions[e['entity_id']]=float(((features-w['mean'])/w['scale'])@w['coef']+w['intercept'])
    freeze(OUT/'pre_teacher_predictions.json',{'predictions':predictions,'input_scope':'designPDB+sequence only',
        'ridge_sha256':digest(WEIGHTS),'entities_sha256':digest(OUT/'entities.json')})


def entity_key(e,m,s):
    return (e['component_id'],e['heavy_sequence'],e['light_sequence'],e['antigen_sequence'],m,s)


def teacher():
    verify()
    entities=read(OUT/'entities.json')['entities']
    byid={e['entity_id']:e for e in entities}
    cache={}; logfile=OUT/'teacher_results.jsonl'
    if logfile.exists():
        for line in logfile.read_text().splitlines():
            r=json.loads(line)
            if r['success']:
                assert digest(ROOT/r['pae_path'])==r['pae_sha256']
                cache[entity_key(byid[r['entity_id']],r['model'],r['seed'])]=r
    transport.OUT=OUT
    total_slots=len({entity_key(e,m,s) for e in entities for m in (1,2) for s in (7103,7111,7121)})
    for m in (1,2):
        suffix='_model2' if m==2 else ''
        cfg=yaml.safe_load((ROOT/f'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2{suffix}.yml').read_text())['af2']
        weight=Path(os.environ['COLABFOLD_CACHE'])/'params'/Path(cfg['model_parameters']).name
        assert digest(weight)==cfg['model_parameters_sha256']
        jobs=[]; mapping={}; seen=set()
        for e in sorted(entities,key=lambda r:(len(r['heavy_sequence'])+len(r['light_sequence'])+len(r['antigen_sequence']),r['entity_id'])):
            for seed in (7103,7111,7121):
                k=entity_key(e,m,seed)
                if k in cache or k in seen: continue
                seen.add(k)
                id=f'{e["entity_id"]}|model{m}|seed{seed}'
                stem=hashlib.sha256(id.encode()).hexdigest()[:24]
                pae=OUT/'pae'/(stem+'.npz')
                mapping[id]=(e,seed,pae)
                jobs.append({'id':id,'seq':e['heavy_sequence']+':'+e['light_sequence'],'epi_seq':e['antigen_sequence'],
                    'seed':seed,'recycle':3,'output_pdb':windows_to_wsl(OUT/'pdb'/(stem+'.pdb')),'output_pae':windows_to_wsl(pae)})
        if not jobs: continue
        path=OUT/f'jobs_model{m}.jsonl'
        path.write_text(''.join(json.dumps(j)+'\n' for j in jobs))
        command=f'cd {windows_to_wsl(ROOT)} && source {cfg["environment"]}/bin/activate && python scripts/utils/af2_wsl_batch.py --recycle 3 --model-number {m}'
        consecutive=0
        with path.open() as stdin, (OUT/f'af2_model{m}.stderr.log').open('a') as stderr,logfile.open('a') as output:
            process=transport.BatchedWorker(['wsl.exe','-d',cfg['wsl_distribution'],'--','bash','-lc',command],
                stdin=stdin,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',errors='replace',bufsize=1)
            for line in process.stdout:
                try: raw=json.loads(line)
                except ValueError: continue
                if raw.get('id') not in mapping:
                    print(line.strip(),flush=True); continue
                e,s,pae=mapping[raw['id']]
                success=bool(raw.get('success')) and pae.exists() and digest(pae)==raw.get('pae_sha256')
                r={'entity_id':e['entity_id'],'model':m,'seed':s,'success':success,'pae_path':str(pae.relative_to(ROOT)),
                    'pae_sha256':raw.get('pae_sha256'),'elapsed':raw.get('elapsed')}
                output.write(json.dumps(r)+'\n'); output.flush()
                if success: cache[entity_key(e,m,s)]=r
                consecutive=0 if success else consecutive+1
                status('af2',completed_unique_slots=len(cache),total_unique_slots=total_slots,worker_pid=process.pid,latest_success=success)
                print('AF2',len(cache),'/',total_slots,success,flush=True)
                if consecutive>=3:
                    process.terminate(); raise RuntimeError('Three consecutive teacher failures')
            if process.wait()!=0: raise RuntimeError('Bounded AF2 worker exhausted retries')
    from scripts.run_pae_dense_v6 import target
    rows=[]
    for e in entities:
        observations=[]
        for m in (1,2):
            for s in (7103,7111,7121):
                r=cache.get(entity_key(e,m,s))
                if r is None: raise ValueError('Incomplete6replicate teacher panel')
                observations.append({**r,'target':target(ROOT/r['pae_path'],e)})
        rows.append({**e,'observations':observations,'target':float(np.mean([o['target'] for o in observations]))})
    freeze(OUT/'teacher_labels.json',{'rows':rows,'incomplete_entities':[],'independent_validation':False})


def screening_metrics(ids,prediction,target):
    n=len(ids); k=int(np.ceil(.2*n))
    chosen=sorted(range(n),key=lambda i:(prediction[i],ids[i]))[:k]
    best=sorted(range(n),key=lambda i:(target[i],ids[i]))[:k]
    rho=float(spearmanr(prediction,target).statistic) if np.ptp(prediction)>0 and np.ptp(target)>0 else None
    return {'n':n,'budget':k,'recall':len(set(chosen)&set(best))/k,'random_expected_recall':k/n,'spearman':rho}


def evaluate():
    verify()
    rows=read(OUT/'teacher_labels.json')['rows']
    pred=read(OUT/'pre_teacher_predictions.json')['predictions']
    cells={}; families={}
    for c in read(OUT/'holdout.json')['components']:
        selected=[r for r in rows if r['component_id']==c['component_id'] and r['entity_type']=='candidate']
        cell=screening_metrics([r['entity_id'] for r in selected],[pred[r['entity_id']] for r in selected],[r['target'] for r in selected])
        cells[c['component_id']]=cell
        families.setdefault(c['family_stratum'],[]).append(cell)
    strata={k:{'components':len(v),'recall':float(np.mean([r['recall'] for r in v])),
        'random_expected_recall':float(np.mean([r['random_expected_recall'] for r in v]))} for k,v in families.items()}
    freeze(OUT/'evaluation.json',{'classification':'retrospective_descriptive_not_confirmatory','components':cells,'antigen_strata':strata,
        'stratum_equal_recall':float(np.mean([v['recall'] for v in strata.values()])),
        'stratum_equal_random_expected_recall':float(np.mean([v['random_expected_recall'] for v in strata.values()])),
        'teacher_sha256':digest(OUT/'teacher_labels.json'),'pre_teacher_predictions_sha256':digest(OUT/'pre_teacher_predictions.json')})
    status('evaluation_complete',evaluation_path=str((OUT/'evaluation.json').relative_to(ROOT)))


def run():
    verify()
    status('generation')
    generate()
    select_and_predict()
    status('af2',note='96job boundedworkers; counters update on first successful slot')
    teacher()
    evaluate()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    args=parser.parse_args()
    try:
        (prepare if args.stage=='prepare' else run)()
    except Exception as error:
        if OUT.exists(): status('blocked',error=str(error))
        raise
