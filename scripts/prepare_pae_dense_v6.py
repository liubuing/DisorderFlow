"""Prepare a balanced exposed-scaffold PAE panel without score-based selection."""
import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze
from scripts.benchmark_aayl_learnability import digest

OUT=ROOT/'results/pae_dense_v6'
RAW=ROOT/'results/candidate_interface_multiscaffold_calibration_ext/raw_generation.json'
HOLD=ROOT/'data/candidate_interface_multiscaffold_calibration_ext_v1/holdout_manifest.json'
ARMS=['current_bfn_disorder_on','current_bfn_disorder_off','stage_a_antibody_bfn','proteinmpnn']


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def write_yaml(path,value):
    text=yaml.safe_dump(value,sort_keys=False)
    if path.exists() and path.read_text()!=text: raise ValueError('Frozen config differs')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)


def prepare():
    freeze(OUT/'protocol.json',{'status':'exposed_scaffold_dense_development','date':'2026-09-17',
        'raw_sha256':digest(RAW),'holdout_sha256':digest(HOLD),
        'old_pool':'use existing 24 attempts per scaffold/arm, no generator or teacher score selection',
        'supplement':'ProteinMPNN only, four frozen seeds x32 draws for all six scaffolds; unchanged temperature0.1; no adaptation to PAE',
        'seeds':[20260917,20261917,20262917,20263917],'per_seed':32,'target_unique_per_arm':20,
        'arms':ARMS,'selection':'exclude exact native; deduplicate within scaffold/arm; ascending SHA256(20260917|scaffold|arm|sequence); first20; fail if insufficient',
        'teacher':'same AF2-Multimer v3 models1/2 x seeds7103/7111/7121, recycles3, single sequence; full matrices; reuse only exact sequence+scaffold+protocol+seed',
        'models':'original v4 Ridge alpha10, frozen v5 composition/position/bigram; no new fitting on expanded candidates',
        'models_sha256':{n:digest(ROOT/'results/pae_sequence_order_v5'/n) for n in ['composition_tuned.npz','composition_position_tuned.npz','composition_bigram_tuned.npz']},
        'primary':'component-equal within-generator rank and screening, then pooled generators; missing teacher slots retained; complete six-replicate cohort explicit',
        'independence':'same six exposed scaffolds, no new independent antigen component; old selected labels exposed; not a blind test',
        'cost':'record new worker wall clock and reused slots separately; historical reuse is not end-to-end speedup',
        'script_sha256':digest(Path(__file__))})
    config=yaml.safe_load((ROOT/'configs/benchmarks/multiscaffold_confirmatory_v2_calibration_ext_generation.yml').read_text())
    config['protocol']='results/pae_dense_v6/protocol.json'; config['protocol_sha256']=digest(OUT/'protocol.json')
    config['generation']['seeds']=[20260917,20261917,20262917,20263917]
    config['generation']['candidates_per_seed']=32
    config['generation']['arms']={'proteinmpnn':config['generation']['arms']['proteinmpnn']}
    config['generation']['raw_attempts_expected']=768
    config['output']['path']='results/pae_dense_v6/supplement_generation.json'
    config['claim_boundary']='exposed scaffold supplemental sampling, no independent confirmation'
    write_yaml(OUT/'generation.yml',config)
    print('Frozen 20 candidates x4 arms x6 scaffolds; supplement generation configured')


def select():
    protocol=read(OUT/'protocol.json')
    assert digest(RAW)==protocol['raw_sha256'] and digest(HOLD)==protocol['holdout_sha256']
    hold=read(HOLD); reps={c['component_id']:c['representative'] for c in hold['components']}
    pools=defaultdict(dict); counts=defaultdict(lambda:defaultdict(int))
    for path in [RAW,OUT/'supplement_generation.json']:
        for row in read(path)['attempts']:
            key=(row['component_id'],row['arm']); counts[key][row['status']]+=1
            if row['status']!='success': continue
            if row['sequence']==reps[key[0]]['cdr_h3_sequence']: counts[key]['native']+=1; continue
            pools[key].setdefault(row['sequence'],row)
    selected=[]; inventory=[]
    for scaffold in sorted(reps):
        for arm in ARMS:
            pool=pools[(scaffold,arm)]
            inventory.append({'scaffold':scaffold,'arm':arm,'unique_non_native':len(pool),'counts':dict(counts[(scaffold,arm)])})
            if len(pool)<20:
                freeze(OUT/'insufficient_inventory.json',{'rows':inventory,'incomplete_at':[scaffold,arm]})
                raise ValueError(f'Insufficient unique candidates {scaffold} {arm}: {len(pool)}')
            seqs=sorted(pool,key=lambda seq:hashlib.sha256(f'20260917|{scaffold}|{arm}|{seq}'.encode()).hexdigest())[:20]
            for slot,seq in enumerate(seqs,1):
                row=pool[seq]
                selected.append({'selection_id':f'{scaffold}|{arm}|dense_{slot:02d}', 'status':'selected',
                    'component_id':scaffold,'arm':arm,'selection_slot':slot,'source_attempt_id':row['attempt_id'],
                    'sequence':seq,'full_heavy_sequence':row['full_heavy_sequence']})
    freeze(OUT/'selection.json',{'selections':selected,'inventory':inventory,'protocol_sha256':digest(OUT/'protocol.json'),
        'supplement_sha256':digest(OUT/'supplement_generation.json'),'score_inputs_used':False})
    from scripts.run_multiscaffold_v2_af2 import build_entities
    entities=build_entities({'selections':selected},hold,7307)
    freeze(OUT/'entities.json',{'entities':entities})
    # Restrict all reuse to the previously retained six exposed components.
    cache=[]
    for model,sub in [(1,'af2'),(2,'af2_model2')]:
        old=read(ROOT/f'results/candidate_interface_multiscaffold_calibration_ext/{sub}/results.json')
        byid={e['entity_id']:e for e in old['entities']}
        for r in old['results']:
            if r['status']=='success':
                e=byid[r['entity_id']]
                if e['component_id'] in reps: cache.append({'model':model,'entity':e,'result':r})
    freeze(OUT/'historical_cache.json',{'records':cache})
    print(f'Frozen {len(selected)} generated candidates +12 controls; ready for six-replicate AF2')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--select',action='store_true'); a=p.parse_args()
    if a.prepare: prepare()
    if a.select: select()
