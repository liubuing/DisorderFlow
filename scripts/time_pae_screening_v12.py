"""Fresh same-device wall-clock comparison from candidate pool to AF2 outputs."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import read,freeze,digest
from scripts.run_multiscaffold_v2_af2 import atomic_json,windows_to_wsl
from scripts import resume_pae_dense_batched as transport
from scripts.strengthen_pae_v12 import feature_matrix

OUT=ROOT/'results/pae_timing_v12'
SOURCE=ROOT/'results/pae_public_pilot_v11'
WEIGHTS=ROOT/'results/pae_screening_v7/ridge_fixed10.npz'


def status(stage,**kw):
    atomic_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'protocol.json',{
        'classification':'single_paired_same_device_timing_on_exposed_v11_pool',
        'order':['screened','full'],'candidates':120,'screened':24,
        'teacher':'both models1/2 seeds7103/7111/7121 recycles3; unchanged sequence-only configuration',
        'boundary':'existing candidate pool through AF2 output; includes feature parsing, weight load, selection, job writing, worker startup/warmup, retries and output checksum checks; excludes upstream generation, shared Python imports and preflight weight hashing',
        'cache':'no historical AF2 output reuse; fresh output directory per arm; no resume within an interrupted arm',
        'limitations':'one pair; no repeated-run uncertainty; fixed order permits filesystem/thermal order effects; not GPU-active hours or full design-to-result speedup',
        'sources':{p:digest(SOURCE/p) for p in ['entities.json','holdout.json','pre_teacher_predictions.json']},
        'ridge_sha256':digest(WEIGHTS),
        'code':{p:digest(ROOT/p) for p in ['scripts/time_pae_screening_v12.py','scripts/strengthen_pae_v12.py','scripts/resume_pae_dense_batched.py','scripts/utils/af2_wsl_batch.py']},
    })


def run():
    protocol=read(OUT/'protocol.json')
    for p,sha in protocol['code'].items(): assert digest(ROOT/p)==sha
    for p,sha in protocol['sources'].items(): assert digest(SOURCE/p)==sha
    assert digest(WEIGHTS)==protocol['ridge_sha256']
    configs={}
    for m in (1,2):
        suffix='_model2' if m==2 else ''
        cfg=yaml.safe_load((ROOT/f'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2{suffix}.yml').read_text())['af2']
        weight=Path(os.environ['COLABFOLD_CACHE'])/'params'/Path(cfg['model_parameters']).name
        assert digest(weight)==cfg['model_parameters_sha256']
        configs[m]=cfg
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],capture_output=True,text=True,check=True).stdout
    freeze(OUT/'hardware.json',{'nvidia_smi':hardware})
    receipts={}
    for arm in protocol['order']:
        folder=OUT/arm
        if (folder/'receipt.json').exists():
            receipts[arm]=read(folder/'receipt.json'); continue
        if folder.exists(): raise RuntimeError('Incomplete timed arm: preserve it; a new explicitly recorded attempt is required')
        folder.mkdir()
        status('running',arm=arm,completed=0)
        start=time.perf_counter()
        entities=[e for e in read(SOURCE/'entities.json')['entities'] if e['entity_type']=='candidate']
        groups=read(SOURCE/'holdout.json')['components']
        if arm=='screened':
            x=feature_matrix(groups,entities)
            with np.load(WEIGHTS) as w: pred=((x-w['mean'])/w['scale'])@w['coef']+w['intercept']
            old=read(SOURCE/'pre_teacher_predictions.json')['predictions']
            np.testing.assert_allclose(pred,[old[e['entity_id']] for e in entities],atol=1e-7,rtol=0)
            selected=[]
            for c in groups:
                indices=[i for i,e in enumerate(entities) if e['component_id']==c['component_id']]
                selected += sorted(indices,key=lambda i:(pred[i],entities[i]['entity_id']))[:4]
            entities=[entities[i] for i in selected]
        selection_seconds=time.perf_counter()-start
        freeze(folder/'entities.json',{'entities':entities})
        transport.OUT=folder
        expected=len(entities)*6
        completed=0; inference_seconds=0; warmups=[]
        for m in (1,2):
            cfg=configs[m]; jobs=[]; paths={}
            for e in sorted(entities,key=lambda e:(len(e['heavy_sequence'])+len(e['light_sequence'])+len(e['antigen_sequence']),e['entity_id'])):
                for seed in (7103,7111,7121):
                    jid=f'{e["entity_id"]}|model{m}|seed{seed}'
                    stem=hashlib.sha256(jid.encode()).hexdigest()[:24]
                    pae=folder/'pae'/(stem+'.npz'); paths[jid]=pae
                    jobs.append(dict(id=jid,seq=e['heavy_sequence']+':'+e['light_sequence'],epi_seq=e['antigen_sequence'],seed=seed,recycle=3,
                        output_pdb=windows_to_wsl(folder/'pdb'/(stem+'.pdb')),output_pae=windows_to_wsl(pae)))
            path=folder/f'jobs_model{m}.jsonl'
            path.write_text(''.join(json.dumps(j)+'\n' for j in jobs),encoding='utf-8')
            command=f'cd {windows_to_wsl(ROOT)} && source {cfg["environment"]}/bin/activate && python scripts/utils/af2_wsl_batch.py --recycle 3 --model-number {m}'
            with path.open() as stdin,(folder/f'model{m}.stderr.log').open('a') as stderr,(folder/'results.jsonl').open('a') as logfile:
                worker=transport.BatchedWorker(['wsl.exe','-d',cfg['wsl_distribution'],'--','bash','-lc',command],stdin=stdin,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',errors='replace',bufsize=1)
                seen=set()
                for line in worker.stdout:
                    try: raw=json.loads(line)
                    except ValueError: continue
                    logfile.write(json.dumps(raw)+'\n'); logfile.flush()
                    if raw.get('status')=='warmup': warmups.append(raw)
                    if raw.get('id') not in paths: continue
                    jid=raw['id']
                    if not raw.get('success') or not paths[jid].exists() or digest(paths[jid])!=raw.get('pae_sha256'):
                        worker.terminate(); raise RuntimeError('AF2 result failed verification; timing arm incomplete')
                    if jid in seen: raise RuntimeError('Duplicate result')
                    seen.add(jid); completed+=1; inference_seconds+=raw['elapsed']
                    status('running',arm=arm,completed=completed,expected=expected,wall_seconds=time.perf_counter()-start)
                if worker.wait()!=0 or len(seen)!=len(jobs): raise RuntimeError('Incomplete worker')
        elapsed=time.perf_counter()-start
        assert completed==expected
        receipt=dict(arm=arm,candidates=len(entities),slots=completed,wall_seconds=elapsed,selection_seconds=selection_seconds,
            summed_slot_seconds=inference_seconds,warmups=warmups,protocol_sha256=digest(OUT/'protocol.json'))
        freeze(folder/'receipt.json',receipt); receipts[arm]=receipt
        print(json.dumps(receipt),flush=True)
    reduction=1-receipts['screened']['wall_seconds']/receipts['full']['wall_seconds']
    freeze(OUT/'comparison.json',dict(receipts=receipts,measured_wall_reduction=reduction,scope=protocol['boundary'],limitations=protocol['limitations']))
    status('complete',measured_wall_reduction=reduction)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run']); args=parser.parse_args()
    try: (prepare if args.stage=='prepare' else run)()
    except Exception as exc:
        if OUT.exists(): status('blocked',error=str(exc))
        raise
