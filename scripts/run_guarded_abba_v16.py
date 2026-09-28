"""Guarded replication of the interrupted v14 engineering probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import guard_pae_batch_v16 as guard
OUT=ROOT/'results/pae_guarded_abba_v16'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise RuntimeError(f'Refusing overwrite: {path}')
    guard.write(path,value)

def prepare():
    jobs=read(ROOT/'results/pae_worker_probe_v14/protocol.json')['jobs']
    assert len(jobs)==24
    code={p:sha(ROOT/p) for p in ['scripts/run_guarded_abba_v16.py','scripts/guard_pae_batch_v16.py','scripts/utils/af2_wsl_batch.py','modules/af2_jax_runner.py']}
    save(OUT/'protocol.json',{'classification':'guarded_ABBA_engineering_probe','order':['A1','B1','B2','A2'],
        'A':'24jobs one freshworker','B':'24jobs four fresh6job workers','code':code,
        'boundary':'96 new predictions; same24 candidates/model2/seed7103/recycle3; two repeats percondition, no biological validation; failed guard stops whole experiment with no retries',
        'guard':'18500MiB GPU two consecutive2sec samples; availableRAM2GiB; worker600s; total experiment2400s; no claim of guaranteed host stability',
        'timing':'includes perworker startup/warmup/guard overhead; different from original unguarded walltime; compare only within this experiment'})
    for unit in ['A1','B1','B2','A2']:
        size=24 if unit.startswith('A') else 6
        for offset in range(0,24,size):
            folder=OUT/unit/f'batch_{offset:02d}'; folder.mkdir(parents=True)
            batch=[]
            for i,j in enumerate(jobs[offset:offset+size],offset):
                base=f'/mnt/d/biological/DisorderFlow/results/pae_guarded_abba_v16/{unit}/batch_{offset:02d}'
                batch.append({**j,'output_pae':f'{base}/pae/{i:02d}.npz','output_pdb':f'{base}/pdb/{i:02d}.pdb'})
            (folder/'jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in batch))
            save(folder/'protocol.json',{'jobs':len(batch),'timeout_seconds':600,'gpu_memory_mib_stop':18500,'consecutive_samples':2,'poll_seconds':2,
                'minimum_available_ram_kib':2097152,'script_sha256':sha(Path(guard.__file__)),
                'worker_sha256':sha(ROOT/'scripts/utils/af2_wsl_batch.py'),'jobs_sha256':sha(folder/'jobs.jsonl')})

def status(stage,**kw): guard.write(OUT/'status.json',dict(stage=stage,wsl_pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))

def run():
    if os.name!='posix': raise RuntimeError('WSL only')
    p=read(OUT/'protocol.json')
    for path,digest in p['code'].items(): assert sha(ROOT/path)==digest
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2_model2.yml').read_text())['af2']
    assert sha(Path('/mnt/d/DisorderFlowRuntime/cache/colabfold/params')/Path(cfg['model_parameters']).name)==cfg['model_parameters_sha256']
    receipts={}; count=0; overall=time.monotonic()
    for unit in p['order']:
        start=time.monotonic(); batches=sorted((OUT/unit).glob('batch_*'))
        for folder in batches:
            if time.monotonic()-overall>2400: status('stopped',reason='experiment_timeout',completed=count); return
            status('running',unit=unit,batch=folder.name,completed=count,expected=96)
            guard.OUT=folder; guard.run()
            r=read(folder/'status.json'); count+=r['completed']
            if not r['verified']:
                status('stopped',unit=unit,batch=folder.name,completed=count,reason=r['reason'] or 'batch_verification_failed'); return
        receipt={'unit':unit,'slots':24,'wall_seconds':time.monotonic()-start,'workers':len(batches)}
        save(OUT/unit/'receipt.json',receipt); receipts[unit]=receipt
    a=sum(receipts[k]['wall_seconds'] for k in ['A1','A2'])/2
    b=sum(receipts[k]['wall_seconds'] for k in ['B1','B2'])/2
    save(OUT/'comparison.json',{'receipts':receipts,'mean_A_seconds':a,'mean_B_seconds':b,'B_wall_reduction_relative_A':1-b/a,
        'boundary':p['boundary'],'interpretation':'24job guarded workload only; no extrapolation to72/720job stability'})
    status('complete',completed=count)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run']); args=parser.parse_args()
    try: (prepare if args.stage=='prepare' else run)()
    except Exception as exc:
        if OUT.exists(): status('blocked',error=str(exc))
        raise
