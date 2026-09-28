"""Fixed ABBA execution probe; no changes to AF2 parameters or candidates."""
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import read,freeze,digest
from scripts.run_multiscaffold_v2_af2 import atomic_json,windows_to_wsl
from scripts import remeasure_pae_screen_v13 as telemetry
OUT=ROOT/'results/pae_worker_probe_v14'
SOURCE=ROOT/'results/pae_timing_v13/screened'


def status(stage,**kw):
    atomic_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))


def prepare():
    jobs=[json.loads(s) for s in (SOURCE/'jobs_model2.jsonl').read_text().splitlines()]
    jobs=[j for j in jobs if j['seed']==7103]
    assert len(jobs)==24
    freeze(OUT/'protocol.json',{'classification':'post_hoc_engineering_probe_not_biological_validation',
        'jobs':jobs,'order':['A1','B1','B2','A2'],'A':'one24job worker','B':'four6job workers',
        'selection':'all24 frozen screened candidates, seed7103, model2, recycle3, same sorted order each unit',
        'boundary':'walltime includes worker startup/warmup and output verification; no retries or cache reuse',
        'limits':'24jobs may not reproduce72job failure; ABBA only two repeats percondition, no robust statistical claim; GPU telemetry overhead shared',
        'script_sha256':digest(Path(__file__)),'worker_sha256':digest(ROOT/'scripts/utils/af2_wsl_batch.py'),
        'monitor_sha256':digest(ROOT/'scripts/remeasure_pae_screen_v13.py'),
        'source_sha256':digest(SOURCE/'jobs_model2.jsonl')})


def run():
    p=read(OUT/'protocol.json')
    assert digest(Path(__file__))==p['script_sha256']
    assert digest(ROOT/'scripts/utils/af2_wsl_batch.py')==p['worker_sha256']
    assert digest(ROOT/'scripts/remeasure_pae_screen_v13.py')==p['monitor_sha256']
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2_model2.yml').read_text())['af2']
    assert digest(Path(os.environ['COLABFOLD_CACHE'])/'params'/Path(cfg['model_parameters']).name)==cfg['model_parameters_sha256']
    telemetry.OUT=OUT
    stop=threading.Event(); watcher=threading.Thread(target=telemetry.monitor,args=(stop,),daemon=True); watcher.start()
    receipts={}
    try:
        for unit in p['order']:
            folder=OUT/unit
            if folder.exists(): raise RuntimeError('Existing probe unit: preserve; do not silently resume timing')
            folder.mkdir(); start=time.perf_counter(); completed=0
            size=24 if unit.startswith('A') else 6
            jobs=[]; output_paths={}
            for index,old in enumerate(p['jobs']):
                pae=folder/'pae'/f'{index:02d}.npz'
                jobs.append({**old,'output_pae':windows_to_wsl(pae),'output_pdb':windows_to_wsl(folder/'pdb'/f'{index:02d}.pdb')})
                output_paths[old['id']]=pae
            status('running',unit=unit,completed=0,total=24)
            for offset in range(0,24,size):
                batch=jobs[offset:offset+size]
                path=folder/f'jobs_{offset:02d}.jsonl'; path.write_text(''.join(json.dumps(j)+'\n' for j in batch))
                command=f'cd {windows_to_wsl(ROOT)} && source {cfg["environment"]}/bin/activate && python scripts/utils/af2_wsl_batch.py --recycle 3 --model-number 2 < '+shlex.quote(windows_to_wsl(path))
                with (folder/'stderr.log').open('a') as stderr,(folder/'events.jsonl').open('a') as output:
                    process=subprocess.Popen(['wsl.exe','-d',cfg['wsl_distribution'],'--','bash','-lc',command],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',errors='replace')
                    seen=set()
                    try:
                        for line in process.stdout:
                            try: r=json.loads(line)
                            except ValueError: r={'raw':line.strip()}
                            output.write(json.dumps({'received_at':time.time(),'result':r})+'\n'); output.flush()
                            if r.get('id') not in output_paths: continue
                            assert r['id'] in {j['id'] for j in batch} and r['id'] not in seen
                            assert r.get('success') and digest(output_paths[r['id']])==r['pae_sha256']
                            seen.add(r['id']); completed+=1
                            status('running',unit=unit,completed=completed,total=24,wall_seconds=time.perf_counter()-start)
                        assert process.wait()==0 and len(seen)==len(batch)
                    finally:
                        if process.poll() is None: process.terminate(); process.wait()
            receipt={'unit':unit,'slots':completed,'wall_seconds':time.perf_counter()-start,'workers':24//size}
            freeze(folder/'receipt.json',receipt); receipts[unit]=receipt
        freeze(OUT/'comparison.json',{'receipts':receipts,'mean_A_seconds':sum(receipts[k]['wall_seconds'] for k in ['A1','A2'])/2,
            'mean_B_seconds':sum(receipts[k]['wall_seconds'] for k in ['B1','B2'])/2,
            'boundary':p['limits']})
        status('complete')
    finally: stop.set(); watcher.join(timeout=15)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run']); args=parser.parse_args()
    try: (prepare if args.stage=='prepare' else run)()
    except Exception as exc:
        if OUT.exists(): status('blocked',error=str(exc))
        raise
