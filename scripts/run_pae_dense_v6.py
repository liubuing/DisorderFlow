"""Resumable AF2 teacher worker; writes progress for a long local GPU run."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import yaml
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import OUT,read,digest,freeze
from scripts.run_multiscaffold_v2_af2 import windows_to_wsl,atomic_json


def key(e,model,seed):
    return (e['component_id'],e['heavy_sequence'],e['light_sequence'],e['antigen_sequence'],model,seed)


def target(path,e):
    from scripts.evaluate_mpnn_likelihood_baseline import locate_h3
    # Frozen full-heavy H3 string positions; no teacher geometry in selection.
    h=e['heavy_sequence']; h3=e['h3_sequence']; start=h.find(h3)
    if start<0 or h.count(h3)!=1: raise ValueError('Ambiguous H3 label indices')
    ag=len(h)+len(e['light_sequence'])
    with np.load(path) as z: matrix=z['pae']
    assert matrix.shape==(ag+len(e['antigen_sequence']),)*2
    return float(matrix[start:start+len(h3),ag:].mean()/31)


def main():
    started=time.time(); entities=read(OUT/'entities.json')['entities']
    protocol=read(OUT/'protocol.json')
    code_manifest=read(OUT/'execution_code_manifest.json')
    for path,sha in code_manifest['files'].items():
        if digest(ROOT/path)!=sha: raise ValueError('Execution code changed after launch freeze')
    assert all(e['component_id'].startswith('EXT') for e in entities)
    cache={}
    for r in read(OUT/'historical_cache.json')['records']:
        old=r['result']; p=ROOT/old['pae_npz']
        if digest(p)!=old['pae_sha256']: raise ValueError('Historical PAE checksum mismatch')
        cache[key(r['entity'],r['model'],old['af2_seed'])]={'pae_path':old['pae_npz'],'pae_sha256':old['pae_sha256'],
            'elapsed':old['elapsed'],'reused_historical':True}
    logs=OUT/'new_results.jsonl'
    existing=[]
    if logs.exists():
        for line in logs.read_text().splitlines():
            record=json.loads(line); existing.append(record)
            if record.get('success'):
                e=next(e for e in entities if e['entity_id']==record['entity_id'])
                if digest(ROOT/record['pae_path'])!=record['pae_sha256']: raise ValueError('New cache checksum mismatch')
                cache[key(e,record['model'],record['seed'])]=record
    jobs_by_model={}; mappings={}; new_count=0; reused=0
    for model in [1,2]:
        jobs=[]; seen=set()
        old_cfg=ROOT/f'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2{"_model2" if model==2 else ""}.yml'
        cfg=yaml.safe_load(old_cfg.read_text()); af2=cfg['af2']
        assert af2['seeds']==[7103,7111,7121] and af2['recycles']==3
        weights=Path(os.environ['COLABFOLD_CACHE'])/'params'/Path(af2['model_parameters']).name
        assert digest(weights)==af2['model_parameters_sha256']
        for e in sorted(entities,key=lambda e:(len(e['heavy_sequence'])+len(e['light_sequence'])+len(e['antigen_sequence']),e['entity_id'])):
            for seed in af2['seeds']:
                k=key(e,model,seed)
                if k in seen: continue
                seen.add(k)
                if k in cache: reused+=1; continue
                identifier=f"{e['entity_id']}|model{model}|seed{seed}"
                import hashlib
                stem=hashlib.sha256(identifier.encode()).hexdigest()[:24]
                pdb=OUT/'pdb'/f'{stem}.pdb'; pae=OUT/'pae'/f'{stem}.npz'
                mappings[identifier]=(e,model,seed,pae)
                jobs.append({'id':identifier,'seq':e['heavy_sequence']+':'+e['light_sequence'],
                    'epi_seq':e['antigen_sequence'],'seed':seed,'recycle':3,
                    'output_pdb':windows_to_wsl(pdb),'output_pae':windows_to_wsl(pae)})
        jobs_by_model[model]=(jobs,af2)
    total=sum(len(x[0]) for x in jobs_by_model.values())
    def progress(status,**extra):
        atomic_json(OUT/'progress.json',{'status':status,'pid':os.getpid(),'entities':len(entities),
            'new_jobs_this_run':total,'new_results_this_run':new_count,'cached_unique_slots_at_start':reused,
            'seconds_this_run':time.time()-started,**extra})
    freeze(OUT/'run_contract.json',{'protocol_sha256':digest(OUT/'protocol.json'),'entities_sha256':digest(OUT/'entities.json'),
        'worker_sha256':digest(Path(__file__)),'teacher_worker_sha256':digest(ROOT/'scripts/utils/af2_wsl_batch.py'),
        'meaning':'Models1/2, seeds7103/7111/7121, unchanged old protocol; same six exposed components'})
    progress('running')
    for model,(jobs,af2) in jobs_by_model.items():
        if not jobs: continue
        path=OUT/f'jobs_model{model}.jsonl'
        path.write_text(''.join(json.dumps(j)+'\n' for j in jobs))
        command=f'cd {windows_to_wsl(ROOT)} && source {af2["environment"]}/bin/activate && python scripts/utils/af2_wsl_batch.py --recycle 3 --model-number {model}'
        failures=0
        with path.open('r') as stdin, (OUT/f'worker_model{model}.stderr.log').open('a') as stderr,logs.open('a') as output:
            process=subprocess.Popen(['wsl.exe','-d',af2['wsl_distribution'],'--','bash','-lc',command],stdin=stdin,
                stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',errors='replace',bufsize=1)
            progress('running',model=model,worker_pid=process.pid)
            for line in process.stdout:
                try: r=json.loads(line)
                except ValueError: continue
                if r.get('id') not in mappings:
                    print(line.strip(),flush=True); continue
                e,m,seed,pae=mappings[r['id']]
                success=bool(r.get('success')) and pae.exists() and digest(pae)==r.get('pae_sha256')
                record={'entity_id':e['entity_id'],'model':m,'seed':seed,'success':success,
                    'pae_path':str(pae.relative_to(ROOT)),'pae_sha256':r.get('pae_sha256'),'elapsed':r.get('elapsed'),
                    'reused_historical':False}
                if success: cache[key(e,m,seed)]=record
                output.write(json.dumps(record)+'\n'); output.flush(); new_count+=1
                failures=0 if success else failures+1
                progress('running',model=model,worker_pid=process.pid,latest_success=success)
                print(f'AF2 new result {new_count}/{total}: {success}',flush=True)
                if failures>=3:
                    process.terminate(); process.wait(); progress('blocked_consecutive_worker_failures',model=model)
                    return
            code=process.wait()
            if code:
                progress('blocked_worker_exit',model=model,exit_code=code); return
    records=[]; incomplete=[]
    for e in entities:
        observations=[]
        for m in [1,2]:
            for seed in [7103,7111,7121]:
                r=cache.get(key(e,m,seed))
                if r: observations.append({'model':m,'seed':seed,'target':target(ROOT/r['pae_path'],e),**r})
        row={**e,'observations':observations,'complete':len(observations)==6}
        if row['complete']: row['target']=float(np.mean([r['target'] for r in observations]))
        else: incomplete.append(e['entity_id'])
        records.append(row)
    atomic_json(OUT/'teacher_labels.json',{'rows':records,'incomplete_entities':incomplete,
        'status':'complete' if not incomplete else 'incomplete','independent_validation':False})
    progress('teacher_complete' if not incomplete else 'teacher_incomplete',incomplete_entities=len(incomplete))
    if not incomplete:
        completed=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_pae_dense_v6.py')],cwd=ROOT)
        progress('evaluation_complete' if completed.returncode==0 else 'evaluation_failed',evaluation_exit_code=completed.returncode)


if __name__=='__main__': main()
