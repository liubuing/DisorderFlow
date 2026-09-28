"""Per-batch GPU admission, ownership telemetry, bounded retry; no speedup claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import guard_pae_batch_v19 as guard
from scripts.run_guarded_abba_v16 import read,save,sha
OUT=ROOT/'results/pae_queued_v19'
SOURCE=ROOT/'results/pae_screen_batches_v18'

def status(stage,**kw):
    guard.write(OUT/'status.json',dict(stage=stage,wsl_pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))

def prepare():
    files=['scripts/run_pae_queued_v19.py','scripts/guard_pae_batch_v19.py','scripts/run_guarded_abba_v16.py','scripts/utils/af2_wsl_batch.py','modules/af2_jax_runner.py']
    files += ['configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2'+s+'.yml' for s in ['', '_model2']]
    save(OUT/'protocol.json',{'classification':'queued_full_screened_feasibility_not_continuous_timing','jobs':144,'batch_size':6,
        'code':{p:sha(ROOT/p) for p in files},'source_protocol_sha256':sha(SOURCE/'protocol.json'),
        'admission':'before EACH worker, no GPU compute PIDs, GPU utilization<20 and memory<3000MiB for3samples10secondsapart; log owners; never stop foreign processes',
        'retry':'at most2 fresh attempts perbatch for GPU memory guard or timeout only; no retry on verification failure; preserve allattempts; global24h deadline checked between workers',
        'guard':'unchanged18500MiB two samples; RAM2GiB;600s perworker; cannot guarantee host stability',
        'boundary':'fresh144 scientific inputs; queue/retries separated; not a paired cost estimate, no silent merging as continuous runtime'})
    for source in sorted(SOURCE.glob('model*_batch_*')):
        for attempt in (1,2):
            folder=OUT/source.name/f'attempt{attempt}'; folder.mkdir(parents=True)
            rows=[]
            for i,j in enumerate(map(json.loads,(source/'jobs.jsonl').read_text().splitlines())):
                base=f'/mnt/d/biological/DisorderFlow/results/pae_queued_v19/{source.name}/attempt{attempt}'
                rows.append({**j,'output_pae':f'{base}/pae/{i:02d}.npz','output_pdb':f'{base}/pdb/{i:02d}.pdb'})
            (folder/'jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in rows))
            save(folder/'protocol.json',{**read(source/'protocol.json'),'script_sha256':sha(Path(guard.__file__)),'jobs_sha256':sha(folder/'jobs.jsonl')})

def admission(deadline,batch,completed):
    idle=0
    with (OUT/'admission.jsonl').open('a') as log:
        while time.monotonic()<deadline:
            a=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            b=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            if a.returncode or b.returncode: raise RuntimeError('GPU ownership telemetry unavailable')
            util,mem=map(float,a.stdout.strip().split(','))
            idle=idle+1 if util<20 and mem<3000 and not b.stdout.strip() else 0
            row={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'batch':batch,'util':util,'memory_mib':mem,'owners':b.stdout.strip(),'consecutive_idle':idle}
            log.write(json.dumps(row)+'\n');log.flush()
            status('waiting_for_gpu',batch=batch,completed=completed,expected=144,gpu_memory_mib=mem,gpu_utilization=util,owners=b.stdout.strip())
            if idle>=3:return True
            time.sleep(10)
    return False

def run():
    if os.name!='posix':raise RuntimeError('WSL only')
    if (OUT/'status.json').exists():raise RuntimeError('Existing run; do not overwrite')
    p=read(OUT/'protocol.json')
    for path,digest in p['code'].items():assert sha(ROOT/path)==digest
    import yaml
    for suffix in ['', '_model2']:
        cfg=yaml.safe_load((ROOT/f'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2{suffix}.yml').read_text())['af2']
        assert sha(Path('/mnt/d/DisorderFlowRuntime/cache/colabfold/params')/Path(cfg['model_parameters']).name)==cfg['model_parameters_sha256']
    start=time.monotonic();deadline=start+86400;completed=0;attempts=[];queue=0
    for batch in sorted(OUT.glob('model*_batch_*')):
        for attempt in (1,2):
            waitstart=time.monotonic()
            ready=admission(deadline,batch.name,completed);queue+=time.monotonic()-waitstart
            if not ready:status('stopped',reason='24h_admission_deadline',completed=completed);return
            folder=batch/f'attempt{attempt}'
            status('running',batch=batch.name,attempt=attempt,completed=completed,expected=144)
            guard.OUT=folder;guard.run();r=read(folder/'status.json');attempts.append({'batch':batch.name,'attempt':attempt,**r})
            guard.write(OUT/'attempts.json',{'attempts':attempts,'queue_seconds':queue})
            if r['verified']:completed+=6;break
            if attempt==2 or r['reason'] not in ['gpu_memory_guard','timeout']:
                status('stopped',reason=r['reason'] or 'verification_failed',batch=batch.name,completed=completed);return
    result={'completed':completed,'verified':True,'queue_seconds':queue,'elapsed_seconds':time.monotonic()-start,
        'sum_attempt_seconds':sum(r['wall_seconds'] for r in attempts),'attempts':attempts,'boundary':p['boundary']}
    save(OUT/'result.json',result);status('complete',completed=completed,expected=144)
    (ROOT/'docs/PAE_QUEUED_V19_RESULTS.md').write_text(f"# v19排队运行结果\n\n144/144验证通过。排队{queue:.1f}秒，含失败尝试执行{result['sum_attempt_seconds']:.1f}秒。\n\n队列等待与重试已分开记录；不作为连续配对加速结果。\n",encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    try:(prepare if args.stage=='prepare' else run)()
    except Exception as exc:
        if OUT.exists():status('blocked',error=str(exc))
        raise
