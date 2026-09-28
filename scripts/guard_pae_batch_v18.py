"""Bounded WSL smoke test. Kill only the owned worker group on guard failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pae_guard_smoke_v15'


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists(): raise RuntimeError('Existing protocol; do not overwrite')
    old=json.loads((ROOT/'results/pae_worker_probe_v14/protocol.json').read_text())
    jobs=[]
    for i,j in enumerate(old['jobs'][:3]):
        base='/mnt/d/biological/DisorderFlow/results/pae_guard_smoke_v15'
        jobs.append({**j,'output_pae':f'{base}/pae/{i:02d}.npz','output_pdb':f'{base}/pdb/{i:02d}.pdb'})
    (OUT/'jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    write(OUT/'protocol.json',{'classification':'bounded_runtime_smoke_not_timing_comparison',
        'jobs':3,'model':2,'seed':7103,'recycle':3,'timeout_seconds':300,
        'gpu_memory_mib_stop':18500,'consecutive_samples':2,'poll_seconds':2,
        'minimum_available_ram_kib':2097152,
        'guard':'GPU total memory threshold is conservative and may stop on other applications; not proof of AF2 memory ownership; guard cannot guarantee host stability',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'worker_sha256':hashlib.sha256((ROOT/'scripts/utils/af2_wsl_batch.py').read_bytes()).hexdigest(),
        'jobs_sha256':hashlib.sha256((OUT/'jobs.jsonl').read_bytes()).hexdigest()})


def run():
    if os.name!='posix': raise RuntimeError('Run guard inside WSL')
    p=json.loads((OUT/'protocol.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==p['script_sha256']
    assert hashlib.sha256((ROOT/'scripts/utils/af2_wsl_batch.py').read_bytes()).hexdigest()==p['worker_sha256']
    assert hashlib.sha256((OUT/'jobs.jsonl').read_bytes()).hexdigest()==p['jobs_sha256']
    if (OUT/'status.json').exists(): raise RuntimeError('No silent resume of smoke test')
    start=time.monotonic(); reason=None; high=0
    with (OUT/'jobs.jsonl').open() as stdin,(OUT/'worker.jsonl').open('w') as stdout,(OUT/'worker.stderr.log').open('w') as stderr,(OUT/'telemetry.jsonl').open('w') as log:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts/utils/af2_wsl_batch.py'),'--model-number',str(p['model']),'--recycle','3'],stdin=stdin,stdout=stdout,stderr=stderr,cwd=ROOT,start_new_session=True)
        write(OUT/'status.json',{'stage':'running','wsl_guard_pid':os.getpid(),'worker_pid':child.pid})
        try:
            while child.poll() is None:
                elapsed=time.monotonic()-start
                gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                if gpu.returncode: reason='telemetry_unavailable'; break
                fields=gpu.stdout.splitlines()[0].split(','); memory=float(fields[0])
                available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
                log.write(json.dumps({'timestamp':time.time(),'elapsed':elapsed,'gpu':gpu.stdout.strip(),'available_ram_kib':available})+'\n'); log.flush()
                high=high+1 if memory>=p['gpu_memory_mib_stop'] else 0
                if high>=p['consecutive_samples']: reason='gpu_memory_guard'; break
                if available<p['minimum_available_ram_kib']: reason='host_ram_guard'; break
                if elapsed>p['timeout_seconds']: reason='timeout'; break
                time.sleep(p['poll_seconds'])
        except Exception as exc: reason='monitor_error:'+str(exc)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
    rows=[]
    for line in (OUT/'worker.jsonl').read_text().splitlines():
        try: r=json.loads(line)
        except ValueError: continue
        if r.get('id'): rows.append(r)
    valid=reason is None and child.returncode==0 and len(rows)==p['jobs'] and {r['id'] for r in rows}=={json.loads(line)['id'] for line in (OUT/'jobs.jsonl').read_text().splitlines()}
    for r in rows:
        valid=valid and r.get('success',False) and Path(r.get('pae_path','missing')).is_file()
        if valid: valid=hashlib.sha256(Path(r['pae_path']).read_bytes()).hexdigest()==r['pae_sha256']
    write(OUT/'status.json',{'stage':'complete' if valid else 'stopped','reason':reason,'exit_code':child.returncode,
        'completed':len(rows),'verified':bool(valid),'wall_seconds':time.monotonic()-start,'boundary':'guarded batch only; evaluate complete ABBA before claims'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run']); args=parser.parse_args()
    (prepare if args.stage=='prepare' else run)()


