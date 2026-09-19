"""Transport-only amendment: bounded AF2 workers and WSL-local stdin redirection.

Frozen model, seed, recycle, selection and evaluation code are unchanged.
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_pae_dense_v6 as original
from scripts.prepare_pae_dense_v6 import OUT,freeze,digest
from scripts.run_multiscaffold_v2_af2 import windows_to_wsl,atomic_json

REAL_POPEN=subprocess.Popen
BATCH_SIZE=96


def batches(items,size=BATCH_SIZE):
    return [items[i:i+size] for i in range(0,len(items),size)]


class BatchedWorker:
    def __init__(self,args,**kwargs):
        self.args=args; self.kwargs=kwargs
        self.jobs=[json.loads(line) for line in kwargs['stdin'] if line.strip()]
        self.pid=None; self.process=None; self.returncode=None; self.stopped=False
        self.stdout=self.stream()

    def stream(self):
        tag='model2' if self.args[-1].endswith('--model-number 2') else 'model1'
        for number,group in enumerate(batches(self.jobs)):
            pending=group
            for attempt in range(3):
                if self.stopped: self.returncode=-1; return
                path=OUT/f'{tag}_bounded_{number:03d}_{attempt}.jsonl'
                path.write_text(''.join(json.dumps(j)+'\n' for j in pending),encoding='utf-8')
                args=list(self.args)
                args[-1]+=' < '+shlex.quote(windows_to_wsl(path))
                kwargs=dict(self.kwargs); kwargs['stdin']=subprocess.DEVNULL
                self.process=REAL_POPEN(args,**kwargs); self.pid=self.process.pid
                atomic_json(OUT/'batch_transport.json',{'model':tag,'batch':number,'batch_size':len(pending),
                    'attempt':attempt,'worker_pid':self.pid,'status':'running'})
                delivered=set()
                for line in self.process.stdout:
                    try: result=json.loads(line)
                    except ValueError: result={}
                    if result.get('id') is not None: delivered.add(result['id'])
                    yield line
                code=self.process.wait()
                pending=[job for job in pending if job['id'] not in delivered]
                if not pending: break
                if attempt==2:
                    self.returncode=code or 1
                    atomic_json(OUT/'batch_transport.json',{'model':tag,'batch':number,'status':'failed',
                        'remaining_jobs':len(pending),'exit_code':self.returncode})
                    return
            else: self.returncode=1; return
        self.returncode=0
        atomic_json(OUT/'batch_transport.json',{'model':tag,'status':'complete'})

    def wait(self): return self.returncode if self.returncode is not None else 1

    def terminate(self):
        self.stopped=True
        if self.process and self.process.poll() is None: self.process.terminate()
        if self.process: self.returncode=self.process.wait()


def popen(args,*positional,**kwargs):
    if isinstance(args,list) and args and args[0]=='wsl.exe' and 'scripts/utils/af2_wsl_batch.py' in args[-1]:
        if positional: raise ValueError('Unexpected positional Popen args')
        return BatchedWorker(args,**kwargs)
    return REAL_POPEN(args,*positional,**kwargs)


if __name__=='__main__':
    freeze(OUT/'transport_amendment_01.json',{'classification':'execution_only_no_scientific_protocol_change',
        'reason':'two exit9 interruptions with WSL delayed-stdin errors and unmount events; no OOM cause established',
        'batch_size':BATCH_SIZE,'max_attempts_per_batch':3,'stdin':'WSL shell opens job file; Windows stdin disabled',
        'reuse':'original checksum-based resume skips successful saved slots',
        'wrapper_sha256':digest(Path(__file__)),'original_worker_sha256':digest(Path(original.__file__))})
    atomic_json(OUT/'pipeline_status.json',{'stage':'af2','execution':'bounded_workers_transport_amendment_01'})
    original.subprocess.Popen=popen
    try:
        original.main()
        status=json.loads((OUT/'progress.json').read_text())
        atomic_json(OUT/'pipeline_status.json',{'stage':status['status'],'final_progress':status})
    except Exception as exc:
        atomic_json(OUT/'pipeline_status.json',{'stage':'blocked','error':str(exc)})
        raise
