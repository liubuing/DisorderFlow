"""Bounded preflight wait for this one experiment, not a recurring scheduler."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pae_screen_batches_v18'

def main():
    start=time.monotonic(); idle=0
    with (OUT/'preflight.jsonl').open('a') as log:
        while time.monotonic()-start<1800:
            p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            fields=p.stdout.strip().split(',')
            if p.returncode or len(fields)!=2: raise RuntimeError('Unable to verify GPU idle state')
            util,memory=map(float,fields); idle=idle+1 if util<20 and memory<3000 else 0
            row={'stage':'waiting_for_idle','time':time.strftime('%Y-%m-%d %H:%M:%S'),'gpu_utilization':util,'gpu_memory_mib':memory,'consecutive_idle_samples':idle}
            log.write(json.dumps(row)+'\n'); log.flush()
            (OUT/'launch_status.json').write_text(json.dumps(row,indent=2))
            if idle>=3:
                (OUT/'launch_status.json').write_text(json.dumps({'stage':'launched','time':time.strftime('%Y-%m-%d %H:%M:%S')}))
                code=subprocess.call([sys.executable,str(ROOT/'scripts/run_pae_screen_batches_v18.py'),'run'],cwd=ROOT)
                (OUT/'launch_status.json').write_text(json.dumps({'stage':'worker_exited','exit_code':code,'result_status_file':'status.json'}))
                return
            time.sleep(10)
    (OUT/'launch_status.json').write_text(json.dumps({'stage':'not_started_gpu_busy','wait_seconds':time.monotonic()-start}))

if __name__=='__main__': main()
