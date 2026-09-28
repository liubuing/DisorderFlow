"""Fresh screened-arm measurement; reference is the completed post-reboot full arm."""
import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import time_pae_screening_v12 as runner
from scripts.prepare_pae_dense_v6 import read,freeze,digest

OUT=ROOT/'results/pae_timing_v13'
OLD=ROOT/'results/pae_timing_v12'


def prepare():
    original=read(OLD/'protocol.json')
    for p,sha in original['code'].items(): assert digest(ROOT/p)==sha
    reference=read(OLD/'full/receipt.json')
    assert reference['slots']==720
    protocol={**original,
        'classification':'post_reboot_screen_remeasurement_with_previously_completed_full_reference',
        'reference_full_receipt_sha256':digest(OLD/'full/receipt.json'),
        'reference_full_results_sha256':digest(OLD/'full/results.jsonl'),
        'reference_hardware_sha256':digest(OLD/'hardware.json'),
        'order':['screened','full'],
        'full_arm':'copy exact completed post-reboot receipt, no new full run; not a contemporaneous randomized pair',
        'limitations':'screen remeasured later than full reference; single measurement; no full-arm GPU telemetry; comparable performance must be assessed, not assumed; no significance or end-to-end generation speedup claim',
        'telemetry':'nvidia-smi GPU utilization, memory, temperature, power every30seconds; monitoring overhead included in screen walltime',
        'code':{**original['code'],'scripts/remeasure_pae_screen_v13.py':digest(Path(__file__))}}
    freeze(OUT/'protocol.json',protocol)
    freeze(OUT/'full/receipt.json',reference)
    freeze(OUT/'full/reference.json',{'source':'results/pae_timing_v12/full','receipt_sha256':digest(OLD/'full/receipt.json'),
        'boundary':'receipt reused for comparison only; all screened AF2 predictions rerun'})


def monitor(stop):
    with (OUT/'gpu_telemetry.jsonl').open('a') as out:
        while not stop.is_set():
            try:
                p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
                row={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'returncode':p.returncode,'data':p.stdout.strip(),'error':p.stderr.strip()}
            except Exception as exc: row={'error':str(exc)}
            out.write(json.dumps(row)+'\n'); out.flush()
            stop.wait(30)


def records(folder):
    result={}
    for line in (folder/'results.jsonl').read_text().splitlines():
        r=json.loads(line)
        if not r.get('id'): continue
        assert r['success'] and r['id'] not in result
        assert digest(folder/'pae'/Path(r['pae_path']).name)==r['pae_sha256']
        result[r['id']]=r
    return result


def run():
    p=read(OUT/'protocol.json')
    assert digest(OLD/'full/receipt.json')==p['reference_full_receipt_sha256']
    assert digest(OLD/'full/results.jsonl')==p['reference_full_results_sha256']
    assert digest(OLD/'hardware.json')==p['reference_hardware_sha256']
    hw=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],capture_output=True,text=True,check=True).stdout
    assert hw==read(OLD/'hardware.json')['nvidia_smi']
    full=records(OLD/'full'); assert len(full)==720
    stop=threading.Event(); thread=threading.Thread(target=monitor,args=(stop,),daemon=True)
    thread.start()
    runner.OUT=OUT
    try: runner.run()
    finally: stop.set(); thread.join(timeout=15)
    screened=records(OUT/'screened'); assert len(screened)==144
    assert read(OUT/'screened/entities.json')==read(OLD/'screened/entities.json')
    ratios=[v['elapsed']/full[k]['elapsed'] for k,v in screened.items()]
    comparison=read(OUT/'comparison.json')
    receipt={'verified_screen_slots':144,'verified_reference_full_slots':720,
        'same_job_screen_over_full_elapsed_ratio_median':statistics.median(ratios),
        'same_job_ratio_min':min(ratios),'same_job_ratio_max':max(ratios),
        'measured_wall_reduction':comparison['measured_wall_reduction'],
        'interpretation':'descriptive later screened run against earlier post-reboot full reference; not a simultaneous controlled pair',
        'comparison_sha256':digest(OUT/'comparison.json')}
    freeze(OUT/'verification.json',receipt)
    a=comparison['receipts']['screened']['wall_seconds']; b=comparison['receipts']['full']['wall_seconds']
    lines=['# PAE筛选路线重新计时 v13','',f'筛选路线：{a:.1f}秒，144任务；重启后既有全量参考：{b:.1f}秒，720任务。',
        f'描述性墙钟减少比例：{1-a/b:.1%}。相同144任务的筛选/全量单任务耗时比中位数：{statistics.median(ratios):.3f}。',
        '', '两组输出哈希与任务数校验通过，筛选候选与原冻结名单完全一致。',
        '', '限制：全量参考早于本次筛选重测，只有筛选重测具有GPU遥测；没有同期随机交叉实验。不能仅凭硬件相同认定负载相同，不能将本次重测隐藏为原始连续配对，也不能把计算成本结果解释为结合或临床效果。','']
    (ROOT/'docs/PAE_TIMING_V13_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run']); args=parser.parse_args()
    try: (prepare if args.stage=='prepare' else run)()
    except Exception as exc:
        if OUT.exists():
            runner.OUT=OUT; runner.status('blocked',error=str(exc))
        raise
