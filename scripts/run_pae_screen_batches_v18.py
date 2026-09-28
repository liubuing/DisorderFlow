"""Full frozen screened cohort with six jobs per guarded worker."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import guard_pae_batch_v18 as guard
from scripts.run_guarded_abba_v16 import read,save,sha
OUT=ROOT/'results/pae_screen_batches_v18'
SOURCE=ROOT/'results/pae_timing_v13/screened'

def prepare():
    assert read(ROOT/'results/pae_short_batches_v17/result.json')['all24_verified']
    paths=['scripts/run_pae_screen_batches_v18.py','scripts/guard_pae_batch_v18.py','scripts/run_guarded_abba_v16.py','scripts/utils/af2_wsl_batch.py','modules/af2_jax_runner.py']
    for m in (1,2): paths.append('configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2'+('_model2' if m==2 else '')+'.yml')
    save(OUT/'protocol.json',{'classification':'guarded_full_screened_cohort_feasibility','jobs':144,'workers':24,
        'code':{p:sha(ROOT/p) for p in paths},'source':{f:sha(SOURCE/f) for f in ['jobs_model1.jsonl','jobs_model2.jsonl','entities.json']},
        'guard':'same18500MiB two consecutive2s samples; availableRAM2GiB; worker600s; stop all on any failure; no retry or relaxed threshold',
        'boundary':'same24 screened candidates x2models x3seeds, recycle3; fresh outputs, sixjobs perworker; no concurrent full comparator; no new independent biological validation',
        'cost':'includes worker startup/warmup/guard and output verification; excludes upstream candidate generation and prior screening; not end-to-end screening speedup'})
    for model in (1,2):
        jobs=[json.loads(l) for l in (SOURCE/f'jobs_model{model}.jsonl').read_text().splitlines()]
        assert len(jobs)==72
        for offset in range(0,72,6):
            name=f'model{model}_batch_{offset:02d}'; folder=OUT/name; folder.mkdir()
            batch=[]
            for i,j in enumerate(jobs[offset:offset+6],offset):
                base=f'/mnt/d/biological/DisorderFlow/results/pae_screen_batches_v18/{name}'
                batch.append({**j,'output_pdb':f'{base}/pdb/{i:02d}.pdb','output_pae':f'{base}/pae/{i:02d}.npz'})
            (folder/'jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in batch))
            save(folder/'protocol.json',{'jobs':6,'model':model,'timeout_seconds':600,'gpu_memory_mib_stop':18500,'consecutive_samples':2,'poll_seconds':2,
                'minimum_available_ram_kib':2097152,'script_sha256':sha(Path(guard.__file__)),
                'worker_sha256':sha(ROOT/'scripts/utils/af2_wsl_batch.py'),'jobs_sha256':sha(folder/'jobs.jsonl')})

def status(stage,**kw): guard.write(OUT/'status.json',dict(stage=stage,wsl_pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))

def run():
    if os.name!='posix': raise RuntimeError('WSL only')
    if (OUT/'status.json').exists(): raise RuntimeError('Do not silently resume')
    p=read(OUT/'protocol.json')
    for path,digest in p['code'].items(): assert sha(ROOT/path)==digest
    for path,digest in p['source'].items(): assert sha(SOURCE/path)==digest
    import yaml
    for model in (1,2):
        suffix='_model2' if model==2 else ''
        cfg=yaml.safe_load((ROOT/f'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2{suffix}.yml').read_text())['af2']
        assert sha(Path('/mnt/d/DisorderFlowRuntime/cache/colabfold/params')/Path(cfg['model_parameters']).name)==cfg['model_parameters_sha256']
    start=time.monotonic(); completed=0; verified=0; receipts=[]
    for folder in sorted(OUT.glob('model*_batch_*')):
        status('running',batch=folder.name,completed=completed,verified=verified,expected=144)
        guard.OUT=folder; guard.run(); r=read(folder/'status.json'); receipts.append({'batch':folder.name,**r}); completed+=r['completed']
        if not r['verified']:
            status('stopped',batch=folder.name,completed=completed,verified=verified,expected=144,reason=r['reason'] or 'verification_failed'); break
        verified+=6
    else: status('complete',completed=completed,verified=verified,expected=144)
    result={'status':read(OUT/'status.json'),'batches':receipts,'wall_seconds':time.monotonic()-start,
        'all144_verified':len(receipts)==24 and all(r['verified'] for r in receipts),'boundary':p['boundary'],'cost':p['cost']}
    save(OUT/'result.json',result)
    lines=['# v18完整筛选任务的小批次验证','',f"状态：{result['status']['stage']}；已输出{completed}/144；完整批次已校验{verified}/144；墙钟{result['wall_seconds']:.1f}秒。",
        '', '每批6任务，2个模型、3个种子和24个候选不变。任一批触发保护则停止，不自动重试。',
        '', '这是一次完整筛选规模的运行可行性检查，不是同期全量对照；未计入候选生成和此前筛选时间，不能据此宣称端到端加速或长期稳定性。','']
    (ROOT/'docs/PAE_SCREEN_BATCHES_V18_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run']); args=parser.parse_args()
    try: (prepare if args.stage=='prepare' else run)()
    except Exception as exc:
        if OUT.exists(): status('blocked',error=str(exc))
        raise
