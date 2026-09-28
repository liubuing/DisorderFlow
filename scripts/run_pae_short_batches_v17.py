"""One guarded 24-job feasibility run, four fresh workers; no speedup claim."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import guard_pae_batch_v16 as guard
from scripts.run_guarded_abba_v16 import read,save,sha
OUT=ROOT/'results/pae_short_batches_v17'
OLD=ROOT/'results/pae_guarded_abba_v16'


def prepare():
    parent=read(OLD/'protocol.json')
    for path,digest in parent['code'].items(): assert sha(ROOT/path)==digest
    files=['scripts/run_pae_short_batches_v17.py','scripts/run_guarded_abba_v16.py','scripts/guard_pae_batch_v16.py','scripts/utils/af2_wsl_batch.py','modules/af2_jax_runner.py','configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2_model2.yml']
    save(OUT/'protocol.json',{'classification':'post_guard_failure_short_batch_feasibility','jobs':24,'workers':4,
        'parent_protocol_sha256':sha(OLD/'protocol.json'),'parent_status_sha256':sha(OLD/'status.json'),
        'code':{p:sha(ROOT/p) for p in files},
        'boundary':'same24 frozen inputs/model2/seed7103/recycle3; fresh outputs; no retries; stop whole run on any failed batch; one replicate only; not completed ABBA or paired speedup',
        'guard':'unchanged18500MiB GPU two2s samples; RAM2GiB; eachworker600s; fourworkers maximum; no automatic threshold relaxation'})
    ids=[]
    for source in sorted((OLD/'B1').glob('batch_*')):
        folder=OUT/source.name; folder.mkdir()
        batch=[]
        for j in map(json.loads,(source/'jobs.jsonl').read_text().splitlines()):
            new={**j}
            for key in ['output_pdb','output_pae']:
                new[key]=j[key].replace('pae_guarded_abba_v16/B1','pae_short_batches_v17')
            batch.append(new); ids.append(j['id'])
        (folder/'jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in batch))
        bp={**read(source/'protocol.json'),'jobs_sha256':sha(folder/'jobs.jsonl')}
        save(folder/'protocol.json',bp)
    assert len(ids)==len(set(ids))==24


def status(stage,**kw): guard.write(OUT/'status.json',dict(stage=stage,wsl_pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))


def run():
    if os.name!='posix': raise RuntimeError('WSL only')
    if (OUT/'status.json').exists(): raise RuntimeError('Existing run; do not resume timing')
    p=read(OUT/'protocol.json')
    for path,digest in p['code'].items(): assert sha(ROOT/path)==digest
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2_model2.yml').read_text())['af2']
    assert sha(Path('/mnt/d/DisorderFlowRuntime/cache/colabfold/params')/Path(cfg['model_parameters']).name)==cfg['model_parameters_sha256']
    start=time.monotonic(); completed=0; receipts=[]
    for folder in sorted(OUT.glob('batch_*')):
        status('running',batch=folder.name,completed=completed,expected=24)
        guard.OUT=folder; guard.run(); r=read(folder/'status.json'); receipts.append(r); completed+=r['completed']
        if not r['verified']:
            status('stopped',batch=folder.name,completed=completed,expected=24,reason=r['reason'] or 'verification_failed')
            break
    else: status('complete',completed=completed,expected=24)
    result={'status':read(OUT/'status.json'),'batches':receipts,'wall_seconds':time.monotonic()-start,
        'boundary':p['boundary'],'all24_verified':len(receipts)==4 and all(r['verified'] for r in receipts)}
    save(OUT/'result.json',result)
    lines=['# v17短批次可行性结果','',f"状态：{result['status']['stage']}；输出任务数：{completed}/24；总墙钟：{result['wall_seconds']:.1f}秒。",
        f"四批全部校验通过：{result['all24_verified']}。",'', '这是一次每6任务重启的可行性试验。原24任务连续进程在显存保护下中止，没有完整对照耗时，不能计算可靠的长短批次加速比例。通过也不证明72/720任务稳定。',
        '', '若触发保护，不放宽阈值，不自动重试；按status.json中的原因保留结果。','']
    (ROOT/'docs/PAE_SHORT_BATCHES_V17_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run']); args=parser.parse_args()
    try: (prepare if args.stage=='prepare' else run)()
    except Exception as exc:
        if OUT.exists(): status('blocked',error=str(exc))
        raise
