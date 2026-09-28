"""Read-only source audit and reproducible paper evidence summary."""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_guarded_abba_v16 import read,save,sha
OUT=ROOT/'results/pae_evidence_v20'
SOURCE=ROOT/'results/pae_queued_v19'


def main():
    result=read(SOURCE/'result.json')
    assert result['completed']==144 and result['verified']
    reference={r['id']:r for r in map(json.loads,(ROOT/'results/pae_timing_v13/screened/results.jsonl').read_text().splitlines()) if r.get('id')}
    failures=[]; success=[]; ids=set(); hashes={}; comparisons=[]
    for attempt in result['attempts']:
        folder=SOURCE/attempt['batch']/f"attempt{attempt['attempt']}"
        for name in ['protocol.json','jobs.jsonl','status.json','worker.jsonl','telemetry.jsonl']:
            f=folder/name; hashes[str(f.relative_to(ROOT))]=sha(f)
        telemetry=[json.loads(l) for l in (folder/'telemetry.jsonl').read_text().splitlines()]
        rows=[json.loads(l) for l in (folder/'worker.jsonl').read_text().splitlines()]
        jobs=[r for r in rows if r.get('id')]
        foreign=[]
        for t in telemetry:
            others=[line for line in t['gpu_processes'].splitlines() if line.strip() and line.split(',')[0].strip()!=str(t['owned_worker_pid'])]
            if others:foreign.append({'time':t['timestamp'],'processes':others})
        entry={'batch':attempt['batch'],'attempt':attempt['attempt'],'wall_seconds':attempt['wall_seconds'],
            'peak_sampled_gpu_mib':max(float(t['gpu'].split(',')[0]) for t in telemetry),
            'other_compute_process_samples':len(foreign),'telemetry_query_failures':sum(t['gpu_process_query_returncode']!=0 for t in telemetry),
            'warmup_reported':any(r.get('status')=='warmup' for r in rows),'job_results':len(jobs)}
        if not attempt['verified']:
            assert attempt['reason']=='gpu_memory_guard' and not jobs
            failures.append(entry);continue
        expected=[json.loads(l)['id'] for l in (folder/'jobs.jsonl').read_text().splitlines()]
        assert len(jobs)==6 and {r['id'] for r in jobs}==set(expected)
        for r in jobs:
            assert r['success'] and r['id'] not in ids
            ids.add(r['id'])
            path=folder/'pae'/Path(r['pae_path']).name
            assert sha(path)==r['pae_sha256']
            ref=reference[r['id']]; old=ROOT/'results/pae_timing_v13/screened/pae'/Path(ref['pae_path']).name
            assert sha(old)==ref['pae_sha256']
            a=np.load(path)['pae'];b=np.load(old)['pae']
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            comparisons.append({'id':r['id'],'exact_equal':bool(np.array_equal(a,b)),
                'max_absolute_pae_difference':float(np.max(np.abs(a-b))),
                'mean_absolute_pae_difference':float(np.mean(np.abs(a-b)))})
        success.append(entry)
    assert len(ids)==144 and len(success)==24 and len(failures)==12
    summary={'classification':'post_hoc_evidence_audit_not_new_validation',
        'verified_unique_results':len(ids),'successful_batches':len(success),'failed_attempts':len(failures),
        'retry_batch_fraction':len(failures)/len(success),'failed_job_outputs':sum(r['job_results'] for r in failures),
        'failed_before_warmup_report':sum(not r['warmup_reported'] for r in failures),
        'failed_after_warmup_before_first_job':sum(r['warmup_reported'] for r in failures),
        'failed_attempts_with_other_recorded_compute_processes':sum(r['other_compute_process_samples']>0 for r in failures),
        'queue_seconds':result['queue_seconds'],'attempt_seconds':result['sum_attempt_seconds'],
        'failed_attempt_seconds':sum(r['wall_seconds'] for r in failures),'elapsed_seconds':result['elapsed_seconds'],
        'exact_pae_matches_to_v13':sum(r['exact_equal'] for r in comparisons),
        'max_pae_difference_to_v13':max(r['max_absolute_pae_difference'] for r in comparisons),
        'causal_boundary':'No other recorded compute PID does not exclude WDDM graphics, unattributed driver memory, sampling gaps or fragmentation; no root cause established',
        'failures':failures,'successes':success,'pae_comparisons':comparisons,
        'source_sha256':{**hashes,'results/pae_queued_v19/result.json':sha(SOURCE/'result.json')},'script_sha256':sha(Path(__file__))}
    save(OUT/'audit.json',summary)
    lines=['# PAE补强证据审计 v20','','## 运行与重试核查','',
        f"144个唯一任务的ID、完整PAE文件及哈希复核通过。24个批次成功，其中12个批次第二次尝试成功（50%批次曾触发保护）。失败尝试均未输出任务结果。",
        f"12次失败中，{summary['failed_before_warmup_report']}次未报告完成预热，{summary['failed_after_warmup_before_first_job']}次在预热后首个结果输出前停止。12次均未记录其他GPU计算PID；不能据此断言绝无外部占用，也不能将全部重试归因于CatPred。",
        '',f"等待{summary['queue_seconds']/3600:.2f}小时；执行尝试累计{summary['attempt_seconds']/60:.2f}分钟，其中失败尝试{summary['failed_attempt_seconds']/60:.2f}分钟；总历时{summary['elapsed_seconds']/3600:.2f}小时。",
        '',f"与v13同任务的完整PAE矩阵比较：{summary['exact_pae_matches_to_v13']}/144逐元素完全一致，全部矩阵中最大绝对差为{summary['max_pae_difference_to_v13']:.8g}。差异逐任务保留，不将文件完整性等同数值完全一致。",
        '', '## 论文证据与主张边界','',
        '| 证据 | 已有结果 | 允许的表述 | 仍缺什么 |','|---|---|---|---|',
        '| 回顾性筛选 | 六组、四个抗原类别、120候选；20%预算类别等权召回60.4%，随机期望20% | 该候选池中具有PAE优选保留价值 | 严格独立抗原家族验证 |',
        '| 公平基线 | 同池ProteinMPNN采样H3评分8.3%、骨架评分6.2%；固定方向 | 在所测条件下优于这些评分 | 其他生成器、更多实际基线与新数据；不能声称普遍优越 |',
        '| 机制/消融 | 冻结模型H3组成贡献单独排序与全模型相同；重新训练组成模型58.3% | 同骨架候选排序由H3组成驱动 | 不能宣称几何增强组内排序或序列顺序识别 |',
        '| 执行可行性 | 排队、6任务分批与有限重试后144/144完成 | 在共享GPU环境中完成一次完整筛选规模运行 | 50%批次曾失败；不能声称无故障长期稳定或根因修复 |',
        '| 计算成本 | 调用数720→144，减少80%；本轮尝试约49.6分钟另有排队 | 预算限定的调用次数减少；拆分实测成本 | 同期可比全量对照；不能拿旧全量时间计算最终加速比 |',
        '| 生物学效用 | 目标为AF2 PAE，现无新增实验结合结果 | AF2置信度预筛选 | 不能宣称亲和力、功能或临床改善 |',
        '', '## 投稿补强顺序','',
        '现在可撰写“组成驱动的抗体候选PAE预筛选：回顾性效用与适用边界”的结果及方法。运行问题不应继续无限占用科学验证资源。优先扩大独立抗原家族证据；若继续保留速度卖点，则另设同期配对成本实验。现有运行记录只能支撑执行可行性与成本透明披露。','']
    (ROOT/'docs/PAE_EVIDENCE_AUDIT_V20.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['failures','successes','pae_comparisons','source_sha256']},indent=2))

if __name__=='__main__': main()
