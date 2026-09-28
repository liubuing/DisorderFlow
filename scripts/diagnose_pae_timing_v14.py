"""Post-hoc runtime audit. File mtime alignment is approximate, not causal proof."""
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import read,freeze,digest
OUT=ROOT/'results/pae_timing_diagnosis_v14'
SOURCE=ROOT/'results/pae_timing_v13'


def main():
    telemetry=[]
    for line in (SOURCE/'gpu_telemetry.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r.get('returncode')!=0: continue
        fields=[s.strip() for s in r['data'].split(',')]
        telemetry.append(dict(t=datetime.strptime(fields[0],'%Y/%m/%d %H:%M:%S.%f').timestamp(),util=float(fields[2].split()[0]),
            memory_mib=float(fields[3].split()[0]),temperature=float(fields[4]),power=float(fields[5].split()[0])))
    full={r['id']:r for r in map(json.loads,(ROOT/'results/pae_timing_v12/full/results.jsonl').read_text().splitlines()) if r.get('id')}
    rows=[]; groups=defaultdict(list)
    for r in map(json.loads,(SOURCE/'screened/results.jsonl').read_text().splitlines()):
        if not r.get('id'): continue
        path=SOURCE/'screened/pae'/Path(r['pae_path']).name
        assert digest(path)==r['pae_sha256']
        end=path.stat().st_mtime; begin=end-r['elapsed']
        near=[t for t in telemetry if begin<=t['t']<=end]
        item=dict(id=r['id'],elapsed=r['elapsed'],reference_elapsed=full[r['id']]['elapsed'],
            approximate_end=datetime.fromtimestamp(end).isoformat(),telemetry_samples=len(near),
            telemetry={k:statistics.median(t[k] for t in near) for k in ['memory_mib','util','temperature','power']} if near else None)
        rows.append(item); groups[r['id'].split('|')[0]+'|'+r['id'].split('|')[-2]].append(item)
    summary={k:dict(n=len(v),seconds=sum(r['elapsed'] for r in v),reference_seconds=sum(r['reference_elapsed'] for r in v),
        median_seconds=statistics.median(r['elapsed'] for r in v),
        median_memory_mib=statistics.median(r['telemetry']['memory_mib'] for r in v if r['telemetry']) if any(r['telemetry'] for r in v) else None) for k,v in groups.items()}
    freeze(OUT/'audit.json',{'classification':'post_hoc_execution_diagnosis','source_sha256':digest(SOURCE/'screened/results.jsonl'),
        'telemetry_sha256':digest(SOURCE/'gpu_telemetry.jsonl'),'alignment':'PAE file modification timestamp minus worker elapsed; approximate; not event timestamp instrumentation',
        'groups':summary,'rows':rows,'boundary':'GPU process ownership, shared memory paging and OS load not recorded; memory pressure is a hypothesis, not established cause'})
    lines=['# v14计时失速诊断','','| 组件/模型 | 推理耗时秒 | 相同任务全量参考秒 | 单任务中位数秒 | 遥测显存中位数MiB |','|---|---:|---:|---:|---:|']
    for k,v in summary.items(): lines.append(f"| {k} | {v['seconds']:.1f} | {v['reference_seconds']:.1f} | {v['median_seconds']:.1f} | {v['median_memory_mib']} |")
    lines+=['','## 解释与下一步','','耗时增加集中在model2的后五组；model1和model2第一组没有同等级失速。预热总计58.4秒，无法解释总计6065秒的运行时间。',
        '', '对齐使用PAE文件修改时间，属于近似回顾性定位。遥测没有进程归属、GPU共享内存或系统分页计数，不能直接断言OOM、内存泄漏或外部程序竞争。',
        '', '下一步固定相同24个model2任务（六组各四条，seed7103），比较单个24任务工作进程与每6任务新建进程；按固定ABBA顺序执行四个测量单元。记录端到端耗时、每任务耗时、输出校验、GPU遥测及时间戳，显式计入额外启动开销。该试验是执行诊断，不是新增独立生物学验证。','']
    (ROOT/'docs/PAE_TIMING_DIAGNOSIS_V14.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
