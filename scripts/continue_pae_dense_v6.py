"""One bounded local pipeline continuation after the active generator completes."""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import OUT,read
from scripts.run_multiscaffold_v2_af2 import atomic_json


def main():
    # 有界续跑器：等待活跃的生成任务收尾，再依次触发下游两步。
    #   1. 最长等 7200 秒（2 小时），每 10 秒轮询 supplement_generation.json
    #      （读失败按"正在原子写入"处理，继续等）；expected_attempts=768 为
    #      本轮冻结生成网格的总槽位数，仅作进度记账；直到状态为
    #      raw_generation_complete，否则超时抛错。
    #   2. 依次运行 prepare_pae_dense_v6.py --select（候选选择）与
    #      run_pae_dense_v6.py（AF2 批量评分），实时计数在 progress.json。
    #   3. 每阶段覆写 pipeline_status.json；任何异常标记为 blocked 并附带
    #      traceback 后原样抛出（保持非零退出码）。
    started=time.time()
    while time.time()-started<7200:
        p=OUT/'supplement_generation.json'
        if p.exists():
            try: data=read(p)
            except (ValueError,PermissionError): time.sleep(10); continue
            successes=sum(r['status']=='success' for r in data['attempts'])
            atomic_json(OUT/'pipeline_status.json',{'stage':'generation','completed_attempts':len(data['attempts']),
                'successes':successes,'expected_attempts':768,'elapsed_wait_seconds':time.time()-started})
            if data['status']=='raw_generation_complete': break
        time.sleep(10)
    else:
        raise RuntimeError('Generation did not complete within 2 hours; inspect generator log/session')
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_pae_dense_v6.py'),'--select'],check=True,cwd=ROOT)
    atomic_json(OUT/'pipeline_status.json',{'stage':'af2','note':'Live counts are in progress.json'})
    subprocess.run([sys.executable,str(ROOT/'scripts/run_pae_dense_v6.py')],check=True,cwd=ROOT)
    status=read(OUT/'progress.json')
    atomic_json(OUT/'pipeline_status.json',{'stage':status['status'],'final_progress':status})


if __name__=='__main__':
    try: main()
    except Exception as exc:
        atomic_json(OUT/'pipeline_status.json',{'stage':'blocked','error':str(exc),'traceback':traceback.format_exc()})
        raise
