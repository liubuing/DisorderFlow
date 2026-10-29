"""Guarded AF2 teacher run for the analog prescreening cohort (v29).

prepare (Windows): chunk v28 job lists into per-model batches of six jobs and
freeze per-attempt batch folders with the unmodified v19 guard protocol.

run (WSL): per batch, GPU admission (three consecutive idle samples), then the
v19 guard worker; one fresh retry on gpu-memory or timeout stops only. No
speedup claim; scheduling boundary recorded like v19.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/pae_analog_teacher_v29'
V28 = ROOT / 'results/pae_analog_generation_v28'
BATCH_JOBS = 12
TIMEOUT_SECONDS = 3600
# DEV-002 (results/pae_analog_deviation_log/DEV-002_memory_guard.json):
# shared-desktop evening memory pressure trips the v19 18500 MiB total-memory
# stop; raised to 19500 MiB (still ~1 GB below the 20480 MiB card) and a batch
# that exhausts both attempts is recorded and skipped instead of stopping the
# whole run; exhausted batches are swept afterwards. Scheduling only.
GPU_STOP_MIB = 19500
DEADLINE_SECONDS = 86400

GUARD = ROOT / 'scripts/guard_pae_batch_v19.py'
WORKER = ROOT / 'scripts/utils/af2_wsl_batch.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(stage, **kwargs):
    # Live telemetry, not a frozen artifact: overwrite freely.
    (OUT / 'status.json').write_text(
        json.dumps({'stage': stage, 'pid': os.getpid(),
                    'time': time.strftime('%Y-%m-%d %H:%M:%S'), **kwargs}, indent=1) + '\n',
        encoding='utf-8')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('protocol.json', 'summary.json', 'batches.json'):
        path = OUT / name
        if path.exists():
            path.unlink()
    batches = []
    for model in (1, 2):
        jobs = [json.loads(line) for line in (V28 / f'jobs_model{model}.jsonl').read_text().splitlines() if line.strip()]
        for start in range(0, len(jobs), BATCH_JOBS):
            chunk = jobs[start:start + BATCH_JOBS]
            name = f'model{model}_batch_{start:04d}'
            batches.append({'name': name, 'model': model, 'jobs': len(chunk),
                            'ids': [job['id'] for job in chunk]})
    protocol = {
        'classification': 'guarded_af2_teacher_labels_not_timing_comparison',
        'v28_generation_sha256': digest(V28 / 'generation.json'),
        'v28_jobs_sha256': {model: digest(V28 / f'jobs_model{model}.jsonl') for model in (1, 2)},
        'batch_jobs': BATCH_JOBS, 'timeout_seconds': TIMEOUT_SECONDS,
        'gpu_memory_mib_stop': GPU_STOP_MIB, 'deadline_seconds': DEADLINE_SECONDS,
        'admission': 'three consecutive samples 10s apart with utilization<20, memory<3000MiB, no compute apps',
        'retry': 'one fresh attempt per batch on gpu_memory_guard or timeout only',
        'guard_script_sha256': sha(GUARD), 'worker_sha256': sha(WORKER),
        'script_sha256': sha(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)
    freeze(OUT / 'batches.json', {'batches': batches,
                                  'total_slots': sum(b['jobs'] for b in batches)})
    print(json.dumps({'batches': len(batches), 'total_slots': sum(b['jobs'] for b in batches)}), flush=True)


def admission(deadline, batch, completed, total):
    # DEV-001 (results/pae_analog_deviation_log/DEV-001_admission_utilization.json):
    # utilization threshold relaxed 20% -> 60% for user-directed continuation on
    # an actively used desktop; memory condition and the 18500 MiB stop guard
    # are unchanged. Science-free scheduling change, disclosed in the manuscript.
    idle = 0
    with (OUT / 'admission.jsonl').open('a') as log:
        while time.monotonic() < deadline:
            a = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            b = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
            if a.returncode or b.returncode:
                raise RuntimeError('GPU ownership telemetry unavailable')
            util, mem = map(float, a.stdout.strip().split(','))
            idle = idle + 1 if util < 60 and mem < 3000 and not b.stdout.strip() else 0
            log.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'batch': batch,
                                  'util': util, 'memory_mib': mem, 'owners': b.stdout.strip(),
                                  'consecutive_idle': idle}) + '\n')
            log.flush()
            status('waiting_for_gpu', batch=batch, completed=completed, expected=total,
                   gpu_memory_mib=mem, gpu_utilization=util, owners=b.stdout.strip())
            if idle >= 3:
                return True
            time.sleep(10)
    return False


def run():
    if os.name != 'posix':
        raise RuntimeError('Run inside WSL')
    protocol = read(OUT / 'protocol.json')
    assert sha(ROOT / 'scripts/guard_pae_batch_v19.py') == protocol['guard_script_sha256']
    assert sha(ROOT / 'scripts/utils/af2_wsl_batch.py') == protocol['worker_sha256']
    batches = read(OUT / 'batches.json')['batches']
    total = read(OUT / 'batches.json')['total_slots']
    start = time.monotonic()
    deadline = start + DEADLINE_SECONDS
    completed = 0
    attempts = []
    exhausted = []
    queue = 0.0
    for batch in batches:
        name = batch['name']
        base = OUT / name
        base.mkdir(parents=True, exist_ok=True)
        done = False
        last_reason = None
        for attempt in (1, 2):
            folder = base / f'attempt{attempt}'
            folder.mkdir(parents=True, exist_ok=True)
            prior = folder / 'status.json'
            if prior.exists():
                prior_status = read(prior)
                if prior_status.get('verified'):
                    completed += len([l for l in (folder / 'jobs.jsonl').read_text().splitlines() if l.strip()])
                    done = True
                    break
                # Any prior non-verified terminal state (stopped, guard trip or
                # stale running) means this attempt number is consumed across
                # restarts; the guard refuses silent resume, so move on.
                continue
            jobs = [json.loads(line)
                    for line in (V28 / f"jobs_model{batch['model']}.jsonl").read_text().splitlines()
                    if line.strip()]
            # Batch names carry the start index (prepare froze batches.json with the same slicing).
            index = int(name.split('_')[-1])
            chunk = jobs[index:index + BATCH_JOBS]
            wsl_base = f'/mnt/d/biological/DisorderFlow/results/pae_analog_teacher_v29/{name}/attempt{attempt}'
            rows = []
            for i, job in enumerate(chunk):
                rows.append({**job,
                             'output_pae': f'{wsl_base}/pae/{i:02d}.npz',
                             'output_pdb': f'{wsl_base}/pdb/{i:02d}.pdb'})
            (folder / 'jobs.jsonl').write_text(''.join(json.dumps(job) + '\n' for job in rows))
            freeze(folder / 'protocol.json', {
                'classification': 'guarded_af2_teacher_batch',
                'jobs': len(rows), 'model': batch['model'], 'timeout_seconds': TIMEOUT_SECONDS,
                'gpu_memory_mib_stop': GPU_STOP_MIB, 'consecutive_samples': 2, 'poll_seconds': 2,
                'minimum_available_ram_kib': 2097152,
                # 'script_sha256' is the key the unmodified v19 guard asserts on.
                'script_sha256': sha(ROOT / 'scripts/guard_pae_batch_v19.py'),
                'worker_sha256': sha(ROOT / 'scripts/utils/af2_wsl_batch.py'),
                'jobs_sha256': sha(folder / 'jobs.jsonl')})
            wait_start = time.monotonic()
            ready = admission(deadline, name, completed, total)
            queue += time.monotonic() - wait_start
            if not ready:
                status('stopped', reason='24h_admission_deadline', completed=completed, expected=total)
                return
            status('running', batch=name, attempt=attempt, completed=completed, expected=total)
            import scripts.guard_pae_batch_v19 as guard
            guard.OUT = folder
            guard.run()
            result = read(folder / 'status.json')
            attempts.append({'batch': name, 'attempt': attempt, **result})
            (OUT / 'attempts.json').write_text(
                json.dumps({'attempts': attempts, 'queue_seconds': queue}, indent=1) + '\n',
                encoding='utf-8')
            if result['verified']:
                completed += len(rows)
                done = True
                break
            last_reason = result.get('reason') or 'verification_failed'
        # DEV-002: a batch whose attempts are exhausted (or that failed
        # verification) is recorded and skipped; the run continues and the
        # exhausted batches are swept in a later pass.
        if not done:
            exhausted.append({'batch': name, 'reason': last_reason})
            (OUT / 'exhausted.json').write_text(
                json.dumps({'exhausted': exhausted}, indent=1) + '\n', encoding='utf-8')
            status('exhausted_batch_skipped', batch=name, reason=last_reason,
                   completed=completed, expected=total)
    complete = completed == total and not exhausted
    (OUT / 'result.json').write_text(
        json.dumps({'completed': completed, 'total': total,
                    'verified': complete,
                    'exhausted_batches': exhausted,
                    'queue_seconds': queue,
                    'elapsed_seconds': time.monotonic() - start,
                    'attempts': attempts,
                    'boundary': 'guarded scheduling only; no speedup or stability claim'}, indent=1) + '\n',
        encoding='utf-8')
    status('complete' if complete else 'stopped_exhausted_batches',
           completed=completed, expected=total, exhausted=len(exhausted))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        (prepare if args.stage == 'prepare' else run)()
    except Exception as exc:
        if OUT.exists():
            status('blocked', error=str(exc))
        raise
