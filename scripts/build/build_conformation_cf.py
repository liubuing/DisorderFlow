#!/usr/bin/env python
"""
Build Multi-Conformation LMDB using AF2 JAX (tested and working in C:\cf venv).

For each IDP in V11:
  1. Run AF2 multimer x N seeds -> get backbone coordinates
  2. Align conformations via Kabsch on Ca
  3. Compute per-residue Ca RMSF -> disorder label
  4. Store in data/confidence_conformation_v2/

Usage (from project root):
  /c/cf/Scripts/python scripts/build/build_conformation_cf.py --n_seeds 3 --max_idp 2 --split val
  /c/cf/Scripts/python scripts/build/build_conformation_cf.py --n_seeds 5 --max_idp 863 --split train
"""
import os, sys, json, time, pickle, argparse, gc, lmdb, numpy as np
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "modules"))
os.chdir(_project_root)

SRC_LMDB = "data/confidence_unified_v2"
DST_DIR = "data/confidence_conformation_v5"
# 序列长度上限：超过 500 残基的 IDP 直接跳过（AF2 CPU/GPU 内存与运行时间的实际约束）。
MAX_SEQ_LEN = 500
# 重原子张量第 2 维中 CA 原子的索引（af2 37 原子布局: N=0, CA=1, C=2, O=3...）。
CA_IDX = 1
# 单个 IDP 条目（所有 seed 合计）的子进程超时：90 分钟，按最长序列 + 全部 seed 估算。
DEFAULT_TIMEOUT = 5400  # 90 min per IDP entry (all seeds combined)


def _get_rss_mb():
    """Return current process RSS in MB, cross-platform. Returns -1 on failure."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except (ImportError, AttributeError):
        pass
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
    except (ImportError, Exception):
        return -1


def kabsch_align(P, Q):
    # Kabsch 算法：求把点集 P 最优叠合到 Q 的旋转 R 与平移 t。
    # 通过对协方差矩阵 H = Pc^T @ Qc 做 SVD 得到旋转；当行列式为负时翻转
    # 最后一个奇异向量以排除反射（保证是纯旋转）。返回 (R, t)，
    # 满足 Q ≈ P @ R.T + t（RMSF 计算中 P/Q 为不同构象的 CA 坐标，单位: 埃）。
    p_mean = P.mean(axis=0); q_mean = Q.mean(axis=0)
    Pc = P - p_mean; Qc = Q - q_mean
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    t = q_mean - R @ p_mean
    return R, t


def compute_rmsf(pos_list):
    """Per-residue Ca RMSF from list of (N_res, 37, 3) arrays."""
    ca = [p[:, CA_IDX, :] for p in pos_list]
    ref = ca[0]
    aligned = [ref]
    for i in range(1, len(ca)):
        R, t = kabsch_align(ca[i], ref)
        aligned.append(ca[i] @ R.T + t)
    aligned = np.stack(aligned, axis=0)
    sq = ((aligned - aligned.mean(axis=0)) ** 2).sum(axis=-1)
    return np.sqrt(sq.mean(axis=0))


def rmsf_to_disorder(rmsf, scale=2.0):
    # 把逐残基 CA RMSF（埃）映射到 0-1 的无序标签：tanh(rmsf/scale)。
    # scale=2.0 埃为柔性尺度参考：RMSF 远小于 2 Å 的残基标签趋于 0（有序），
    # 达到或超过 2 Å 的趋于 1（无序）。tanh 保证有界且单调。
    return np.tanh(rmsf / scale)


# One-seed worker: runs a SINGLE AF2 prediction, returns JSON result.
# Each seed gets its own subprocess so a stack-overflow crash in one seed
# doesn't kill the others, and JAX memory is fully released between seeds.
_ONE_SEED_WORKER = r'''
import sys, json, os, threading
# Increase thread stack size to reduce risk of STATUS_STACK_BUFFER_OVERRUN
# in JAX/XLA compiled code on Windows (0xC0000409 on L > 200).
threading.stack_size(8 * 1024 * 1024)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_min_max=false")
os.chdir({project_root!r})
sys.path.insert(0, {project_root!r})
sys.path.insert(0, {modules_dir!r})
from af2_jax_runner import run_multimer_prediction
import numpy as _np

data = json.loads(sys.stdin.read())
seq = data["seq"]
seed = data["seed"]
nr = data["recycle"]
try:
    r = run_multimer_prediction(seq, "", num_recycle=nr,
                                 jax_random_seed=seed, return_structure=True)
    if r.get("success"):
        pos = r.get("final_atom_positions")
        if pos is not None and pos.shape[0] == len(seq):
            print(json.dumps({{"ok": True, "pos": _np.array(pos).tolist()}}))
            sys.exit(0)
    print(json.dumps({{"ok": False}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)[:200]}}))
'''.format(project_root=_project_root, modules_dir=os.path.join(_project_root, "modules"))


def run_af2_seeds_inprocess(sequence, n_seeds=5, num_recycle=1):
    """Run AF2 JAX — in-process, reusing JIT-compiled XLA kernels across seeds.

    On Linux/WSL2, XLA stack overflows don't occur and subprocess isolation
    is unnecessary. Running seeds in-process allows the first seed to trigger
    JIT compilation, and subsequent seeds reuse the cached XLA binary (~3-4x
    faster overall).

    Returns (positions, failure_reason) tuple (same contract as run_af2_seeds).
    """
    from af2_jax_runner import run_multimer_prediction

    seq_clean = "".join(aa for aa in sequence.upper() if aa in "ARNDCQEGHILKMFPSTWYV")
    if len(seq_clean) > MAX_SEQ_LEN:
        return None, "skipped"

    # 固定奇数种子 1/3/5/7/9：与种子清单协议一致，保证跨批次可复现、可续跑。
    seeds = [1, 3, 5, 7, 9][:n_seeds]
    positions = []
    crashes = 0
    failures = 0

    for seed in seeds:
        print(f"  seed={seed}...", end="", flush=True)
        try:
            r = run_multimer_prediction(seq_clean, "", num_recycle=num_recycle,
                                         jax_random_seed=seed, return_structure=True)
            if r.get("success"):
                pos = r.get("final_atom_positions")
                if pos is not None and pos.shape[0] == len(seq_clean):
                    positions.append(np.array(pos))
                    print(f"OK", end="", flush=True)
                else:
                    failures += 1
                    print(f"FAIL(shape)", end="", flush=True)
            else:
                failures += 1
                err = r.get("error", "?")[:40]
                print(f"FAIL({err})", end="", flush=True)
        except Exception as e:
            crashes += 1
            print(f"CRASH({str(e)[:40]})", end="", flush=True)

    crash_str = f" {crashes}C" if crashes else ""
    fail_str = f" {failures}F" if failures else ""
    if len(positions) >= 2:
        print(f" OK ({len(positions)} confs{crash_str}{fail_str})", flush=True)
        return positions, None
    else:
        reason = "crash" if crashes >= n_seeds else "failed"
        print(f" FAILED ({len(positions)}/{n_seeds}{crash_str}{fail_str})", flush=True)
        return None, reason


def run_af2_seeds(sequence, n_seeds=5, num_recycle=1, timeout=DEFAULT_TIMEOUT):
    """Run AF2 JAX — one subprocess per seed to survive stack-overflow crashes.

    On Windows, JAX/XLA can cause STATUS_STACK_BUFFER_OVERRUN (0xC0000409)
    for sequences L > ~200. Running each seed in its own subprocess isolates
    the crash and allows partial success: if 3/5 seeds crash, we still get 2
    valid conformations.

    On Linux/WSL2, delegates to in-process runner for 3-4x speedup via XLA cache.

    Returns (positions, failure_reason) tuple:
      - positions: list of (N_res, 37, 3) arrays, or None if < 2 succeeded
      - failure_reason: None on success, "crash" on non-zero exit,
                        "timeout" on subprocess timeout, "failed" on zero-result
    """
    # Linux/WSL2: use in-process runner (no stack-overflow issue, faster)
    if sys.platform != "win32":
        return run_af2_seeds_inprocess(sequence, n_seeds, num_recycle)

    import subprocess, json

    seq_clean = "".join(aa for aa in sequence.upper() if aa in "ARNDCQEGHILKMFPSTWYV")
    if len(seq_clean) > MAX_SEQ_LEN:
        return None, "skipped"

    # 固定奇数种子 1/3/5/7/9（与 in-process 版本一致），保证两种执行路径产出可对齐。
    seeds = [1, 3, 5, 7, 9][:n_seeds]
    positions = []
    crashes = 0
    timeouts = 0
    failures = 0

    # Per-seed timeout: fraction of total, min 5 minutes
    # 每个 seed 的超时 = 总超时均摊，下限 5 分钟（过短会把长序列全部误判为超时）。
    seed_timeout = max(300, timeout // n_seeds)

    for seed in seeds:
        print(f"  seed={seed}...", end="", flush=True)

        payload = json.dumps({"seq": seq_clean, "seed": seed, "recycle": num_recycle})

        try:
            r = subprocess.run(
                [sys.executable, "-c", _ONE_SEED_WORKER],
                input=payload, capture_output=True, text=True,
                timeout=seed_timeout
            )

            if r.returncode != 0:
                crashes += 1
                stderr_tail = r.stderr[-120:] if r.stderr else ""
                print(f"CRASH(exit={r.returncode})", end="", flush=True)
                continue

            result = json.loads(r.stdout.strip())
            if result.get("ok"):
                positions.append(np.array(result["pos"]))
                print(f"OK", end="", flush=True)
            else:
                failures += 1
                print(f"FAIL", end="", flush=True)

        except subprocess.TimeoutExpired:
            timeouts += 1
            print(f"TIMEOUT", end="", flush=True)
        except Exception as e:
            crashes += 1
            print(f"ERR:{e}", end="", flush=True)

    # Summarize
    crash_str = f" {crashes}C" if crashes else ""
    to_str = f" {timeouts}T" if timeouts else ""
    fail_str = f" {failures}F" if failures else ""
    if len(positions) >= 2:
        print(f" OK ({len(positions)} confs{crash_str}{to_str}{fail_str})", flush=True)
        return positions, None
    else:
        reason = "failed"
        if crashes >= n_seeds:
            reason = "crash"
        elif timeouts >= n_seeds:
            reason = "timeout"
        print(f" FAILED ({len(positions)}/{n_seeds}{crash_str}{to_str}{fail_str})", flush=True)
        return None, reason


def main():
    # 主循环：逐 IDP 跑 AF2 多 seed、算 RMSF 无序标签并写入目标 LMDB。
    # 管线角色: 教师标签生成——以 AF2 构象集合的 CA RMSF 经 tanh 压缩作为
    # 无序监督信号，落盘到 confidence_conformation_v5 供置信度头训练。
    # 关键机制:
    #   - 按长度升序处理（短序列先跑，快且成功率高，尽早暴露问题）;
    #   - 断点续跑: 目标 LMDB 已有条目时自动推进 resume_from，键从计数写入;
    #   - 稳健性: 连续失败达到 max_consecutive_failures（默认 5，视为系统性
    #     故障）时暂停等待人工确认; crash 后额外冷却 crash_cooldown_sec
    #     （默认 120s）让 OS 回收资源;
    #   - 内存卫生: 每条 IDP 处理后强制 gc + 停顿，RSS 超 12 GB 才清 JAX
    #     缓存（8 GB 显卡留 2 GB 余量; 每条都清会毁掉 XLA 编译缓存，慢约 5 倍）。
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--num_recycle", type=int, default=0,
                        help="AF2 recycle count; 0 saves ~50% memory vs 1, RMSF quality slightly lower")
    parser.add_argument("--max_idp", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--timeout_minutes", type=int, default=90,
                        help="Max minutes per IDP entry before killing subprocess")
    parser.add_argument("--gc_pause_sec", type=int, default=3,
                        help="Sleep seconds between IDPs to let OS reclaim memory")
    parser.add_argument("--max_consecutive_failures", type=int, default=5,
                        help="Pause after N consecutive AF2 failures (0=never pause)")
    parser.add_argument("--crash_cooldown_sec", type=int, default=120,
                        help="Extra cooldown sleep after a CRASH before next IDP")
    args = parser.parse_args()

    src_path = SRC_LMDB
    # If SRC_LMDB doesn't exist directly, try adding confidence_{split}.lmdb suffix
    if not os.path.exists(src_path):
        src_path = os.path.join(SRC_LMDB, f"confidence_{args.split}.lmdb")
    if not os.path.exists(src_path):
        print(f"Source not found: {src_path} (tried both direct and suffixed paths)")
        sys.exit(1)

    src_env = lmdb.open(src_path, readonly=True, lock=False)
    # Count entries by scanning (unified LMDB has no __len__ key)
    with src_env.begin() as txn:
        total = txn.stat()['entries']
    print(f"Source LMDB: {total} entries at {src_path}")

    # Collect ALL entries (unified LMDB is pre-filtered to IDP only)
    idp_entries = []  # (length, src_index)
    with src_env.begin() as txn:
        for i in range(total):
            key = f"{i:08d}".encode()
            val = txn.get(key)
            if val is None:
                continue
            entry = pickle.loads(val)
            L = len(entry.get("sequence", ""))
            if L <= MAX_SEQ_LEN:
                idp_entries.append((L, i))

    # Sort by ascending length — short proteins first (fast, reliable)
    idp_entries.sort(key=lambda x: x[0])
    idp_indices = [idx for _, idx in idp_entries]

    idp_indices = idp_indices[args.resume_from:]
    if args.max_idp is not None:
        idp_indices = idp_indices[:args.max_idp]

    # Show length distribution of remaining queue
    if idp_indices:
        lengths = [L for L, _ in idp_entries[args.resume_from:]][:len(idp_indices)]
        print(f"Length range: {min(lengths)}-{max(lengths)}, avg {sum(lengths)/len(lengths):.0f}")

    n_idp = len(idp_indices)
    n_af2_runs = n_idp * args.n_seeds
    print(f"IDPs: {n_idp}  Seeds: {args.n_seeds}  AF2 runs: {n_af2_runs}")
    if sys.platform == "win32":
        print(f"Estimated: ~{n_af2_runs * 3 / 60:.0f}-{n_af2_runs * 5 / 60:.0f} min (CPU)")
    else:
        # GPU in-process: ~170s first seed + ~45s per extra seed per protein
        _est_min = (n_idp * (170 + 45 * (args.n_seeds - 1))) / 60
        print(f"Estimated: ~{_est_min:.0f} min (GPU in-process, XLA cached)")
    print(f"Output: {DST_DIR}/confidence_{args.split}.lmdb")
    print()

    os.makedirs(DST_DIR, exist_ok=True)
    dst_path = os.path.join(DST_DIR, f"confidence_{args.split}.lmdb")
    dst_env = lmdb.open(dst_path, map_size=64 * 1024 * 1024 * 1024, subdir=True)

    # Count existing entries to avoid overwriting on resume
    existing_keys = []
    with dst_env.begin() as txn:
        cursor = txn.cursor()
        for k, _ in cursor:
            if k != b"__len__":
                existing_keys.append(k)
    n_success = len(existing_keys)
    if n_success > 0 and args.resume_from == 0:
        args.resume_from = n_success
        print(f"Found {n_success} existing entries, auto-setting resume_from={n_success}")
    elif n_success > 0:
        print(f"Found {n_success} existing entries, appending from key {n_success:08d}")
    elif args.resume_from > 0:
        # First run with resume_from — set __len__ so keys start from resume_from
        pass

    # Re-slice idp_indices AFTER auto-resume may have updated args.resume_from
    if args.resume_from > 0:
        _full = [idx for _, idx in idp_entries]
        idp_indices = _full[args.resume_from:]
        if args.max_idp is not None:
            idp_indices = idp_indices[:args.max_idp]
        n_idp = len(idp_indices)
        n_af2_runs = n_idp * args.n_seeds
        if idp_indices:
            _ls = [L for L, _ in idp_entries[args.resume_from:]][:n_idp]
            print(f"Resumed: {n_idp} IDPs remaining (skip {args.resume_from} done)")
            print(f"Length range: {min(_ls)}-{max(_ls)}, avg {sum(_ls)/len(_ls):.0f}")
            if sys.platform == "win32":
                print(f"Adjusted estimate: ~{n_af2_runs * 3 / 60:.0f}-{n_af2_runs * 5 / 60:.0f} min (CPU)")
            else:
                _est_min = (n_idp * (170 + 45 * (args.n_seeds - 1))) / 60
                print(f"Adjusted estimate: ~{_est_min:.0f} min (GPU in-process, XLA cached)")
            print()

    if n_success == 0:
        with dst_env.begin(write=True) as txn:
            txn.put(b"__len__", pickle.dumps(0))

    t_start = time.time()
    consecutive_failures = 0

    for local_idx, src_idx in enumerate(idp_indices):
        try:
            with src_env.begin() as txn:
                entry = pickle.loads(txn.get(f"{src_idx:08d}".encode()))

            seq = entry.get("sequence", "")
            pdb = entry.get("pdb_id", "?")
            L = len(seq)
            t0 = time.time()

            positions, fail_reason = run_af2_seeds(seq, n_seeds=args.n_seeds,
                                     num_recycle=args.num_recycle,
                                     timeout=args.timeout_minutes * 60)

            if positions is None:
                consecutive_failures += 1
                elapsed = time.time() - t0
                print(f"[{local_idx+1:4d}/{n_idp}] {pdb} L={L:3d} FAILED({fail_reason}) "
                      f"({elapsed:.0f}s) [consec_fail={consecutive_failures}]")
                gc.collect()
                # Extra cooldown after a crash to let OS stabilize
                pause = args.gc_pause_sec
                if fail_reason == "crash":
                    pause = max(pause, args.crash_cooldown_sec)
                    print(f"  >>> Crash cooldown: sleeping {pause}s to let OS reclaim resources...")
                time.sleep(pause)

                # Auto-pause if too many consecutive failures (likely systemic issue)
                if args.max_consecutive_failures > 0 and consecutive_failures >= args.max_consecutive_failures:
                    print()
                    print("=" * 60)
                    print(f"  AUTO-PAUSE: {consecutive_failures} consecutive failures reached")
                    print(f"  Last failed: {pdb} L={L} reason={fail_reason}")
                    print(f"  Progress: {local_idx+1}/{n_idp}  LMDB records: {n_success}")
                    print(f"  Resume command:")
                    print(f"    /c/cf/Scripts/python -u scripts/build/build_conformation_cf.py \\")
                    print(f"      --n_seeds {args.n_seeds} --max_idp {args.max_idp or n_idp} "
                          f"--split {args.split} --resume_from {args.resume_from + local_idx + 1}")
                    print(f"  Press Enter to continue or Ctrl+C to abort...")
                    print("=" * 60)
                    try:
                        input()
                    except (EOFError, KeyboardInterrupt):
                        print("\n  Aborted by user. Exiting.")
                        sys.exit(0)
                    print("  Resuming...")
                    consecutive_failures = 0
                continue

            # Success — reset failure counter
            consecutive_failures = 0

            rmsf = compute_rmsf(positions)
            disorder = rmsf_to_disorder(rmsf)

            result = {
                "pdb_id": pdb,
                "sequence": seq,
                "n_conformations": len(positions),
                "ca_positions": [p[:, CA_IDX, :].astype(np.float32) for p in positions],
                "rmsf": rmsf.astype(np.float32),
                "disorder_label": disorder.astype(np.float32),
                "batch": entry["batch"],
            }

            with dst_env.begin(write=True) as txn:
                txn.put(f"{n_success:08d}".encode(), pickle.dumps(result))
                txn.put(b"__len__", pickle.dumps(n_success + 1))

            n_success += 1
            elapsed = time.time() - t0
            total_elapsed = time.time() - t_start
            rate = (local_idx + 1) / max(total_elapsed, 1)
            eta = (n_idp - local_idx - 1) / max(rate, 0.0001)
            rss_mb = _get_rss_mb()
            print(f"[{local_idx+1:4d}/{n_idp}] {pdb} L={L:3d} {len(positions)} confs "
                  f"({elapsed:.0f}s) [{rate:.3f}/s ETA {eta/60:.0f}m RSS={rss_mb}MB]")

            # Memory hygiene: clear JAX / numpy / worker caches, then pause so OS
            # can reclaim RSS. Without this, RSS grows monotonically across
            # long-L IDPs and the process gets OOM-killed around seed=9.
            del positions, rmsf, disorder, result
            gc.collect()
            # Only clear JAX caches if RSS > 6GB (GPU has 8GB, keep 2GB headroom)
            # Clearing every entry destroys XLA compilation cache → 5x slower
            rss_mb_now = _get_rss_mb()
            if rss_mb_now > 12000:
                try:
                    import jax; jax.clear_caches()
                    print(f" [JAX cache cleared: RSS={rss_mb_now}MB]", flush=True)
                except Exception:
                    pass
            time.sleep(args.gc_pause_sec)

        except Exception as e:
            print(f"[{local_idx+1:4d}/{n_idp}] ERROR: {e}")
            import traceback; traceback.print_exc()
            gc.collect()
            time.sleep(args.gc_pause_sec)

    with dst_env.begin(write=True) as txn:
        txn.put(b"__len__", pickle.dumps(n_success))

    src_env.close(); dst_env.close()
    elapsed = time.time() - t_start
    print(f"\nDone. {n_success}/{n_idp} succeeded in {elapsed/60:.0f} min.")


if __name__ == "__main__":
    main()
