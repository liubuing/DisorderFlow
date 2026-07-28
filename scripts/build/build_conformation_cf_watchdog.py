#!/usr/bin/env python
"""
Watchdog: keeps running build_conformation_cf.py, auto-restarting on OOM/segfault/etc.

The build script auto-detects existing entries in the destination lmdb and
resumes from there (build_conformation_cf.py lines 193-197), so we only need
to relaunch after a crash.

Usage (from project root):
  /c/cf/Scripts/python scripts/build/build_conformation_cf_watchdog.py
"""
import os, sys, time, subprocess, signal

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_project_root)

LOG_FILE = os.path.join(_project_root, "conformation_build_watchdog.log")
BUILD_CMD = [
    sys.executable,
    os.path.join(_project_root, "scripts", "build", "build_conformation_cf.py"),
    "--n_seeds", "5",
    "--num_recycle", "0",   # CPU memory friendly
    "--max_idp", "863",
    "--split", "train",
    "--gc_pause_sec", "3",
]

# Crashes that warrant immediate restart vs. those that suggest a real bug.
RESTARTABLE = {-9, -1073741819, 1, 2}  # SIGKILL, ACCESS_VIOLATION, generic error
BACKOFF_START = 15  # seconds
BACKOFF_MAX = 300    # cap at 5 min
CRASH_WINDOW = 600   # 10 min — if 3+ crashes in this window, give up

_crash_log = []  # list of (timestamp, exit_code)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_once():
    log(f"Launching: {' '.join(BUILD_CMD)}")
    t0 = time.time()
    r = subprocess.run(BUILD_CMD, capture_output=False)
    dt = time.time() - t0
    log(f"Build exited code={r.returncode} after {dt/60:.1f} min")
    return r.returncode


def main():
    backoff = BACKOFF_START
    while True:
        code = run_once()
        if code == 0:
            log("Build completed successfully. Exiting watchdog.")
            return 0

        now = time.time()
        _crash_log.append((now, code))
        # keep only recent crashes
        _crash_log[:] = [(t, c) for t, c in _crash_log if now - t < CRASH_WINDOW]
        if len(_crash_log) >= 3:
            log(f"!! {len(_crash_log)} crashes in last {CRASH_WINDOW//60} min — manual intervention needed")
            log("   Inspect conformation_build.log for the last '[N/863]' line and check OOM/segfault pattern.")
            log("   Sleeping 30 min before retrying to give OS time to release memory.")
            time.sleep(1800)
            _crash_log.clear()
            backoff = BACKOFF_START
            continue

        log(f"Restartable exit code? {code in RESTARTABLE}. Backing off {backoff}s before retry...")
        time.sleep(backoff)
        backoff = min(backoff * 2, BACKOFF_MAX)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Watchdog interrupted by user")
        sys.exit(130)
