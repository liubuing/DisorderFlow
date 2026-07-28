"""v5 多构象数据集构建进度监控
用法（任选一种）：
  wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/biological/DisorderFlow && source venv_wsl/bin/activate && python monitor_v5_progress.py"
  C:\\cf\\Scripts\\python.exe monitor_v5_progress.py
显示：当前进度、ETA、速率、内存、最近完成条目、进程状态
"""
import lmdb, pickle, os, time, subprocess, re, sys

# 自动定位项目根（支持 WSL 和 Windows）
_here = os.path.dirname(os.path.abspath(__file__))
os.chdir(_here)

V5 = "data/confidence_conformation_v5/confidence_train.lmdb"
LOG = "_build_v5.log"
TOTAL_TARGET = 1301  # 统一源 v2 总条目
WSL_DISTRO = os.environ.get("DISORDERFLOW_WSL_DISTRO", "Ubuntu-24.04")

def get_progress():
    if not os.path.exists(V5):
        return 0, []
    e = lmdb.open(V5, readonly=True, lock=False)
    t = e.begin()
    v = t.get(b"__len__")
    n = pickle.loads(v) if v else 0
    keys = [k for k, _ in t.cursor() if k != b"__len__"]
    last_entries = []
    for k in keys[-3:]:
        try:
            entry = pickle.loads(t.get(k))
            last_entries.append({
                "pdb": entry.get("pdb_id", "?"),
                "L": len(entry.get("sequence", "")),
                "n_conf": entry.get("n_conformations", 0),
            })
        except Exception:
            last_entries.append({"pdb": "?", "L": 0, "n_conf": 0})
    e.close()
    return n, last_entries

def get_log_stats():
    if not os.path.exists(LOG):
        return None
    with open(LOG, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # 找最近的 [N/1190] 行
    prog_lines = [l for l in lines if re.search(r"\[\s*\d+/\d+\]", l)]
    recent = prog_lines[-5:] if prog_lines else []
    # 解析最新速率/ETA
    rate = eta = rss = None
    if prog_lines:
        m = re.search(r"\[([\d.]+)/s ETA (\d+)m RSS=(\d+)MB\]", prog_lines[-1])
        if m:
            rate, eta, rss = float(m.group(1)), int(m.group(2)), int(m.group(3))
    # 检查崩溃/重启
    launches = [l for l in lines if "Launching build" in l]
    crashes = [l for l in lines if "exited code" in l]
    return {"recent": recent, "rate": rate, "eta_min": eta, "rss_mb": rss,
            "n_launches": len(launches), "n_crashes": len(crashes),
            "last_crash": crashes[-1].strip() if crashes else None}

def get_proc_status():
    in_wsl = "microsoft" in os.uname().release.lower() if hasattr(os, "uname") else False
    try:
        if in_wsl:
            r = subprocess.run(["bash", "-lc", "pgrep -fa build_conformation | head -1"],
                               capture_output=True, text=True, timeout=10)
        else:
            r = subprocess.run(["wsl", "-d", WSL_DISTRO, "bash", "-lc",
                                "pgrep -fa build_conformation | head -1"],
                               capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.stdout.strip() else "NOT RUNNING"
    except Exception as e:
        return f"check failed: {e}"

def main():
    print("=" * 64)
    print("  DisOrderFlow v5 多构象数据集构建进度")
    print("=" * 64)

    n, last = get_progress()
    pct = n / TOTAL_TARGET * 100
    print(f"进度: {n}/{TOTAL_TARGET}  ({pct:.1f}%)")

    stats = get_log_stats()
    if stats:
        if stats["rate"]:
            print(f"速率: {stats['rate']:.3f} 蛋白/秒  ({stats['rate']*60:.1f}/分)")
        if stats["eta_min"]:
            eta_h = stats["eta_min"] / 60
            print(f"ETA:  {stats['eta_min']} 分钟  ({eta_h:.1f} 小时)")
        if stats["rss_mb"]:
            print(f"内存: RSS={stats['rss_mb']}MB  (上限 8GB GPU)")
        print(f"重启: 启动 {stats['n_launches']} 次, 崩溃 {stats['n_crashes']} 次")
        if stats["last_crash"]:
            print(f"  最近崩溃: {stats['last_crash']}")
        if stats["recent"]:
            print(f"\n最近完成:")
            for l in stats["recent"]:
                print(f"  {l.strip()}")

    print(f"\n最近条目:")
    for e in last:
        print(f"  {e['pdb']:10s} L={e['L']:3d} n_conf={e['n_conf']}")

    proc = get_proc_status()
    print(f"\n进程: {proc}")

    # 健康判断
    print("\n" + "-" * 64)
    if n == TOTAL_TARGET:
        print("[完成] 数据集构建完毕！可运行校验: python _inspect_conformation.py")
    elif "python -u scripts/build" in proc or "build_conformation" in proc:
        print("[运行中] 健康运行。断电后会自动从 {} 继续。".format(n))
    else:
        print(f"[停止] 进程未运行！手动恢复: bash _watchdog_v5.sh")
        print(f"  或重启电脑（开机自启动会自动恢复）")
    print("-" * 64)

if __name__ == "__main__":
    main()
