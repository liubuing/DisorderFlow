#!/usr/bin/env python
"""Check GPU/runtime readiness for StateContrast-Ab continuation."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Check StateContrast-Ab GPU readiness")
    parser.add_argument("--out", required=True)
    parser.add_argument("--colabfold", default=r"C:\cf\Scripts\colabfold_batch.exe")
    parser.add_argument("--python", default=sys.executable, help="Python executable to test for torch/JAX GPU runtime")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = build_checks(args.colabfold, args.python)
    summary = summarize_checks(checks)
    with open(out_dir / "gpu_readiness_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "checks": checks}, f, indent=2)
    write_report(out_dir / "gpu_readiness_report.md", summary, checks)
    print(f"Wrote GPU readiness report to {out_dir}")
    print(f"status={summary['overall_status']} required_failures={summary['required_failures']}")


def build_checks(colabfold_path, python_executable=sys.executable):
    checks = []
    checks.append(command_check("nvidia_smi", ["nvidia-smi"], required=False))
    checks.append(python_check("torch_cuda", "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())", python_executable, required=False))
    checks.append(python_check("jax_gpu", "import jax; print(jax.__version__); print(jax.default_backend()); print(jax.devices())", python_executable, required=False))
    checks.append(path_check("colabfold_executable", colabfold_path, required=False))
    if Path(colabfold_path).exists():
        checks.append(command_check("colabfold_version", [colabfold_path, "--version"], required=False, timeout=30))
    checks.extend(project_file_checks())
    return checks


def project_file_checks():
    required_paths = [
        "configs/idp/abeta_reference_whitelist.yml",
        "data/anti_abeta_refs/4HIX.pdb",
        "data/anti_abeta_refs/5CSZ.pdb",
        "scripts/pipeline/run_statecontrast_ab_effect_guided_v4.py",
        "scripts/pipeline/build_statecontrast_ab_final_report.py",
        "outputs/statecontrast_ab_final_report_v1/statecontrast_ab_final_method_report.md",
    ]
    return [path_check(f"project_file:{p}", PROJECT_ROOT / p, required=True) for p in required_paths]


def path_check(name, path, required=True):
    p = Path(path)
    return {
        "name": name,
        "kind": "path",
        "required": required,
        "status": "pass" if p.exists() else "fail",
        "path": str(p),
        "detail": "exists" if p.exists() else "missing",
    }


def command_check(name, cmd, required=True, timeout=15):
    executable = shutil.which(cmd[0]) if not Path(cmd[0]).exists() else cmd[0]
    if not executable:
        return {"name": name, "kind": "command", "required": required, "status": "fail", "command": cmd, "detail": "executable_not_found"}
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return {"name": name, "kind": "command", "required": required, "status": "fail", "command": cmd, "detail": str(exc)}
    return {
        "name": name,
        "kind": "command",
        "required": required,
        "status": "pass" if proc.returncode == 0 else "fail",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }


def python_check(name, code, python_executable=sys.executable, required=True):
    return command_check(name, [python_executable, "-c", code], required=required, timeout=30)


def tail(text, max_chars=1200):
    text = text or ""
    return text[-max_chars:]


def summarize_checks(checks):
    required_failures = [c for c in checks if c.get("required") and c.get("status") != "pass"]
    optional_failures = [c for c in checks if not c.get("required") and c.get("status") != "pass"]
    driver_pass = any(c["name"] == "nvidia_smi" and c["status"] == "pass" for c in checks)
    torch_gpu_pass = any(c["name"] == "torch_cuda" and c["status"] == "pass" and "True" in c.get("stdout_tail", "") for c in checks)
    jax_gpu_pass = any(c["name"] == "jax_gpu" and c["status"] == "pass" and "gpu" in c.get("stdout_tail", "").lower() for c in checks)
    python_gpu_pass = torch_gpu_pass or jax_gpu_pass
    colabfold_pass = any(c["name"] == "colabfold_executable" and c["status"] == "pass" for c in checks)
    if required_failures:
        status = "blocked_missing_required_project_files"
    elif driver_pass and python_gpu_pass:
        status = "gpu_ready_python_runtime_ready"
    elif driver_pass and colabfold_pass:
        status = "gpu_driver_ready_python_runtime_not_ready"
    elif driver_pass:
        status = "gpu_driver_detected_runtime_incomplete"
    else:
        status = "cpu_only_ready_gpu_not_detected"
    return {
        "overall_status": status,
        "required_failures": len(required_failures),
        "optional_failures": len(optional_failures),
        "gpu_driver_detected": driver_pass,
        "python_gpu_runtime_detected": python_gpu_pass,
        "torch_cuda_detected": torch_gpu_pass,
        "jax_gpu_detected": jax_gpu_pass,
        "colabfold_detected": colabfold_pass,
    }


def write_report(path, summary, checks):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast-Ab GPU Readiness Report\n\n")
        f.write(f"Overall status: `{summary['overall_status']}`\n\n")
        f.write(f"GPU driver detected: `{summary['gpu_driver_detected']}`\n\n")
        f.write(f"Python GPU runtime detected: `{summary['python_gpu_runtime_detected']}`\n\n")
        f.write(f"ColabFold executable detected: `{summary['colabfold_detected']}`\n\n")
        f.write("| Check | Required | Status | Detail |\n")
        f.write("|---|---:|---|---|\n")
        for c in checks:
            detail = c.get("detail") or c.get("stdout_tail") or c.get("stderr_tail") or ""
            detail = str(detail).replace("\n", "<br>")[:500]
            f.write(f"| {c['name']} | {c['required']} | {c['status']} | {detail} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("StateContrast-Ab scoring and attribution benchmarks are CPU-light. GPU readiness matters mainly for ColabFold/AF2-style fold side-checks and future neural ensemble generation.\n")


if __name__ == "__main__":
    main()
