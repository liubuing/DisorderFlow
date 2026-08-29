"""Resolve native-Windows and WSL2 runtime commands.

The web process and BFN run under Windows Python, while ColabFold runs inside
the configured WSL distribution. Keeping path conversion here prevents every
caller from growing its own ``C:\\...``/``/mnt/c/...`` assumptions.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def windows_to_wsl_path(path):
    """Map an absolute Windows path to its WSL /mnt/<drive> equivalent."""
    resolved = Path(path).resolve().as_posix()
    if len(resolved) >= 3 and resolved[1:3] == ":/":
        return f"/mnt/{resolved[0].lower()}{resolved[2:]}"
    return resolved


def resolve_colabfold_executable(af2_config, project_dir):
    """Resolve ColabFold without silently switching the configured backend.

    For WSL, ``Path.exists`` is intentionally checked against the Windows view
    of the project-local environment before returning the corresponding Linux
    path. A missing configured environment therefore fails early instead of
    accidentally selecting another global ColabFold installation.
    """
    af2_config = af2_config or {}
    backend = str(af2_config.get("backend", "native")).lower()
    executable = af2_config.get("executable", "colabfold_batch")
    if backend == "wsl":
        # The venv itself lives on D:, so both Windows and WSL see the same files.
        environment = af2_config.get("environment", "venv_wsl")
        environment = Path(environment)
        if not environment.is_absolute():
            environment = Path(project_dir) / environment
        candidate = environment / "bin" / executable
        return windows_to_wsl_path(candidate) if candidate.exists() else None

    configured = Path(str(executable)).expanduser()
    if configured.is_absolute() and configured.exists():
        return str(configured)

    venv = af2_config.get("venv")
    if venv:
        venv_path = Path(venv).expanduser()
        if not venv_path.is_absolute():
            venv_path = Path(project_dir) / venv_path
        names = [f"{executable}.exe", executable] if os.name == "nt" else [executable]
        subdir = "Scripts" if os.name == "nt" else "bin"
        for name in names:
            candidate = venv_path / subdir / name
            if candidate.exists():
                return str(candidate)

    return shutil.which(str(executable))


def build_colabfold_command(af2_config, project_dir, input_path, output_path, arguments):
    """Build a ColabFold command for native Windows or WSL2 execution."""
    af2_config = af2_config or {}
    backend = str(af2_config.get("backend", "native")).lower()
    executable = resolve_colabfold_executable(af2_config, project_dir)
    if not executable:
        raise FileNotFoundError("colabfold_batch was not found in the configured environment or PATH")

    if backend == "wsl":
        # Arguments after ``--`` are executed directly by WSL; no shell quoting
        # is needed because subprocess receives an argument list, not a string.
        distro = af2_config.get("distribution", "Ubuntu-24.04-D")
        return [
            "wsl.exe", "-d", distro, "--", executable,
            windows_to_wsl_path(input_path), windows_to_wsl_path(output_path),
            *[str(value) for value in arguments],
        ]
    return [executable, str(input_path), str(output_path), *[str(value) for value in arguments]]


def colabfold_environment(af2_config, project_dir):
    """Return environment variables suitable for the selected AF2 backend."""
    env = os.environ.copy()
    # JAX otherwise reserves most GPU memory at startup, starving concurrent
    # PyTorch/OpenMM jobs even when the current AF2 input is small.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    if str((af2_config or {}).get("backend", "native")).lower() == "wsl":
        # WSL does not inherit arbitrary Windows variables. WSLENV explicitly
        # forwards only the two XLA controls needed by the worker.
        forwarded = ["XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION"]
        existing = [value for value in env.get("WSLENV", "").split(":") if value]
        env["WSLENV"] = ":".join(existing + [name for name in forwarded if name not in existing])
    if str((af2_config or {}).get("backend", "native")).lower() != "wsl":
        executable = resolve_colabfold_executable(af2_config, project_dir)
        if executable:
            env["PATH"] = str(Path(executable).parent) + os.pathsep + env.get("PATH", "")
    return env
