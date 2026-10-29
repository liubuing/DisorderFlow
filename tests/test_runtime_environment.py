"""运行环境适配测试：验证 Windows→WSL 路径映射与 ColabFold 命令拼装。

守护 modules.runtime_environment 的跨平台调用：
Windows 路径必须映射为 /mnt/ 盘符路径、
WSL 后端命令必须经由指定发行版与配置环境的可执行文件，
且额外参数原样追加——防止环境配置被静默忽略或路径映射错误导致调用失败。
"""
from pathlib import Path

from modules.runtime_environment import build_colabfold_command, windows_to_wsl_path


# 路径映射：D:/ 盘路径必须转成 /mnt/d/ 形式——防止 WSL 内找不到输入文件。
def test_windows_path_maps_to_wsl_mount():
    mapped = windows_to_wsl_path(Path("D:/biological/DisorderFlow/input.fasta"))
    assert mapped == "/mnt/d/biological/DisorderFlow/input.fasta"


# 命令拼装：发行版、环境内可执行文件与附加参数必须按配置逐项进入命令——
# 防止配置被绕过（直接用系统 colabfold）或参数丢失。
def test_wsl_colabfold_command_uses_configured_environment(tmp_path):
    project = Path("D:/biological/DisorderFlow")
    command = build_colabfold_command(
        {
            "backend": "wsl",
            "distribution": "Ubuntu-24.04-D",
            "environment": "venv_wsl",
            "executable": "colabfold_batch",
        },
        project,
        project / "input.fasta",
        project / "results",
        ["--num-models", "1"],
    )
    assert command[:5] == ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "/mnt/d/biological/DisorderFlow/venv_wsl/bin/colabfold_batch"]
    assert command[-2:] == ["--num-models", "1"]
