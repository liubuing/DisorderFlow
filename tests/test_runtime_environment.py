from pathlib import Path

from modules.runtime_environment import build_colabfold_command, windows_to_wsl_path


def test_windows_path_maps_to_wsl_mount():
    mapped = windows_to_wsl_path(Path("D:/biological/DisorderFlow/input.fasta"))
    assert mapped == "/mnt/d/biological/DisorderFlow/input.fasta"


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
