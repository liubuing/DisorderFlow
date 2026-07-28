# Create a CUDA-enabled PyTorch environment for DisorderFlow GPU work.
# Run from the project root in PowerShell.

param(
    [string]$VenvPath = ".venv_cuda",
    [string]$PythonVersion = "3.11",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)

py -$PythonVersion -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install torch --index-url $TorchIndex
& "$VenvPath\Scripts\python.exe" -m pip install numpy pyyaml biopython -i https://pypi.org/simple
& "$VenvPath\Scripts\python.exe" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

python scripts/pipeline/check_gpu_readiness.py --python "$VenvPath\Scripts\python.exe" --out outputs/gpu_readiness_cuda_v1
