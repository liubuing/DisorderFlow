# Create a CUDA-enabled PyTorch environment for DisorderFlow GPU work.
# Run from the project root in PowerShell.

param(
    [string]$VenvPath = ".venv_cuda",
    [string]$PythonVersion = "3.14",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu130"
)

py -$PythonVersion -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install torch --index-url $TorchIndex
& "$VenvPath\Scripts\python.exe" -m pip install -e ".[ui,publication]" -i https://pypi.org/simple
& "$VenvPath\Scripts\python.exe" -m pip install anarcii "biopython==1.87" -i https://pypi.org/simple
& "$VenvPath\Scripts\python.exe" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

& "$VenvPath\Scripts\python.exe" scripts/pipeline/check_gpu_readiness.py --python "$VenvPath\Scripts\python.exe" --out outputs/gpu_readiness_cuda_v1
