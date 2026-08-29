$ErrorActionPreference = "Stop"

$runtime = "D:\DisorderFlowRuntime"
$project = "D:\biological\DisorderFlow"
$python = "$runtime\Python314\python.exe"

$env:PYTHONUSERBASE = "$runtime\PythonUser"
$env:PYTHONPYCACHEPREFIX = "$runtime\cache\pycache"
$env:PIP_CACHE_DIR = "$runtime\cache\pip"
$env:TORCH_HOME = "$runtime\cache\torch"
$env:HF_HOME = "$runtime\cache\huggingface"
$env:HUGGINGFACE_HUB_CACHE = "$runtime\cache\huggingface\hub"
$env:COLABFOLD_CACHE = "$runtime\cache\colabfold"
$env:CUDA_CACHE_PATH = "$runtime\cache\cuda"
$env:NUMBA_CACHE_DIR = "$runtime\cache\numba"
$env:TEMP = "$runtime\temp"
$env:TMP = "$runtime\temp"
$env:DISORDERFLOW_AF2_WSL_DISTRO = "Ubuntu-24.04-D"
$env:DISORDERFLOW_WSL_DISTRO = "Ubuntu-24.04-D"
$env:XLA_PYTHON_CLIENT_PREALLOCATE = "false"
$env:XLA_PYTHON_CLIENT_MEM_FRACTION = "0.85"

if (-not (Test-Path -LiteralPath $python)) {
    throw "D-drive Python runtime not found: $python"
}

& $python "$project\manage.py" @args
exit $LASTEXITCODE
