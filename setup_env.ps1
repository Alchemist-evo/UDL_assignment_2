# Creates .venv and installs everything on Windows.
# usage: powershell -ExecutionPolicy Bypass -File setup_env.ps1 [-Backend auto|cuda|xpu|cpu]
param([string]$Backend = "auto")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip wheel

if ($Backend -eq "auto") {
    $gpus = (Get-CimInstance Win32_VideoController).Name -join ";"
    if ($gpus -match "NVIDIA") { $Backend = "cuda" }
    elseif ($gpus -match "Intel\(R\) Arc") { $Backend = "xpu" }
    else { $Backend = "cpu" }
}
Write-Host ">> installing PyTorch for backend: $Backend"
switch ($Backend) {
    "cuda" { & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 }
    "xpu"  { & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu }
    "cpu"  { & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu }
}
& $py -m pip install -r requirements.txt
& $py -m ipykernel install --user --name udl-a2 --display-name "Python (udl-a2)"
& $py -c "import torch; x=hasattr(torch,'xpu') and torch.xpu.is_available(); print('CUDA:', torch.cuda.is_available(), '| XPU:', x, torch.xpu.get_device_name(0) if x else '')"
Write-Host ">> done. activate with:  .\.venv\Scripts\Activate.ps1"
