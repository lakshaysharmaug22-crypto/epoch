# EPOCH on a Windows PC: setup -> doctor -> demo -> export -> dashboard.
# Usage (PowerShell, from the repo root):
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\epoch-pc.ps1 -Step setup      # once
#   $env:ANTHROPIC_API_KEY = "sk-ant-..."
#   .\scripts\epoch-pc.ps1 -Step doctor
#   .\scripts\epoch-pc.ps1 -Step demo       # the long one; log goes to demo.log
#   .\scripts\epoch-pc.ps1 -Step web        # rebuild dashboard with the new numbers
param([ValidateSet("check", "setup", "doctor", "test", "demo", "web", "all")] [string]$Step = "all",
      [int]$Budget = 60, [int]$Seeds = 3)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$epoch = Join-Path $root ".venv\Scripts\epoch.exe"
$gpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
New-Item -ItemType Directory -Force logs | Out-Null
Start-Transcript -Path "logs\$Step.log" -Force | Out-Null

function Check {
  foreach ($c in "py", "python", "git", "node", "npm", "nvidia-smi") {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { Write-Host "OK   $c -> $($cmd.Source)" } else { Write-Host "MISS $c" }
  }
  if (Get-Command py -ErrorAction SilentlyContinue) { py -0p }
  if (Get-Command python -ErrorAction SilentlyContinue) { python --version }
  if (Get-Command git -ErrorAction SilentlyContinue) { git --version }
  if (Get-Command node -ErrorAction SilentlyContinue) { node --version }
  if ($gpu) { nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader }
  Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | Format-Table -AutoSize
  Write-Host "ANTHROPIC_API_KEY set: $([bool]$env:ANTHROPIC_API_KEY)"
}

function Setup {
  if (-not (Test-Path $py)) {
    # pick a real 3.11/3.12 interpreter path (the py launcher can't always resolve -3.12)
    $base = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
      foreach ($line in (py -0p 2>$null)) { if ($line -match '-V:3\.1[12]\S*\s+\*?\s*(\S.*python\.exe)') { $base = $Matches[1].Trim(); break } }
    }
    if (-not $base) { $base = (Get-Command python).Source }
    Write-Host "Creating venv with $base"
    & $base --version
    & $base -m venv .venv
  }
  & $py -m pip install -U pip wheel
  if ($gpu) {
    Write-Host "NVIDIA GPU detected:" -ForegroundColor Green; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    & $py -m pip install -e ".[dev,gpu]"
  } else {
    Write-Host "No NVIDIA GPU: CPU install (Claude agent still works)." -ForegroundColor Yellow
    & $py -m pip install -e ".[dev]"
  }
}
function Doctor { & $epoch doctor }
function Test { & $py -m pytest -q }
function Demo {
  if (-not $env:ANTHROPIC_API_KEY) { Write-Host "ANTHROPIC_API_KEY not set - agents run in heuristic mode." -ForegroundColor Yellow }
  & $epoch demo --workloads triage,rag --budget $Budget --seeds $Seeds 2>&1 | Tee-Object -FilePath demo.log
}
function Web {
  Push-Location web; npm install; npm run build; npm run build:standalone; Pop-Location
  Write-Host "Dashboard: web\out (Vercel) and web\standalone\index.html" -ForegroundColor Green
}
switch ($Step) {
  "check" { Check }
  "setup" { Setup } "doctor" { Doctor } "test" { Test } "demo" { Demo } "web" { Web }
  "all" { Setup; Doctor; Test; Demo; Web }
}
Stop-Transcript | Out-Null
Write-Host "DONE: $Step (log: logs\$Step.log)" -ForegroundColor Green
