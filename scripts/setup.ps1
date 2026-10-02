# Jeviathan - consolidated guided installer (Windows / PowerShell).
# Probes hardware, recommends a tier, and delegates component installs to the
# per-tier scripts. Parallel bash version: scripts/setup.sh
#
# Tiers:
#   A  Ollama + Llama-3.1-8B (~5GB)          -> setup_laptop.ps1 -Option A
#   B  native Torch NF4 shim (local weights) -> setup_laptop.ps1 -Option B
#   C  vLLM Qwen3.8-27B (>=24GB VRAM)       -> setup_5090.sh (WSL2 or fork)
#
# Usage:
#   .\scripts\setup.ps1                          # probe + interactive menu
#   .\scripts\setup.ps1 -Tier C                  # skip the menu, go straight to C
#   .\scripts\setup.ps1 -Persist                 # also write .jeviathan_profile
#   $env:JEVIATHAN_SETUP_TIER="C"; .\scripts\setup.ps1    # non-interactive (CI)

param(
    [ValidateSet("A", "B", "C")] [string]$Tier = "",
    [switch]$Persist,
    [int]$Port = 8200,
    [string]$ModelDir = $env:JEVIATHAN_MODEL_DIR
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Model dir fallback for tier B: .jeviathan_model_dir at repo root (gitignored).
if (-not $ModelDir) {
    $localFile = Join-Path $RepoRoot ".jeviathan_model_dir"
    if (Test-Path $localFile) { $ModelDir = (Get-Content $localFile -First 1).Trim() }
}

function Write-Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Note-Warn($m)  { Write-Host "!! $m" -ForegroundColor Yellow }

# --- probe -------------------------------------------------------------------
Write-Step "Probing hardware ..."
$gpuName = ""; $vramGB = 0; $driverVer = "?"; $cudaVer = "?"; $gpuCount = 0
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        # name/memory.total/driver_version are reliable across drivers. cuda_version
        # is NOT a valid --query-gpu field on all drivers, so read it from the header.
        $lines = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null
        if ($lines) {
            $arr = @($lines)
            $gpuCount = $arr.Count
            $f = @($arr[0] -split "," | ForEach-Object { $_.Trim() })
            $gpuName   = $f[0]
            $vramGB    = [math]::Round([double]$f[1] / 1024, 1)
            if ($f.Count -ge 3) { $driverVer = $f[2] }
        }
        # Header label varies by driver ("CUDA Version:" vs "CUDA UMD Version:"); match both.
        $hdr = (& nvidia-smi 2>$null) -join "`n"
        if ($hdr -match '(?i)cuda[^\r\n|]*version:\s*(\d+\.\d+)') { $cudaVer = $Matches[1] }
    } catch { Note-Warn "nvidia-smi present but failed: $_" }
}

$ramGB = "?"
try { $ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 0) } catch {}
$diskFreeGB = "?"
try {
    $letter = (Split-Path -Qualifier $RepoRoot)
    if ($letter) {
        # DriveInfo.Free returns 0 on some systems (OneDrive/virtualized volumes);
        # Get-PSDrive reports the real value. Prefer whichever is non-zero so we
        # never display a misleading zero.
        $driveName = $letter.TrimEnd(':')
        $candidates = @()
        try { $candidates += (Get-PSDrive -Name $driveName -ErrorAction Stop).Free } catch {}
        try { $candidates += (New-Object System.IO.DriveInfo($letter)).Free } catch {}
        $nonZero = @($candidates | Where-Object { $_ -gt 0 })
        if ($nonZero) { $diskFreeGB = [math]::Round($nonZero[0] / 1GB, 0) }
    }
} catch {}

$hasOllama = [bool](Get-Command ollama -ErrorAction SilentlyContinue)
$venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$hasVenv = Test-Path $venvPy

if ($gpuName) {
    Write-Host ("  GPU   : {0} x{1} | VRAM {2} GB | driver {3} | CUDA {4}" -f $gpuName, $gpuCount, $vramGB, $driverVer, $cudaVer)
} else {
    Write-Host "  GPU   : none (CPU-only)"
}
Write-Host ("  RAM   : {0} GB" -f $ramGB)
Write-Host ("  Disk  : {0} GB free at repo root" -f $diskFreeGB)
Write-Host ("  Ollama: {0}" -f $(if ($hasOllama) { "installed" } else { "not found" }))
Write-Host (".venv : {0}" -f $(if ($hasVenv) { "present" } else { "absent" }))

# --- recommend ----------------------------------------------------------------
$recommended = "A"; $why = ""
if ($gpuName -and [double]$vramGB -ge 24) {
    $recommended = "C"; $why = ">=24 GB VRAM -> vLLM Qwen3.8-27B fits one GPU"
} elseif ($gpuName) {
    $recommended = "A"; $why = "<24 GB VRAM -> Ollama/Torch dev tier (C needs ~24 GB)"
} else {
    $recommended = "B"; $why = "no NVIDIA GPU detected -> CPU Torch shim (slow); A also works on CPU"
}

# --- choose -------------------------------------------------------------------
# NOTE: never assign a raw env value to $Tier directly - the parameter carries
# [ValidateSet], and assigning $null or an invalid value raises ValidateSetFailure.
$envRaw = $env:JEVIATHAN_SETUP_TIER
$envTier = if ($envRaw) { $envRaw.Trim().ToUpper() } else { "" }
# Validate BEFORE assigning: $Tier carries [ValidateSet], so even a valid-looking
# but wrong value (e.g. JEVIATHAN_SETUP_TIER=Z) would raise ValidateSetFailure.
if (-not $Tier -and $envTier -and ($envTier -notin @("A", "B", "C"))) {
    Note-Warn "Invalid JEVIATHAN_SETUP_TIER '$envTier' (expected A, B or C)."
    exit 2
}
if (-not $Tier -and $envTier) { $Tier = $envTier }
# UserInteractive alone is not enough: it can be true while stdin is redirected,
# in which case Read-Host returns $null. Only prompt with a real console.
$interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
if (-not $Tier) {
    if ($interactive) {
        Write-Host ""
        Write-Host "Tiers:" -ForegroundColor Cyan
        Write-Host "  [A] Ollama + Llama-3.1-8B (~5GB)          -> setup_laptop.ps1 -Option A"
        Write-Host "  [B] native Torch NF4 shim (local weights) -> setup_laptop.ps1 -Option B"
        Write-Host "  [C] vLLM Qwen3.8-27B (>=24GB VRAM)       -> setup_5090.sh (WSL2 or fork)"
        Write-Host ""
        $raw = Read-Host "Choose [A/B/C] (default: $recommended)"
        $pick = if ($raw) { $raw.Trim().ToUpper() } else { "" }
        if ($pick -eq "") { $Tier = $recommended } else { $Tier = $pick }
    } else {
        Note-Warn "Non-interactive run requires -Tier A|B|C."
        exit 1
    }
}
if ($Tier -notin @("A", "B", "C")) {
    Note-Warn "Invalid tier '$Tier' (expected A, B or C)."
    exit 2
}

Write-Step "Selected tier: $Tier"
Note-Warn "Recommendation was: $recommended ($why)"

# --- delegate -----------------------------------------------------------------
switch ($Tier) {
    "A" { & (Join-Path $RepoRoot "scripts\setup_laptop.ps1") -Option A }
    "B" {
        $childArgs = @("-Option", "B", "-Port", "$Port")
        if ($ModelDir) { $childArgs += @("-ModelDir", $ModelDir) }
        & (Join-Path $RepoRoot "scripts\setup_laptop.ps1") @childArgs
    }
    "C" {
        Write-Host ""
        Write-Host "Tier C (vLLM) needs a Linux/WSL2 environment for official vLLM." -ForegroundColor Yellow
        $runNow = $false
        if ($interactive) {
            $raw = Read-Host "Run 'bash scripts/setup_5090.sh' now? [Y/n]"
            $ans = if ($raw) { $raw.Trim() } else { "" }
            $runNow = ($ans -eq "" -or $ans -match '^[Yy]')
        }
        if ($runNow -and (Get-Command bash -ErrorAction SilentlyContinue)) {
            & bash scripts/setup_5090.sh
        } else {
            Write-Host "   Manual: elevated PowerShell -> wsl --install -d Ubuntu ; then inside WSL:" -ForegroundColor Yellow
            Write-Host "     cd /mnt/c/jeviathan && bash scripts/setup_5090.sh" -ForegroundColor Yellow
        }
    }
}

# --- profile pointer ----------------------------------------------------------
$profileMap = @{ A = "laptop-4050"; B = "laptop-4050-logprob"; C = "rtx5090" }
$envMap = @{
    A = "JEVIATHAN_BASE_URL=http://localhost:11434/v1 JEVIATHAN_MODEL=llama3.1:8b"
    B = "JEVIATHAN_BASE_URL=http://localhost:$Port/v1 JEVIATHAN_MODEL=llama3.1-8b-local"
    C = "JEVIATHAN_BASE_URL=http://localhost:8001/v1 JEVIATHAN_MODEL=jeviathan-qwen3.8-27b"
}
Write-Host ""
Write-Step ("Done. Use this profile + env for tier {0}:" -f $Tier) -ForegroundColor Green
Write-Host ("  JEVIATHAN_PROFILE={0}" -f $profileMap[$Tier])
Write-Host ("  {0}" -f $envMap[$Tier])

if ($Persist) {
    $pf = Join-Path $RepoRoot ".jeviathan_profile"
    Set-Content -Path $pf -Value $profileMap[$Tier] -Encoding ascii
    Write-Host "  (persisted to .jeviathan_profile)"
}
