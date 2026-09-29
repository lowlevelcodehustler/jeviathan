# Jeviathan - RTX 4050 Laptop (6GB) setup. Installs the components for one or
# both backend options, then prints the exact env/profile to use with Jeviathan.
#
#   Option A - Ollama + Llama-3.1-8B (quantized ~4.9GB, fits 6GB w/ small ctx)
#       -> served at http://localhost:11434/v1 as "llama3.1:8b"
#          matches profiles/laptop-4050.yaml
#   Option B - native Torch shim over your local weights (4-bit NF4, ~5GB VRAM)
#       -> scripts/transformers_server.py on :8200 as "llama3.1-8b-local"
#          matches profiles/laptop-4050-logprob.yaml
#
# Usage:
#   .\scripts\setup_laptop.ps1                 # interactive menu (A / B / both)
#   .\scripts\setup_laptop.ps1 -Option A      # Ollama only
#   .\scripts\setup_laptop.ps1 -Option B      # native Torch shim only
#   .\scripts\setup_laptop.ps1 -Option both   # install components for both
#   .\scripts\setup_laptop.ps1 -Option B -ModelDir D:\weights -Port 8200
#
# Non-interactive (CI): always pass -Option.

param(
    [ValidateSet("A", "B", "both")] [string]$Option = "",
    [int]$Port = 8200,
    [string]$ModelDir = "E:\bfc-today-test-weights\model_run",
    [string]$OllamaModel = "llama3.1:8b"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot   # scripts/.. -> repo root

function Write-Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Note-Warn($m)  { Write-Host "!! $m" -ForegroundColor Yellow }

# --- resolve which options to run -------------------------------------------
$wantA = ($Option -eq "A") -or ($Option -eq "both")
$wantB = ($Option -eq "B") -or ($Option -eq "both")
if (-not $wantA -and -not $wantB) {
    if ([Environment]::UserInteractive) {
        Write-Host "Jeviathan laptop setup - pick a backend option:" -ForegroundColor Cyan
        Write-Host "  A)     Ollama + Llama-3.1-8B   (download ~4.9GB, easiest)"
        Write-Host "  B)     Native Torch shim       (uses local weights in $ModelDir)"
        Write-Host "  both)  install components for both"
        $choice = (Read-Host "Choice [A/B/both]").Trim().ToLower()
        switch ($choice) {
            "a"    { $wantA = $true }
            "b"    { $wantB = $true }
            "both" { $wantA = $true; $wantB = $true }
            default { Note-Warn "No option chosen; exiting."; exit 1 }
        }
    } else {
        Note-Warn "Non-interactive run requires -Option A|B|both."
        exit 1
    }
}

# --- Option A: Ollama --------------------------------------------------------
if ($wantA) {
    Write-Step "Option A - Ollama + $OllamaModel"
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Note-Warn "Ollama not found on PATH."
        Write-Host "   Install it from https://ollama.com/download  (or: winget install Ollama.Ollama)" -ForegroundColor Yellow
        Write-Host "   then re-run this script with -Option A." -ForegroundColor Yellow
    } else {
        Write-Step "Pulling $OllamaModel (~4.9GB)..."
        ollama pull $OllamaModel
        Write-Step "Starting Ollama server if not already running..."
        # `ollama serve` is a harmless no-op if the daemon is already up.
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    }
    Write-Host ""
    Write-Host "Option A ready. Point Jeviathan at Ollama:" -ForegroundColor Green
    Write-Host "  JEVIATHAN_BASE_URL=http://localhost:11434/v1"
    Write-Host "  JEVIATHAN_MODEL=$OllamaModel"
    Write-Host "  (matches profiles/laptop-4050.yaml)"
}

# --- Option B: native Torch shim --------------------------------------------
if ($wantB) {
    Write-Step "Option B - native Torch shim over local weights"
    if (-not (Test-Path $ModelDir)) {
        Note-Warn "Model dir not found: $ModelDir"
        Write-Host "   Pass -ModelDir <path> pointing at your local Llama weights." -ForegroundColor Yellow
    }

    # Prefer the repo .venv (isolated, consistent with the 5090 setup); fall back
    # to the current interpreter if there is none.
    $VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $VenvPy) { $py = $VenvPy } else { $py = "python"; Note-Warn "No .venv found; using system python." }

    Write-Step "Installing Torch + serving deps via: $py"
    Write-Host "   (torch pulls the CUDA build - a large download on first run.)" -ForegroundColor Yellow
    & $py -m pip install --upgrade pip
    # transformers_server.py needs torch + transformers (+ bitsandbytes for NF4);
    # fastapi/uvicorn come from requirements.txt.
    & $py -m pip install torch transformers bitsandbytes fastapi "uvicorn[standard]"

    Write-Host ""
    Write-Host "Option B ready. Start the shim, then point Jeviathan at it:" -ForegroundColor Green
    Write-Host "  python scripts/transformers_server.py --model-dir `"$ModelDir`" --port $Port"
    Write-Host "  JEVIATHAN_BASE_URL=http://localhost:$Port/v1"
    Write-Host "  JEVIATHAN_MODEL=llama3.1-8b-local"
    Write-Host "  (matches profiles/laptop-4050-logprob.yaml)"
}

Write-Host "`nDone." -ForegroundColor Cyan
