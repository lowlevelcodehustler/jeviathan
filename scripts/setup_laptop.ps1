# Jeviathan — RTX 4050 Laptop (6GB) setup.
# Option A: Ollama with Llama-3.1-8B (quantized ~4.9GB, fits 6GB with small ctx).
$ErrorActionPreference = "Stop"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found. Install from https://ollama.com/download then re-run." -ForegroundColor Yellow
} else {
    Write-Host "==> Pulling llama3.1:8b (~4.9GB)..."
    ollama pull llama3.1:8b
    Write-Host "==> Starting Ollama server (if not running)..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
}

Write-Host ""
Write-Host "Option B (uses your existing local weights, no download):"
Write-Host "  pip install torch transformers bitsandbytes fastapi uvicorn"
Write-Host "  python scripts/transformers_server.py --model-dir E:\bfc-today-test-weights\model_run --port 8200"
Write-Host "  then set JEVIATHAN_BASE_URL=http://localhost:8200/v1 and JEVIATHAN_MODEL=llama3.1-8b-local"
