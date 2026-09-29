#!/usr/bin/env bash
# Jeviathan - consolidated guided installer (cross-platform bash).
# Probes hardware, recommends a tier, and delegates component installs to the
# per-tier scripts. Parallel PowerShell version: scripts/setup.ps1
#
# Tiers:
#   A  Ollama + Llama-3.1-8B (~5GB)          -> setup_laptop (Option A)
#   B  native Torch NF4 shim (local weights) -> setup_laptop (Option B)
#   C  vLLM Qwen3.8-27B (>=24GB VRAM)       -> setup_5090.sh (WSL2 or fork)
#
# Works on Linux/WSL and native Windows (Git Bash / MINGW64).
#
# Usage:
#   bash scripts/setup.sh                          # probe + interactive menu
#   JEVIATHAN_SETUP_TIER=C bash scripts/setup.sh   # skip the menu (CI)
#   bash scripts/setup.sh --tier C --persist       # flags too

set -uo pipefail   # no -e: probes are guarded so a missing tool never aborts
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

TIER="${JEVIATHAN_SETUP_TIER:-}"
PERSIST=0
PORT=8200
MODEL_DIR="E:/bfc-today-test-weights/model_run"

# --- args --------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --tier)      TIER="${2:?--tier needs a value}"; shift 2 ;;
    --port)      PORT="${2:?--port needs a value}"; shift 2 ;;
    --model-dir) MODEL_DIR="${2:?--model-dir needs a value}"; shift 2 ;;
    --persist)   PERSIST=1; shift ;;
    -h|--help)   sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$*"; }
warn() { printf '!! %s\n' "$*" >&2; }

# --- probe -------------------------------------------------------------------
step "Probing hardware ..."
GPU_NAME=""; VRAM_GB=0; DRIVER_VER="?"; CUDA_VER="?"; GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  # name/memory.total/driver_version are reliable across drivers. cuda_version is
  # NOT a valid --query-gpu field on all drivers, so read it from the header.
  line="$(nvidia-smi --query-gpu=name,memory.total,driver_version \
            --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  if [ -n "$line" ]; then
    IFS=',' read -r f1 f2 f3 <<<"$line"
    GPU_NAME="${f1# }"; MEM_MIB="${f2// /}"; DRIVER_VER="${f3# }"
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | wc -l | tr -d ' ')"
    VRAM_GB="$(awk -v m="$MEM_MIB" 'BEGIN{printf "%.1f", m/1024}')"
  fi
  # Header label varies by driver ("CUDA Version:" vs "CUDA UMD Version:"); match both.
  CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oiE 'cuda[^|]*version:[^0-9]*[0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
  [ -z "$CUDA_VER" ] && CUDA_VER="?"
fi

# RAM (GB): Linux via /proc, Windows via powershell fallback.
RAM_GB=""
if [ -r /proc/meminfo ]; then
  RAM_GB="$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)"
elif command -v powershell >/dev/null 2>&1; then
  RAM_GB="$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,0)" 2>/dev/null | tr -d '[:space:]')"
fi

# Disk free (GB) at repo root.
DISK_GB="$(df -Pm "$REPO" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024}')"

HAS_OLLAMA=0; command -v ollama >/dev/null 2>&1 && HAS_OLLAMA=1
case "$(uname -s)" in
  MSYS*|MINGW*|CYGWIN*) VENV_PY=".venv/Scripts/python.exe" ;;
  *)                    VENV_PY=".venv/bin/python" ;;
esac
HAS_VENV=0; [ -f "$VENV_PY" ] && HAS_VENV=1

if [ -n "$GPU_NAME" ]; then
  printf '  GPU   : %s x%s | VRAM %s GB | driver %s | CUDA %s\n' \
    "$GPU_NAME" "$GPU_COUNT" "$VRAM_GB" "$DRIVER_VER" "$CUDA_VER"
else
  echo "  GPU   : none (CPU-only)"
fi
printf '  RAM   : %s GB\n' "${RAM_GB:-?}"
printf '  Disk  : %s GB free at repo root\n' "${DISK_GB:-?}"
printf '  Ollama: %s\n' "$([ "$HAS_OLLAMA" -eq 1 ] && echo installed || echo "not found")"
printf '.venv : %s\n' "$([ "$HAS_VENV" -eq 1 ] && echo present || echo absent)"

# --- recommend ----------------------------------------------------------------
RECOMMENDED="A"; WHY=""
if [ -n "$GPU_NAME" ] && awk -v v="$VRAM_GB" 'BEGIN{exit !(v>=24)}'; then
  RECOMMENDED="C"; WHY=">=24 GB VRAM -> vLLM Qwen3.8-27B fits one GPU"
elif [ -n "$GPU_NAME" ]; then
  RECOMMENDED="A"; WHY="<24 GB VRAM -> Ollama/Torch dev tier (C needs ~24 GB)"
else
  RECOMMENDED="B"; WHY="no NVIDIA GPU detected -> CPU Torch shim (slow); A also works on CPU"
fi

# --- choose -------------------------------------------------------------------
if [ -z "$TIER" ]; then
  if [ -t 0 ]; then   # interactive stdin available
    echo ""
    echo "Tiers:"
    echo "  [A] Ollama + Llama-3.1-8B (~5GB)          -> setup_laptop (Option A)"
    echo "  [B] native Torch shim (local weights)     -> setup_laptop (Option B)"
    echo "  [C] vLLM Qwen3.8-27B (>=24GB VRAM)       -> setup_5090.sh (WSL2 or fork)"
    echo ""
    read -r -p "Choose [A/B/C] (default: $RECOMMENDED): " pick || true
    TIER="${pick:-$RECOMMENDED}"
  else
    warn "Non-interactive run requires --tier A|B|C (or JEVIATHAN_SETUP_TIER)."
    exit 1
  fi
fi
TIER="$(printf '%s' "$TIER" | tr '[:lower:]' '[:upper:]')"

step "Selected tier: $TIER"
warn "Recommendation was: $RECOMMENDED ($WHY)"

# --- delegate -----------------------------------------------------------------
run_tier_ab() {  # $1 = A|B ; PowerShell on Windows, inline steps otherwise
  local opt="$1" rc=0
  if command -v powershell >/dev/null 2>&1; then
    step "Tier $opt via setup_laptop.ps1 ..."
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_laptop.ps1 \
      -Option "$opt" -Port "$PORT" -ModelDir "$MODEL_DIR" || rc=$?
  else
    if [ "$opt" = "A" ]; then
      step "Tier A: Ollama + llama3.1:8b (no PowerShell found; inline)"
      if ! command -v ollama >/dev/null 2>&1; then
        warn "Ollama not found; install from https://ollama.com/download and re-run."
      else
        ollama pull llama3.1:8b || rc=$?
      fi
    else
      step "Tier B: native Torch shim (no PowerShell found; inline)"
      if [ ! -f "$VENV_PY" ]; then
        warn "No .venv at $VENV_PY; create one first, then re-run."
      else
        "$VENV_PY" -m pip install --upgrade pip || rc=$?
        "$VENV_PY" -m pip install torch transformers bitsandbytes fastapi "uvicorn[standard]" || rc=$?
      fi
    fi
  fi
  [ "$rc" -ne 0 ] && warn "Tier $opt step exited with code $rc (see output above)."
}

case "$TIER" in
  A|B) run_tier_ab "$TIER" ;;
  C)
    step "Tier C: vLLM via setup_5090.sh (WSL2 or native fork)"
    bash scripts/setup_5090.sh || warn "setup_5090.sh exited non-zero (see its output above)."
    ;;
  *)   warn "Unknown tier '$TIER'."; exit 2 ;;
esac

# --- profile pointer ----------------------------------------------------------
case "$TIER" in
  A) PROFILE="laptop-4050";         ENVLINE="JEVIATHAN_BASE_URL=http://localhost:11434/v1 JEVIATHAN_MODEL=llama3.1:8b" ;;
  B) PROFILE="laptop-4050-logprob"; ENVLINE="JEVIATHAN_BASE_URL=http://localhost:$PORT/v1 JEVIATHAN_MODEL=llama3.1-8b-local" ;;
  C) PROFILE="rtx5090";             ENVLINE="JEVIATHAN_BASE_URL=http://localhost:8001/v1 JEVIATHAN_MODEL=jeviathan-qwen3.8-27b" ;;
esac

step "Done. Use this profile + env for tier $TIER:"
echo "  JEVIATHAN_PROFILE=$PROFILE"
echo "  $ENVLINE"

if [ "$PERSIST" -eq 1 ]; then
  printf '%s' "$PROFILE" > .jeviathan_profile
  echo "  (persisted to .jeviathan_profile)"
fi
