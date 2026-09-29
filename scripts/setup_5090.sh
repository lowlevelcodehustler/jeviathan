#!/usr/bin/env bash
# Jeviathan — RTX 5090 (32GB) setup: serve Qwen3.8-27B with vLLM.
#
# PLATFORM-AWARE, because official vLLM is Linux-only and the PyPI Windows wheel
# ships WITHOUT its compiled CUDA binding (`vllm._C_stable_libtorch`). That is
# exactly why a native-Windows `pip install vllm` "succeeds" then crashes at
# serve time with:
#     ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'
#
# This script therefore branches on platform:
#   * Linux / WSL2  -> installs official vLLM in a local .venv and serves it.
#                       (WSL2 is the officially supported Windows path; CUDA
#                        graphs work on Blackwell sm_120 with WSL2 >= 2.7.)
#   * native Windows-> fails fast: detects MSIX Python, prints a copy-paste
#                       guide for the two working paths (WSL2, or the
#                       SystemPanic/vllm-windows fork), and exits without
#                       wasting time polling a broken install. Set
#                       JEVIATHAN_5090_NATIVE=fork to let it attempt the
#                       native-fork wheel install automatically.
#
# Everything runs inside a local .venv so you never fight PATH quirks — in
# particular the Windows Store (MSIX) Python, whose pip installs console scripts
# into a folder that is NOT on PATH (`vllm: command not found` despite a
# successful install). We launch via `python -m vllm serve`, which needs no PATH.
#
# Env overrides:
#   JEVIATHAN_5090_MODEL    model id (default Inferact/Qwen3.8-27B-NVFP4,
#                           ~24.6 GiB — fits one Blackwell GPU; fallback
#                           Qwen/Qwen3.8-27B-FP8)
#   JEVIATHAN_5090_PORT     default 8001
#   JEVIATHAN_5090_MAX_LEN  max model len, default 32768 (use 16384 for FP8)
#   JEVIATHAN_5090_KV_DTYPE fp8 (default; halves KV memory on Blackwell) or
#                           auto (omit the flag — safer under WSL2 where FP8
#                           may run through an emulated path)
#   JEVIATHAN_5090_NATIVE   set to "fork" on native Windows to auto-attempt the
#                           SystemPanic/vllm-windows wheel install (needs CPython
#                           3.12 from python.org + CUDA 13). Default: guide only.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${JEVIATHAN_5090_MODEL:-Inferact/Qwen3.8-27B-NVFP4}"
PORT="${JEVIATHAN_5090_PORT:-8001}"
MAX_LEN="${JEVIATHAN_5090_MAX_LEN:-32768}"
KV_DTYPE="${JEVIATHAN_5090_KV_DTYPE:-fp8}"
ALIAS="jeviathan-qwen3.8-27b"

# --- 0) platform + interpreter detection -------------------------------------
UNAME_S="$(uname -s)"
case "$UNAME_S" in
  Linux*)              PLATFORM=linux ;;    # includes WSL2 — official vLLM works here
  MSYS*|MINGW*|CYGWIN*) PLATFORM=windows ;; # native Windows (Git Bash / MINGW64)
  *)                   PLATFORM=unknown ;;
esac

if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
PY_PATH="$(command -v "$PY" || true)"
echo "==> Using $("$PY" --version 2>&1) at ${PY_PATH:-?}"

# MSIX (Windows Store) Python detection. Its pip puts console scripts in a dir
# that is NOT on PATH, and the vLLM Windows wheel it resolves lacks the native
# CUDA binding — both root causes of the failures this script exists to fix.
if printf '%s' "$PY_PATH" | grep -qi 'WindowsApps'; then
  echo "!! WARNING: your default Python is the Windows Store (MSIX) build."
  echo "   That is why vLLM installs but crashes with a missing native module,"
  echo "   and why \`vllm\` is not on PATH. Install official CPython from"
  echo "   https://www.python.org/downloads/ (3.12 for the fork path) and put it"
  echo "   first on PATH, then re-run this script."
fi

# --- native Windows: official vLLM cannot run here ---------------------------
if [ "$PLATFORM" = "windows" ]; then
  cat <<'GUIDE'
============================================================================
 Official vLLM is Linux-only. On native Windows the PyPI `vllm` wheel has no
 compiled CUDA binding, so it installs but crashes at serve time with:

     ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'

 Two working paths (pick one):

 PATH A — WSL2 + official vLLM   [recommended: officially supported]
   1. In an ELEVATED PowerShell:      wsl --install -d Ubuntu
      Reboot when prompted, then `wsl --update` to get WSL2 >= 2.7 (needed for
      CUDA graphs on Blackwell sm_120).
   2. Open the new Ubuntu terminal and run this same script there:
         cd /mnt/c/jeviathan && bash scripts/setup_5090.sh
      It auto-detects Linux and installs official vLLM for you.
   Notes for Blackwell under WSL2 (see vllm-project/vllm#37242):
     * Remove Tailscale from the distro if present — it races CUDA at boot:
         sudo apt remove tailscale
     * FP8/NVFP4 may run through an emulated path under WSL2's dxgkrnl. It
       still works, just slower than native tensor cores. If you need full
       speed, use PATH B (or set JEVIATHAN_5090_KV_DTYPE=auto).

 PATH B — native Windows fork   [full Blackwell perf, no virtualization tax]
   The SystemPanic/vllm-windows fork ships native Windows wheels with CUDA 13 +
   Blackwell (sm_120) support: https://github.com/SystemPanic/vllm-windows
     1. Install official CPython 3.12 from python.org (NOT the Store build).
     2. Ensure CUDA 13 + a matching PyTorch — the wheel pins exact versions, so
        check that release's notes for which Python/Torch/CUDA it targets.
     3. Download the latest wheel:
         https://github.com/SystemPanic/vllm-windows/releases/latest
     4. In a fresh .venv from CPython 3.12:   pip install <downloaded.whl>
     5. Serve:  python -m vllm serve <model> --served-model-name jeviathan-qwen3.8-27b ...

   Or let this script attempt PATH B automatically (best-effort wheel fetch):
         JEVIATHAN_5090_NATIVE=fork bash scripts/setup_5090.sh
============================================================================
GUIDE

  if [ "${JEVIATHAN_5090_NATIVE:-}" != "fork" ]; then
    echo ""
    echo "==> Native Windows detected; no action taken (see guide above)."
    exit 3
  fi

  # --- PATH B: attempt the native fork wheel install -------------------------
  echo "==> Attempting native Windows fork install (SystemPanic/vllm-windows)..."
  if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
  VENV_PY=".venv/Scripts/python.exe"

  pyver="$("$VENV_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [ "$pyver" != "3.12" ]; then
    echo "!! The fork wheels target CPython 3.12, but this .venv is $pyver."
    echo "   Rebuild the venv from official CPython 3.12 (python.org), e.g.:"
    echo "     rm -rf .venv && py -3.12 -m venv .venv"
    echo "   then re-run: JEVIATHAN_5090_NATIVE=fork bash scripts/setup_5090.sh"
    exit 4
  fi

  api="https://api.github.com/repos/SystemPanic/vllm-windows/releases/latest"
  tmp="$(mktemp)"
  if ! curl -fsSL "$api" -o "$tmp"; then
    echo "!! Could not query the fork's latest release (network?). Install manually:"
    echo "     https://github.com/SystemPanic/vllm-windows/releases/latest"
    rm -f "$tmp"; exit 5
  fi
  # Prefer a cp312 win_amd64 wheel; fall back to any .whl in the release.
  whl="$(grep -o '"browser_download_url": *"[^"]*cp312[^"]*_win_amd64\.whl"' "$tmp" | head -n1 | sed 's/.*: *"//; s/"$//')"
  if [ -z "${whl:-}" ]; then
    whl="$(grep -o '"browser_download_url": *"[^"]*\.whl"' "$tmp" | head -n1 | sed 's/.*: *"//; s/"$//')"
  fi
  rm -f "$tmp"
  if [ -z "${whl:-}" ]; then
    echo "!! No Windows wheel found in the latest release. Install manually:"
    echo "     https://github.com/SystemPanic/vllm-windows/releases/latest"
    exit 6
  fi

  whl_file="$(basename "$whl")"
  echo "==> Downloading fork wheel: $whl_file"
  curl -fL "$whl" -o "$whl_file"
  echo "==> Installing into .venv ..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -U "./$whl_file"
  # Jeviathan API-layer deps (not needed by vllm itself).
  "$VENV_PY" -m pip install -r requirements.txt || true
  echo "==> Fork wheel installed. Continuing to serve."
fi

# --- Linux / WSL2: official vLLM path ---------------------------------------
if [ ! -d .venv ]; then
  echo "==> Creating .venv (isolates vLLM from system/MSIX Python PATH quirks)"
  "$PY" -m venv .venv
fi
case "$UNAME_S" in
  MSYS*|MINGW*|CYGWIN*) VENV_PY=".venv/Scripts/python.exe" ;;
  *)                    VENV_PY=".venv/bin/python" ;;
esac

if [ "$PLATFORM" = "linux" ]; then
  echo "==> Installing/upgrading vLLM (needs >= 0.17 for Qwen3.8 hybrid attention)"
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -U "vllm>=0.17"
  "$VENV_PY" -m pip install -r requirements.txt

  # FAIL-FAST GATE: catch missing-native-module / driver-mismatch class errors in
  # seconds, not after nohup + a long poll of a process that can never come up.
  echo "==> Verifying vLLM imports cleanly (fail-fast gate) ..."
  if ! "$VENV_PY" -c "import vllm; print('    vLLM', vllm.__version__, 'OK')" ; then
    echo ""
    echo "!! vLLM installed but does not import. Common causes on this box:"
    echo "   * Native Windows wheel missing its CUDA binding -> use WSL2 (PATH A)"
    echo "     or the SystemPanic/vllm-windows fork (PATH B); see guide above."
    echo "   * Driver/CUDA mismatch -> check:  nvidia-smi"
    echo "   Full error was printed above. Do NOT poll a server that cannot start."
    exit 7
  fi
fi

# --- serve -------------------------------------------------------------------
echo "==> Serving $MODEL on :$PORT (alias $ALIAS)"
echo "    First run downloads weights from HuggingFace (~13-29 GB depending on variant)."
echo "    Logs: vllm.log   Stop later with: python scripts/manage.py stop vllm"

# Module form = PATH-proof. --kv-cache-dtype fp8 halves KV memory (Blackwell);
# set JEVIATHAN_5090_KV_DTYPE=auto to omit it (safer under WSL2's emulated FP8).
KV_FLAG=()
if [ "$KV_DTYPE" != "auto" ]; then KV_FLAG=(--kv-cache-dtype "$KV_DTYPE"); fi

nohup "$VENV_PY" -m vllm serve "$MODEL" \
  --served-model-name "$ALIAS" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  ${KV_FLAG[@]+"${KV_FLAG[@]}"} \
  --gpu-memory-utilization 0.92 \
  --port "$PORT" > vllm.log 2>&1 &
echo $! > vllm.pid

# --- wait for readiness ------------------------------------------------------
echo "==> Waiting for http://localhost:$PORT/v1/models (polling up to 30 min) ..."
for i in $(seq 1 180); do
  if curl -s --max-time 5 "http://localhost:$PORT/v1/models" | grep -q "$ALIAS"; then
    echo ""
    echo "==> vLLM is UP. Next steps:"
    echo "    python scripts/manage.py status"
    echo "    python scripts/smoke_logprob.py rtx5090     # verify logprob path (usage.input_tokens should be null)"
    echo "    python scripts/manage.py start api --profile rtx5090"
    exit 0
  fi
  if ! kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo ""
    echo "!! vLLM process died during startup — last log lines:"
    tail -n 30 vllm.log
    break
  fi
  sleep 10
done

echo ""
echo "==> Not ready yet (or still downloading weights). Tail the log and re-check later:"
echo "    tail -f vllm.log"
echo "    curl http://localhost:$PORT/v1/models"
exit 2

# --- Troubleshooting -----------------------------------------------------------
# * `vllm: command not found` after a successful pip install -> you are on the
#   Windows Store (MSIX) Python and its Scripts dir is not on PATH. This script
#   avoids that by using .venv + `python -m vllm serve`. If you installed into
#   system Python before, just run this script — it builds a clean .venv.
# * Native Windows crash `No module named 'vllm._C_stable_libtorch'` -> official
#   vLLM is Linux-only; use WSL2 (PATH A) or the fork (PATH B), see guide above.
# * NVFP4/FP8 load errors (driver/CUDA mismatch) -> switch to official FP8:
#     JEVIATHAN_5090_MODEL=Qwen/Qwen3.8-27B-FP8 JEVIATHAN_5090_MAX_LEN=16384 bash scripts/setup_5090.sh
#   (FP8 27B is ~27 GB of weights — tight on 32 GB, hence the shorter context.)
# * WSL2 + Blackwell: keep WSL2 >= 2.7 (`wsl --update`), remove Tailscale from
#   the distro, and consider JEVIATHAN_5090_KV_DTYPE=auto if FP8 looks slow.
