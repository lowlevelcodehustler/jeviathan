---
name: jeviathan-ops
display-name: Jeviathan Ops
description: Start, stop, restart and check status of the Jeviathan stack (model server + System One API) on the laptop or 5090 box; run calibration fits and logprob smoke tests.
---

Operate **Jeviathan** — a System One compatibility and retrofit layer for open-weight LLMs
(TypeSafe-compatible decision API: state + typed questions → calibrated probabilistic answers).
Use when the user asks to start/stop/restart/check Jeviathan, run or refit calibration, smoke-test
the logprob path, or debug why an answer is slow/falling back.

## Facts

- **Two machines** (detect which you're on — use whichever repo root exists):
  - Laptop (RTX 4050, 6GB, power-capped ~27W): `E:/jeviathan` — dev/test tier.
  - 5090 box (RTX 5090, 32GB): `C:/jeviathan` — prod tier (Windows, Git Bash).
- **Services & ports** (fixed by convention):

  | service | what | port | log / pid file |
  |---|---|---|---|
  | `shim` | `scripts/transformers_server.py` — NF4 Llama-3.1-8B (laptop) | 8200 | model.log / model.pid |
  | `api` | `uvicorn jeviathan.main:app` — the System One API | 8100 | api.log / api.pid |
  | `vllm` | `python -m vllm serve Qwen3.8-27B` (5090 box) | 8001 | vllm.log / vllm.pid |

- **Profiles** (`profiles/*.yaml`, selected via `JEVIATHAN_PROFILE`):
  `laptop-4050` (one_shot), `laptop-4050-logprob` (logprob, laptop fit), `rtx5090` (logprob, prod).
- **Python**: always use the repo venv — `E:/jeviathan/.venv/Scripts/python.exe` (laptop) /
  `C:/jeviathan/.venv/Scripts/python.exe` (5090 box). On the 5090 box NEVER rely on system
  Python PATH: it's Windows Store (MSIX) Python whose pip scripts dir is not on PATH.
- **Model weights** (laptop shim): `E:\bfc-today-test-weights\model_run` (Llama-3.1-8B-Instruct, BF16 → served NF4).

## Commands (Git Bash, from the repo root)

All via the manager — it handles Windows wrapper PIDs, port-owner fallbacks and health waits:

```bash
PY=.venv/Scripts/python.exe   # C:/jeviathan on the 5090 box; E:/jeviathan here

$PY scripts/manage.py status                      # health of shim/api/vllm (+ --strict for exit code)
$PY scripts/manage.py start all --profile laptop-4050        # laptop stack (shim + api)
$PY scripts/manage.py start vllm api --profile rtx5090       # 5090 box stack
$PY scripts/manage.py restart api --profile laptop-4050
$PY scripts/manage.py stop shim [--force]          # --force kills port owner without usable pid file
$PY scripts/manage.py logs shim -n 40              # also: logs api / logs vllm
$PY scripts/manage.py fit --data evals/trinigard_eval.jsonl \
    --out calibration/calib-laptop-demo.json --profile laptop-4050-logprob [--background]
$PY scripts/manage.py smoke rtx5090                # end-to-end logprob path check
```

Raw fallbacks (when manage.py itself is suspect):

```bash
curl -s http://localhost:8100/health              # api: {"status":"ok","profile":...,"model":...}
curl -s http://localhost:8200/health              # shim: model_loaded must be true
curl -s http://localhost:8001/v1/models           # vllm: alias jeviathan-qwen3.8-27b listed
tail -n 40 model.log                               # or api.log / vllm.log
```

## Typical flows

- **Laptop, full stack**: `start all --profile laptop-4050` → wait for shim (cold load 4–6 min; warm ~15–30 s) → `status`.
- **5090 box, first run**: `bash scripts/setup_5090.sh` (builds .venv, installs vLLM, downloads weights, serves :8001) → `smoke rtx5090` → `start api --profile rtx5090`.
- **Calibration fit** (the RLCD-lite loop): laptop ~25 min for 12 rows with logprob; use `--background` and poll `calib_fit.log`. Refit without model calls: `fit --from-raw calibration/raw-laptop-logprob.jsonl ...`. Crash recovery mid-fit: re-run with `--resume`.
- **After any code change**: `restart api` (shim/vllm only need restart if their own scripts changed).

## Expected behaviour (NOT bugs — do not "fix" them)

- **Laptop is slow by design** (~27 W power cap): one_shot pass ~1–5 min; logprob row ~90 s. The 5090 box does the same work in seconds.
- **Logprob fallback**: with `strategy: logprob`, if the server can't do assistant-prefill+logprobs the engine silently falls back to one_shot. Detect via `smoke` output: `usage.input_tokens` present ⇒ fallback ran; null ⇒ real logprob path.
- **Llama eot loops**: trailing `<|eot_id>N</eot_id>` garbage after JSON is normal on the laptop model; the parse ladder handles it.
- **Wrapper PIDs (Windows)**: manage.py may print `pid X, wrapper was Y` — it adopts the real port owner automatically. Stale pid files self-heal on next `status`.
- **Services dying after laptop sleep**: common. Just `start all` again; nothing is lost (stateless API).

## Safety

- **VRAM budget (laptop 6GB)**: NF4 8B ≈ 5 GB. Never run two model servers at once on the laptop, and keep `max_tokens ≤ 768` there.
- **Bind to localhost**: shim binds 0.0.0.0 by default — fine on a private LAN, but tell the user before exposing :8100/:8200 beyond it (the API spends GPU time for anyone who can reach it).
- **Long fits**: a laptop fit is ~25 min of near-full GPU use; prefer `--background` and warn the user before starting one in the foreground.
- **Weights are not code**: Qwen3.8 / Llama weights have their own licenses (Apache-2.0 / Meta community license). Jeviathan's Apache-2.0 license covers only this repo.
- **5090 box first run** downloads ~13–29 GB from HuggingFace — check disk space and network before starting; `vllm.log` shows progress.
