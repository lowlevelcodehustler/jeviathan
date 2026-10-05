# Jeviathan — User Guide

**A System One compatibility and retrofit layer for open-weight LLMs.** Jeviathan speaks TypeSafe's [System One contract](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — the one behind their Jev model — so you can run typed, calibrated probabilistic decisions on your own hardware: Qwen3.8-27B via vLLM on an RTX 5090, or Llama-3.1-8B via Ollama / a local weights shim on a laptop.

> "Compatibility" is at the contract level — same request/response shape, same confidence formula — not a claim of behavioural parity with Jev itself. When real Jev access lands, flip one config line and A/B the two behind an identical API.

---

## 1. Concepts

### The System One contract

One HTTP call in, typed probabilistic answers out:

```
POST /v1/systemone   {state, questions}   ->   {model, answers, usage}
```

- **`state`** — the situation being judged: a string, dict, or list (e.g. a customer ticket, an incident description, agent trace).
- **`questions`** — a map of question-id to typed question (see below).
- **`answers`** — one typed answer per question id, each carrying calibrated probabilities and a confidence score.

### Question types

| Type | Shape | Answer |
|---|---|---|
| `choice` | `{type: "choice", instructions, criteria: {option: description?}}` (1–255 options) | `{choice, confidence, probabilities}` — pick one option from the fixed set |
| `score` | `{type: "score", instructions, levels: [{name?, description?}]}` (≥2 levels, 0-based) | `{score, confidence, probabilities}` — rate on an ordered scale; probabilities keyed by level index `"0"`, `"1"`, … |
| `noul` | `{type: "noul", instructions}` | `{noul}` — a probability in [0, 1] that the statement is true |

### Profiles

A **profile** (`profiles/*.yaml`) binds a machine to its backend + sampling settings. Select with the `JEVIATHAN_PROFILE` env var (name or `.yaml` path); default is `laptop-4050`. Runtime overrides: `JEVIATHAN_BASE_URL`, `JEVIATHAN_MODEL`.

| Profile | Backend | Strategy | Use |
|---|---|---|---|
| `laptop-4050` | Ollama `llama3.1:8b` @ :11434 | `one_shot` | zero-download dev tier |
| `laptop-4050-logprob` | local weights shim @ :8200 (`llama3.1-8b-local`) | `logprob` | exercise the logprob path locally; demo calibration fit |
| `rtx5090` | vLLM Qwen3.8-27B-NVFP4 @ :8001 (alias `jeviathan-qwen3.8-27b`) | `logprob` + calibration on | production tier |

### Sampling strategies

- **`one_shot`** — a single JSON completion pass. Works everywhere, simplest.
- **`logprob`** — per-option prefix scoring: the model scores each option's prefix via assistant-prefill + logprobs in one forward pass (the open-weight analogue of Jev's parallel sampler). Truer distributions; falls back to `one_shot` automatically if the server can't do prefill+logprobs.

### Calibration

Optional Platt-scaling correction (`p' = a·p + b`) fitted from your own labeled eval set, applied per question type when an artifact is present and the profile enables it. See §6.

---

## 2. Hardware tiers & model choice

| Machine | Model | Why |
|---|---|---|
| **RTX 5090 (32GB)** — prod | `Inferact/Qwen3.8-27B-NVFP4` via vLLM ≥0.17 (~24.6 GiB, fits one Blackwell GPU). Fallback: official `Qwen/Qwen3.8-27B-FP8` (use `JEVIATHAN_5090_MAX_LEN=16384`) | best quality that fits one consumer GPU |
| **Laptop 6GB** — dev/test | Llama-3.1-8B-Instruct Q4 (~4.9GB) via Ollama, or any local HF weights dir served by `scripts/transformers_server.py` in NF4 | zero-download option; 6GB VRAM caps you at ~8B-class |
| Future | Qwen3.8-Max class (open weights) | multi-GPU or cloud; revisit when it lands |

---

## 3. Installation

### Guided installer (recommended)

```bash
# PowerShell (Windows) — probes GPU/VRAM/RAM/disk/Ollama/.venv, recommends a tier:
powershell -File scripts/setup.ps1            # interactive menu
powershell -File scripts/setup.ps1 -Tier C    # straight to vLLM tier
$env:JEVIATHAN_SETUP_TIER="C"; powershell -File scripts/setup.ps1   # CI

# bash (Windows Git Bash / Linux) — parallel twin:
bash scripts/setup.sh                          # probe + menu
bash scripts/setup.sh --tier C --persist       # flags; --persist writes .jeviathan_profile
```

Tiers: **A** = Ollama + Llama-3.1-8B (~5GB), **B** = native Torch NF4 shim over local weights, **C** = vLLM Qwen3.8-27B (≥24GB VRAM). The installer delegates to the per-tier scripts below and prints the exact profile + env to use.

### Per-tier scripts

```powershell
# Laptop tier — Option A (Ollama) and/or B (native Torch shim):
.\scripts\setup_laptop.ps1                    # interactive menu (A / B / both)
.\scripts\setup_laptop.ps1 -Option B          # shim only
.\scripts\setup_laptop.ps1 -Option B -ModelDir D:\weights -Port 8200
```

```bash
# 5090 tier — platform-aware: Linux/WSL2 installs official vLLM; native Windows
# fails fast with a copy-paste guide (official vLLM is Linux-only):
bash scripts/setup_5090.sh                    # WSL2 recommended (PATH A)
JEVIATHAN_5090_NATIVE=fork bash scripts/setup_5090.sh   # native fork wheel (PATH B)
```

Env overrides for the 5090 script: `JEVIATHAN_5090_MODEL` (default `Inferact/Qwen3.8-27B-NVFP4`), `JEVIATHAN_5090_PORT` (8001), `JEVIATHAN_5090_MAX_LEN` (32768; 16384 for FP8), `JEVIATHAN_5090_KV_DTYPE` (`fp8` default, or `auto` under WSL2).

### Local weights dir resolution (shim tier)

The NF4 shim resolves its weights directory in this order:

1. `--model-dir` flag / `-ModelDir` parameter
2. `$JEVIATHAN_MODEL_DIR` environment variable
3. `.jeviathan_model_dir` file at repo root (gitignored, one line)

Create the local pointer once and everything else just works:

```bash
echo 'E:\path\to\your\hf-weights' > .jeviathan_model_dir   # Windows
# or on Linux:  printf '%s\n' /path/to/your/hf-weights > .jeviathan_model_dir
```

---

## 4. Day-to-day operations (`scripts/manage.py`)

The manager is the daemon layer for both machines (stdlib-only, cross-platform). Services and ports are fixed by convention:

| Service | Port | PID / log files | What it is |
|---|---|---|---|
| `shim` | 8200 | `model.pid` / `model.log` | NF4 local-weights shim (laptop tier) |
| `api` | 8100 | `api.pid` / `api.log` | Jeviathan System One API (`jeviathan.main:app`) |
| `vllm` | 8001 | `vllm.pid` / `vllm.log` | vLLM model server (5090 tier) |

```bash
PY=.venv/Scripts/python.exe        # repo .venv (Windows); .venv/bin/python on Linux

$PY scripts/manage.py status                       # health of all three (+ --strict for exit code)
$PY scripts/manage.py start all --profile laptop-4050          # laptop stack (shim + api)
$PY scripts/manage.py start vllm api --profile rtx5090         # 5090 box stack
$PY scripts/manage.py restart api --profile laptop-4050
$PY scripts/manage.py stop shim [--force]          # --force kills the port owner without a usable pid file
$PY scripts/manage.py logs shim -n 40              # also: logs api / logs vllm
```

Notes:

- `start` waits and polls for health (model loads are slow on laptop hardware: shim ~40 s–6 min; vLLM minutes to tens of minutes on first run — weight download). Use `--wait 0` to detach immediately.
- `start api` auto-points `JEVIATHAN_BASE_URL`/`JEVIATHAN_MODEL` at whichever local model server is up (shim :8200 or vLLM :8001) unless you set them yourself.
- Windows wrapper-PID self-healing: if the pid file holds a launcher PID while the real server runs as its child, `status` adopts the port owner automatically.
- `--strict` exits 1 if **any** of the three services is down — on a laptop-only machine vLLM will be down, so expect rc=1 there; it's meant for CI on the full stack.

### Quick health checks (no manager)

```bash
curl -s http://localhost:8100/health      # api: {"status":"ok","profile":...,"model":...}
curl -s http://localhost:8200/health      # shim: model_loaded must be true
curl -s http://localhost:8001/v1/models   # vllm: alias jeviathan-qwen3.8-27b listed
```

---

## 5. API reference

Base URL: `http://localhost:8100` (or your profile's port).

### `GET /health`

```json
{"status": "ok", "profile": "laptop-4050", "model": "llama3.1-8b-local", "calibration_active": false}
```

### `GET /v1/models`

OpenAI-shaped: `data[0].id` is the profile's backend model name.

### `POST /v1/systemone`

Request:

```json
{
  "state": {"incident": "The payment service returned HTTP 500 errors for the last 10 minutes."},
  "questions": {
    "q1": {"type": "choice", "instructions": "Is this an infrastructure failure?",
           "criteria": {"yes": null, "no": null}},
    "q2": {"type": "noul", "instructions": "Did the customer explicitly request a refund?"},
    "q3": {"type": "score", "instructions": "How urgent is this ticket?",
           "levels": [{"name": "low"}, {"name": "medium"}, {"name": "high"}]}
  }
}
```

Response:

```json
{
  "model": "llama3.1-8b-local",
  "answers": {
    "q1": {"type": "choice", "choice": "yes", "confidence": 0.6,
           "probabilities": {"yes": 0.8, "no": 0.2}},
    "q2": {"type": "noul", "noul": 0.71},
    "q3": {"type": "score", "score": 2.0, "confidence": 0.55,
           "probabilities": {"0": 0.1, "1": 0.35, "2": 0.55}}
  },
  "usage": {"input_tokens": 413, "output_tokens": 28}
}
```

Errors:

| HTTP | Meaning |
|---|---|
| `422` | Request violates profile limits (`max_questions`, `max_options_per_choice`, `max_state_chars`) — see the detail message for which limit. |
| `502` | Model output invalid after corrective retries (`CompileError`), or backend unreachable/failed (`BackendError`). |

Limits are per-profile (see `profiles/*.yaml → limits:`). Malformed model JSON triggers up to 2 corrective retries before a 502 is returned.

---

## 6. Calibration workflow

Calibration fits Platt parameters `(a, b)` from your own labeled eval set and applies them per question type — correcting systematic over/under-confidence without retraining anything.

1. **Prepare an eval set** (JSONL; default `evals/trinigard_eval.jsonl`). Each row:
   ```json
   {"state": "...", "questions": {...}, "labels": {"q1": "yes", "q2": true, "q3": 2}}
   ```
   Labels are the gold option name (`choice`), boolean (`noul`), or level index (`score`).

2. **Fit** (runs each row through the engine with calibration disabled, pools by question type when a single qid has fewer than 20 rows, prints ECE before/after):
   ```bash
   $PY scripts/manage.py fit --data evals/trinigard_eval.jsonl \
       --out calibration/calib-laptop-demo.json --profile laptop-4050-logprob [--background]
   ```
   `--from-raw` refits from a raw response dump instead of calling the model; `--resume` skips rows already in the dump.

3. **Enable** it in your profile:
   ```yaml
   calibration:
     enabled: true
     path: calibration/calib.json
   ```
   `/health` then reports `"calibration_active": true`. A missing/corrupt artifact degrades gracefully (uncalibrated, no crash).

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `vllm: command not found` after a successful install on Windows | You're on the Windows Store (MSIX) Python — its pip scripts dir isn't on PATH, and the PyPI vLLM wheel lacks the native CUDA binding. Use WSL2 (official vLLM) or the SystemPanic/vllm-windows fork; always launch via `python -m vllm serve` inside a `.venv`. |
| `ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'` | Same root cause — official vLLM is Linux-only. See `bash scripts/setup_5090.sh` for the two working paths (WSL2 recommended). |
| Shim takes minutes to start on a laptop | Normal: NF4 load of an 8B model on a power-capped GPU is ~40 s–6 min. `manage.py start` polls; use `--wait 0` to detach and check `logs shim`. |
| `status` says "port busy (foreign pid …)" | Another process owns the port. Identify it (`netstat -ano \| findstr :8200`) or stop it, then retry. `stop --force` kills the port owner when no usable pid file exists. |
| `No model dir for the NF4 shim` | Set one of: `--model-dir`, `$JEVIATHAN_MODEL_DIR`, or write `.jeviathan_model_dir` (one line) at repo root. |
| `[transformers] … generation flags … temperature` warnings in `model.log` | Harmless — the shim passes sampling params through; transformers ignores what it doesn't use for greedy decoding. |
| API returns 502 `BackendError` | The model server (shim/vLLM) is down or timed out. Check its health endpoint and log first. |
| Ollama option: model not found | `ollama pull llama3.1:8b` (~4.9 GB), then re-run the setup script with `-Option A`. |

---

## 8. Learning from your decisions (feedback loop, nightly)

Jeviathan can learn from its own decision history: every System One pass is logged
(profile opt-in), you review outcomes, and verified corrections are used to
QLoRA-fine-tune the local model into new weights that the shim serves.

**Why adapters:** on a 6 GB card an 8B model trains in NF4 with LoRA (~0.5% of
params). The result is a small adapter file, not a second copy of the weights:
zero extra VRAM at serving time, and rollback is deleting one pointer file.

### Workflow

```bash
# 1. Capture — profiles laptop-4050*.yaml have feedback.enabled: true; every pass
#    appends to data/decisions.jsonl (gitignored). Manual capture too:
python scripts/feedback.py log --state "..." --questions '{...}' --answers '{...}'

# 2. Review — attach verdicts and corrections:
python scripts/feedback.py list --pending
python scripts/feedback.py review <id> --verdict incorrect \
    --correction-json '{"department": "returns"}'

# 3. Build + train (needs GPU; pip install -r requirements-train.txt):
python scripts/feedback.py build-dataset          # -> data/sft.jsonl
python scripts/feedback.py train                  # QLoRA on the shim's base model

# 4. Promote + restart:
python scripts/feedback.py promote <run-id>
python scripts/manage.py restart shim

# Rollback anytime:
python scripts/feedback.py promote --revert && python scripts/manage.py restart shim
```

### Rules of the loop

- A record only trains if **every** question is verified (verdict `correct`, or a
  correction covering it). Partially-verified records are excluded and counted —
  Jeviathan never learns an answer nobody checked.
- Corrections become one-hot targets; explicit `{"probabilities": {...}}` overrides
  are honoured as-is. Residual miscalibration is absorbed by the calibration layer:
  after promoting, refit with `manage.py fit --data <export-eval output>`.
- `feedback.py export-eval` turns verified records into calibration-fit rows.
- Ollama/vLLM tiers keep capturing decisions; `promote` applies to the shim tier
  (the only one that loads HF weights directly). GGUF export for Ollama is future work.

### VRAM notes (6 GB laptop)

NF4 base (~5 GB) + LoRA fits, but long sequences spill into system RAM via Windows
unified memory — keep `--max-len` ≤ 1024 and expect slow steps on a power-capped
GPU. Runs write `finetunes/<base>/<run-id>/train_report.json` (hyperparams,
per-epoch losses, VRAM peak).

---

## 9. Licensing & support

- **Jeviathan is [Apache License 2.0](../LICENSE)** — permissive, with an explicit patent grant and clean contribution terms. Model weights are licensed separately (Qwen3.8: Apache-2.0; Llama 3.1: Meta community license).
- If Jeviathan saves you time or money, consider tipping Trinitris: **Stripe:** <https://donate.stripe.com/cNi00jc4ydVy1Vsg07cs800>
