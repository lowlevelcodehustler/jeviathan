# Jeviathan ⚓

**A System One decision API over open-weight models.** A knock-off of TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) that you can run today on your own hardware — while the real thing sits in a waitlist.

Jeviathan implements Jev's **contract** (state + typed questions → typed probabilistic decisions with calibrated confidence, one parallel pass) over any OpenAI-compatible local model server: **Qwen3.8-27B via vLLM on the RTX 5090**, Llama-3.1-8B (or Qwen small) via Ollama/llama.cpp on the laptop.

> Research background: see [TriniGard's Jev deep-research report](../../OneDrive/Documents/Trinitris/trinigard-neo/docs/JEV_TYPESAFE_DEEP_RESEARCH.md).

---

## Why "meld" instead of just fine-tune?

Jev's three pillars, and how we approximate each with open weights:

| Jev pillar | Jeviathan approach |
|---|---|
| **Typed questions** (Choice / Score / Noul) over a shared state | Same primitives, same request/response shape → drop-in for `typesafe_sdk`-style clients and TriniGard adapters |
| **Parallel sampler** (all questions in one pass) | One prompt ingests the state once; every question is answered in the same completion. Adding questions barely changes latency — same economics as Jev |
| **RLCD → calibrated confidence** | "RLCD-lite": raw model probabilities + post-hoc **Platt calibration** fitted on your own labeled eval data (`jeviathan/calibration/`), then TypeSafe's exact confidence formula `(n·peak−1)/(n−1)` |

The payoff: when real Jev access lands, you flip one config line and A/B the two behind an identical API.

## Architecture

```
                 ┌────────────────────────────────────────────┐
  POST /v1/      │                Jeviathan (FastAPI)         │
  systemone      │                                            │
 {state,         │  limits → prompt compiler → parse+norm     │
  questions} ───►│              ↓                             │
                 │        DecisionBackend (OpenAI-compat)     │
                 │              ↓                             │
                 │   calibrate (Platt artifact) → confidence  │
                 │              ↓                             │
 {answers:       │  Choice / Score / Noul + probabilities     │
  typed+probs} ◄─┤                                            │
                 └──────────────┬─────────────────────────────┘
                                │ /chat/completions
                ┌───────────────┴────────────────┐
                ▼                                ▼
   RTX 5090 (32GB)                     Laptop 4050 (6GB)
   vLLM: Qwen3.8-27B-NVFP4             Ollama: llama3.1:8b
   (~24.6 GiB, prod tier)              or transformers_server.py
                                       (local Llama-3.1-8B 4-bit, dev tier)
```

## Model selection

| Machine | Model | Why |
|---|---|---|
| **RTX 5090 (32GB)** — prod | `Inferact/Qwen3.8-27B-NVFP4` via vLLM ≥0.17 | "Fits one Blackwell GPU in every precision: NVFP4 in 24.6 GiB" ([vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)). Fallback: official `Qwen/Qwen3.8-27B-FP8` |
| **Laptop 4050 (6GB)** — dev/test | Llama-3.1-8B-Instruct Q4 (~4.9GB) via Ollama, or your existing local weights (`E:\bfc-today-test-weights\model_run`) via `scripts/transformers_server.py` in NF4 | Zero-download option already on disk; 6GB VRAM caps you at ~8B-class |
| Future | Qwen3.8-Max class (2.4T / 95B active, open weights) | Multi-GPU or cloud; revisit when it lands |

## Quickstart — RTX 5090 box

```bash
# 1. Serve the model (downloads ~13-29GB on first run)
bash scripts/setup_5090.sh          # vLLM on :8001, alias jevitan-qwen3.8-27b

# 2. In another terminal: run Jeviathan against it
pip install -r requirements.txt
JEVIATHAN_PROFILE=rtx5090 uvicorn jeviathan.main:app --port 8100
```

## Quickstart — Laptop (this machine)

```powershell
# Option A: Ollama + Llama-3.1-8B
powershell -File scripts/setup_laptop.ps1     # pulls llama3.1:8b, starts ollama serve

# Option B: your existing local weights (no download), 4-bit ~5GB VRAM
pip install torch transformers bitsandbytes fastapi uvicorn
python scripts/transformers_server.py --model-dir E:\bfc-today-test-weights\model_run --port 8200
$env:JEVIATHAN_BASE_URL="http://localhost:8200/v1"; $env:JEVIATHAN_MODEL="llama3.1-8b-local"

# Run Jeviathan (default profile = laptop-4050)
pip install -r requirements.txt
uvicorn jeviathan.main:app --port 8100
```

## Try it

```bash
curl http://localhost:8100/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Customer ticket: \"My card was charged twice for order #4512. I want a refund.\"",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Charges, invoices, payment problems",
                                "returns": "Exchanges, wrong or damaged items"}},
    "urgency":    {"type": "noul",  "instructions": "Is this ticket urgent?"},
    "frustration":{"type": "score", "instructions": "How frustrated is the customer?",
                   "levels": [{"name":"calm"},{"name":"frustrated"},{"name":"very_frustrated"}]}
  }
}'
```

Response (TypeSafe-shaped):

```json
{
  "model": "jevitan-qwen3.8-27b",
  "answers": {
    "department":  {"type": "choice", "choice": "billing", "confidence": 0.56,
                    "probabilities": {"billing": 0.78, "returns": 0.22}},
    "urgency":     {"type": "noul", "noul": 0.93},
    "frustration": {"type": "score", "score": 1.4, "confidence": 0.61,
                    "probabilities": {"calm": 0.1, "frustrated": 0.55, "very_frustrated": 0.35}}
  },
  "usage": {"input_tokens": 214, "output_tokens": 96}
}
```

Python:

```python
import httpx
resp = httpx.post("http://localhost:8100/v1/systemone", json={...}, timeout=300)
a = resp.json()["answers"]["department"]
if a["confidence"] > 0.85: route_to(a["choice"])      # act
elif a["confidence"] > 0.5: flag_for_review()          # caution
else: escalate_to_human()                              # don't act
```

## Calibration (the RLCD-lite loop)

1. Build a labeled eval set — JSONL rows `{state, questions, labels}` where the label is the correct option name / level index / bool. Start from `evals/sample_eval.jsonl`; **best source: TriniGard's pilot traffic with human-verified outcomes.**
2. Fit against your running model server:

```bash
JEVIATHAN_PROFILE=rtx5090 python -m jeviathan.calibration.cli fit \
    --data evals/sample_eval.jsonl --out calibration/calib.json
# prints ECE raw -> calibrated per question type; writes the artifact
```

3. The `rtx5090` profile already points at `calibration/calib.json` with `enabled: true` — restart the API and confidence is now outcome-calibrated, which is what makes TriniGard-style thresholds meaningful.

## TriniGard integration (the meld, part two)

Jeviathan speaks TypeSafe's contract, so TriniGard gets a new provider adapter (`core/adapters/jevitan.py`) that:
- calls `POST /v1/systemone` with per-use-case questions compiled from the trust config,
- maps `confidence` → existing threshold logic (0.65–0.98) and `fallback_behavior`,
- logs every decision through the existing WAL audit trail (HMAC-signed, SOC2 exportable).

Cascade pattern: **Jeviathan front door** (classify/route at ~$0 cost, <1s locally) → ordinary code for deterministic cases → full multi-source verification engine for flagged claims. When real Jev access arrives: same adapter, `JEVIATHAN_BASE_URL` pointed at the TypeSafe API — instant A/B.

## Proven live (2026-09-23, laptop tier)

Llama-3.1-8B-Instruct (NF4 on RTX 4050 Laptop) via `scripts/transformers_server.py`, full pipeline:

```json
POST /v1/systemone  →  HTTP 200 in ~5 min (power-limited laptop GPU, ~1-2 tok/s)
{
  "model": "llama3.1-8b-local",
  "answers": {
    "department":  {"type":"choice","choice":"billing","confidence":0.6,
                    "probabilities":{"billing":0.8,"returns":0.2}},
    "urgency":     {"type":"noul","noul":1.0},
    "frustration": {"type":"score","score":1.3,"confidence":0.55,
                    "probabilities":{"calm":0.0,"frustrated":0.7,"very_frustrated":0.3}}
  },
  "usage": {"input_tokens":568,"output_tokens":768}
}
```

All three answers semantically correct for a double-charge refund ticket; confidence values match TypeSafe's formula exactly.

**Laptop reality notes (learned the hard way):**
- This 4050 Laptop is power-capped (~27 W): expect ~1–2 tok/s with an 8B model. The 5090 box will be dramatically faster.
- Llama-3.1 quirks handled: trailing `<|eot_id>N</eot_id>` index loops (stripped in the shim + balanced-brace extraction), dropped closing braces (brace-completion ladder + answer hoisting), `null` probabilities (treated as 0 mass).
- Model load: ~40 s for NF4 fully on GPU; keep `max_tokens` ≤ 768 on this tier.

## Roadmap

- **v1.1 — logprob scoring:** replace self-reported JSON probabilities with per-option token-logprob softmax (vLLM `logprobs`) for a truer "parallel sampler" distribution; keep Strategy-A as fallback for long criteria text.
- **v1.2 — TriniGard adapter + cascade** in the TriniGard repo.
- **v2 — RLCD-lite SFT:** fine-tune Qwen3.8 (LoRA) on synthetic System One data generated from TriniGard's verified decision logs, then re-fit calibration. This is where "knock-off" becomes "close enough to matter."
- **Watch:** real Jev waitlist access → A/B harness; Qwen3.8-Max open weights (2.4T/95B) for a multi-GPU tier.

## Layout

```
jeviathan/            # the package
  api/systemone.py    # FastAPI app: /v1/systemone, /health, /v1/models
  compiler/           # state+questions -> one prompt; parse + renormalize
  backends/           # OpenAI-compatible backend (vLLM/Ollama/llama.cpp/shim)
  confidence/         # TypeSafe's exact confidence formula
  calibration/        # Platt fit/apply + CLI (RLCD-lite)
  engine/             # orchestration: validate -> compile -> call -> calibrate
profiles/             # laptop-4050.yaml, rtx5090.yaml
scripts/              # setup_5090.sh, setup_laptop.ps1, transformers_server.py
evals/sample_eval.jsonl
tests/                # GPU-free test suite (mock backend)
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v      # no GPU needed; mock backend
```
