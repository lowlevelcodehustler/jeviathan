# Jeviathan ⚓

**A System One compatibility and retrofit layer for open-weight LLMs.** Jeviathan speaks TypeSafe's [System One contract](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — the one behind their Jev model — so you can run typed, calibrated probabilistic decisions today on your own hardware: **Qwen3.8-27B via vLLM on the RTX 5090**, Llama-3.1-8B (or Qwen small) via Ollama/llama.cpp on the laptop.

> Positioning: "compatibility" is at the contract level — same request/response shape, same confidence formula — not a claim of behavioural parity with Jev itself. When real Jev access lands, flip one config line and A/B the two behind an identical API.
>
> Research background: the Jev deep-research report lives in the TriniGard repo (`docs/JEV_TYPESAFE_DEEP_RESEARCH.md`).

## Documentation

- **[User guide](docs/user-guide.md)** — concepts, install on both machines, API reference, day-to-day ops, calibration workflow, troubleshooting.
- **[Developer guide](docs/developer-guide.md)** — architecture deep-dive, extending backends and question types, testing, contributing.

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
| **Laptop 4050 (6GB)** — dev/test | Llama-3.1-8B-Instruct Q4 (~4.9GB) via Ollama, or any local HF weights dir (`--model-dir`, $JEVIATHAN_MODEL_DIR, or `.jeviathan_model_dir`) served by `scripts/transformers_server.py` in NF4 | Zero-download option; 6GB VRAM caps you at ~8B-class |
| Future | Qwen3.8-Max class (2.4T / 95B active, open weights) | Multi-GPU or cloud; revisit when it lands |

## Sampling strategies (v1.1)

`profiles/*.yaml → sampling.strategy` picks how distributions are produced:

| Strategy | How it works | When to use |
|---|---|---|
| `one_shot` (default) | One completion returns a JSON object of self-reported probabilities; the parse + repair ladder normalizes them. | Any OpenAI-compatible server, long criteria text, small models that wobble on strict single-label output. |
| `logprob` | Per-option prefix scoring: prefill the assistant turn with each option label and read the mean logprob of that span (one forward pass per option); softmax across options → distribution. The open-weight analogue of Jev's parallel sampler — probabilities come from the model's own token scores, not self-report. | vLLM/SGLang/our shim; short option labels; when you want truer distributions. Falls back to `one_shot` automatically if the server can't do assistant-prefill + logprobs. |

Smoke-test a serving stack end-to-end (also your first move on the 5090 box):

```bash
python scripts/smoke_logprob.py rtx5090   # usage.input_tokens null => real logprob path ran
```

## Quickstart — RTX 5090 box

```bash
# 1. Serve the model (downloads ~13-29GB on first run)
bash scripts/setup_5090.sh          # vLLM on :8001, alias jeviathan-qwen3.8-27b

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
python scripts/transformers_server.py --model-dir <your-model-dir> --port 8200   # or set JEVIATHAN_MODEL_DIR
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
  "model": "jeviathan-qwen3.8-27b",
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

Crash recovery: each row's full response is appended to `calibration/raw-<profile>.jsonl`; re-run with `--resume` to skip rows already collected (the laptop GPU takes minutes per row), or refit instantly from the dump with `--from-raw <file>`.

## TriniGard integration (the meld, part two)

Jeviathan speaks TypeSafe's contract, so TriniGard gets a new provider adapter (`core/adapters/jeviathan.py`) that:
- calls `POST /v1/systemone` with per-use-case questions compiled from the trust config,
- maps `confidence` → existing threshold logic (0.65–0.98) and `fallback_behavior`,
- logs every decision through the existing WAL audit trail (HMAC-signed, SOC2 exportable).

Cascade pattern: **Jeviathan front door** (classify/route at ~$0 cost, <1s locally) → ordinary code for deterministic cases → full multi-source verification engine for flagged claims. When real Jev access arrives: same adapter, `JEVIATHAN_BASE_URL` pointed at the TypeSafe API — instant A/B.

## Proven live (2026-09-25, v1.1 logprob on laptop tier)

`strategy: logprob` against the NF4 Llama shim — 12 scoring calls per row in ~90 s (vs ~5 min for one_shot), and smoother distributions than self-reported JSON:

```json
"claim_supported": {"type":"noul","noul":0.468791},
"risk_level":      {"type":"score","score":0.6139,"confidence":0.416296,
                    "probabilities":{"low":0.610864,"medium":0.164412,"high":0.224724}},
"vertical":        {"type":"choice","choice":"healthcare","confidence":0.925165,
                    "probabilities":{"healthcare":0.935855,"aviation":0.021986,...}}
```

Demo calibration fit (`profiles/laptop-4050-logprob.yaml`, 12 TriniGard-vertical claims):

| type | n | Platt a, b | ECE raw → calibrated |
|---|---|---|---|
| choice | 12 | +2.20, +0.58 | 0.074 → **0.004** |
| score | 12 | +2.08, −1.79 | 0.247 → **0.079** |
| noul | 12 | +3.24, −0.99 | 0.199 → **0.019** |

Artifact: `calibration/calib-laptop-demo.json` (refit anytime with `--from-raw calibration/raw-laptop-logprob.jsonl`). The real artifact is the same fit on the 5090 box's Qwen3.8.

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

- **v1.1 — logprob scoring (done 2026-09-25):** per-option prefix probabilities with one_shot fallback; shim prefill support; laptop demo fit above. Verify live on the 5090 box with `scripts/smoke_logprob.py`.
- **v1.2 — TriniGard adapter + cascade (done, in the TriniGard repo):** `core/adapters/jeviathan.py`, weight 0.95 in ConfidenceScorer, judgment-aware discrepancy counting.
- **v2 — Decision feedback loop (nightly):** capture -> review -> QLoRA -> promote, all local (`scripts/feedback.py`, user guide §8); the shim serves the resulting adapter with zero extra VRAM and rollback is one pointer file. Refit calibration after promoting. This is where a compatibility layer becomes close enough to matter.
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
profiles/             # laptop-4050.yaml, laptop-4050-logprob.yaml, rtx5090.yaml
scripts/              # setup_5090.sh, setup_laptop.ps1, transformers_server.py, smoke_logprob.py, manage.py
docs/                 # user-guide.md, developer-guide.md
skills/jeviathan-ops/ # Bionic skill for operating the stack (install: see dev guide)
evals/sample_eval.jsonl
tests/                # GPU-free test suite (mock backend)
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v      # no GPU needed; mock backend
```

## Licensing & support

- **Jeviathan is [Apache License 2.0](LICENSE)** — permissive, with an explicit patent grant (which matters in AI infrastructure) and clean contribution terms. Built on open foundations (vLLM, transformers, Qwen/Llama weights); built to give back.
- Model weights are licensed separately: Qwen3.8 (Apache-2.0), Llama 3.1 (Meta community license). The Apache-2.0 license covers this repo's code only.
- **TriniGard** (the enterprise verification harness that consumes Jeviathan) stays closed-source in its own repo; the adapter (`core/adapters/jeviathan.py`) is the seam — permissive upstream, proprietary downstream.

### Tip jar ☕

Jeviathan is free and open. If it saves you time or money, consider tipping Trinitris:

> **Stripe:** [donate.stripe.com/cNi00jc4ydVy1Vsg07cs800](https://donate.stripe.com/cNi00jc4ydVy1Vsg07cs800)

## Example: ticket triage grounded by the Brave Search API

`examples/brave_search_triage.py` shows the pattern for a decision that *verifies itself against the live web* when confidence is low. Triage runs locally (~$0, <1s); only the grounding step calls out to the [Brave Search API](https://brave.com/search/api) ($5 per 1,000 requests; $5 free credits monthly).

```bash
export JEVIATHAN_BASE_URL=http://localhost:8100     # or your profile's port
export BRAVE_API_KEY=<key from brave.com/search/api>

python examples/brave_search_triage.py "My card was charged twice for order #4512. I want a refund."
```

What happens:

1. `POST /v1/systemone` -> department (choice), urgency (noul), frustration (score) with calibrated confidence.
2. If routing confidence < 0.7 (or `--always-ground`): `GET https://api.search.brave.com/res/v1/web/search?q=...&count=5` with the `X-Subscription-Token` header.
3. Report: decision + cited evidence, ready to feed an agent or a human. Deterministic on purpose - no second model call, so it runs anywhere and costs only search calls.

The raw pieces, for your own wiring:

```bash
# 1) triage (Jeviathan)
curl http://localhost:8100/v1/systemone -H "Content-Type: application/json" \
     -d '{"state": "Customer ticket: \"My card was charged twice. I want a refund.\"", "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Charges, refunds", "returns": "Exchanges, damaged items"}}}'

# 2) grounding (Brave Search API)
curl "https://api.search.brave.com/res/v1/web/search?q=refund+double+charge&count=5" \
     -H "X-Subscription-Token: $BRAVE_API_KEY"
```

Upgrade path: swap the web endpoint for the [LLM Context endpoint](https://api-dashboard.search.brave.com/documentation/services/llm-context) when you want results pre-packaged for model consumption instead of human-readable snippets.
