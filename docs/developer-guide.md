# Jeviathan — Developer Guide

Architecture deep-dive, extension points, testing, and contribution conventions. If you're just running it, start with the [User guide](user-guide.md).

---

## 1. Repository layout

```
jeviathan/                  # the package (pure Python; deps: fastapi, uvicorn, httpx, pydantic, PyYAML, numpy)
  main.py                   # entrypoint: app = create_app(); uvicorn runner
  config.py                 # Profile dataclasses + load_profile() (JEVIATHAN_PROFILE / .yaml path)
  api/systemone.py          # FastAPI app factory: /health, /v1/models, POST /v1/systemone
  engine/systemone_engine.py# the decision pass: validate -> compile -> complete -> parse -> calibrate -> derive
  compiler/prompt_compiler.py  # request -> prompt; answer JSON parsing (CompileError), expected_options()
  backends/base.py          # DecisionBackend interface + BackendError
  backends/openai_backend.py   # OpenAI-compatible chat-completions client (httpx)
  confidence/derive.py      # argmax, distribution_confidence, score_expectation
  calibration/calibrator.py # Platt fit (a,b), ECE, load_eval_rows()
  calibration/cli.py        # `python -m jeviathan.calibration.cli fit ...`
  schemas/typesafe.py       # Pydantic models mirroring the TypeSafe System One contract
profiles/                   # laptop-4050.yaml, laptop-4050-logprob.yaml, rtx5090.yaml
scripts/                    # manage.py (daemon layer), setup*.ps1/.sh (installers), transformers_server.py (NF4 shim)
evals/                      # labeled eval JSONL for calibration fits
calibration/                # fit artifacts (*.json); raw dumps (raw-*.jsonl, gitignored)
tests/                      # pytest suite (mock backend; no GPU needed)
examples/                   # end-user example scripts (e.g. Brave Search grounding triage)
```

## 2. Request flow deep-dive

`SystemOneEngine.system_one(req)` is the whole decision pass:

1. **Validate limits** — `state` size, question count, options-per-choice against `profile.limits`; violations raise `EngineValidationError` (→ HTTP 422).
2. **Compile prompt** — `compile_request()` renders state + questions into a single scoring prompt (`SCORING_SYSTEM_PROMPT`); the logprob strategy additionally compiles per-option prefix messages via `compile_scoring_user_message()`.
3. **Backend completion** — `OpenAICompatibleBackend` (or your injected backend). Malformed model JSON triggers up to `MAX_CORRECTIVE_RETRIES = 2` corrective retries, mirroring TypeSafe's adapter behaviour; still bad → `CompileError` (→ HTTP 502).
4. **Parse + normalize** — `parse_answers()` extracts per-question distributions and normalizes them to sum to 1.
5. **Calibrate** — if the profile enables calibration and an artifact loads, apply Platt `(a,b)` per question type; otherwise pass through.
6. **Derive typed answers** — `confidence/derive.py`: `argmax` for choice/noul, `score_expectation` (probability-weighted level index) for score, `distribution_confidence` for the confidence field. Output is TypeSafe-shaped (`schemas/typesafe.py`).

### Sampling strategies

- **`one_shot`**: one completion returns all answers as JSON.
- **`logprob`**: per-option prefix scoring — each option's label is prefilled as an assistant message and scored via `logprobs` in a single forward pass (the shim implements this: last message with role `assistant` = prefill; `"logprobs": true` → `choices[0].logprobs.content` carries per-token `{token, logprob, bytes, top_logprobs}`). The engine falls back to `one_shot` automatically when the server can't do prefill+logprobs. Scoring calls cap generation at 32 tokens so a stray large `max_tokens` can't burn GPU time on tokens nobody reads.

## 3. Profiles & configuration

`config.py` loads `profiles/<name>.yaml` (or any `.yaml` path) into dataclasses; env overrides apply after load:

```yaml
backend:    # type, base_url, model, api_key?, timeout_s=120, max_retries=3
sampling:   # temperature=0.0, top_p=1.0, max_tokens=2048, strategy: one_shot|logprob
calibration:# enabled=false, path=null
limits:     # max_questions=64, max_options_per_choice=255, max_state_chars=32000
```

Selection order: explicit `load_profile(name)` → `JEVIATHAN_PROFILE` env → default `laptop-4050`. Runtime overrides: `JEVIATHAN_BASE_URL`, `JEVIATHAN_MODEL`. The setup scripts persist a tier pointer in `.jeviathan_profile` (gitignored) for convenience.

## 4. Extension points

### Adding a backend

Implement the `DecisionBackend` interface (`backends/base.py`) — completion + close, raising `BackendError` on transport failure — then inject it:

```python
from jeviathan.config import load_profile
from jeviathan.engine.systemone_engine import SystemOneEngine

engine = SystemOneEngine(load_profile("rtx5090"), backend=MyBackend(...))
```

The OpenAI-compatible client is the reference implementation; any server speaking `chat/completions` works out of the box (Ollama, vLLM, the NF4 shim). For the logprob strategy your backend must support assistant-prefill + per-token logprobs as described in §2.

### Adding a question type

1. **Schema** (`schemas/typesafe.py`): add `MyQuestion(BaseModel)` with `type: Literal["my_type"]`, and extend the discriminated `Question` union; add the matching `Answer` variant to the `Answer` union.
2. **Engine**: handle the new type in `_validate_limits()` (if it has its own limits) and in answer derivation (argmax / expectation / custom).
3. **Compiler** (`compiler/prompt_compiler.py`): teach `compile_request()` how to render it, and `expected_options()` what options a valid answer may take; `parse_answers()` must accept the new shape.
4. **Calibration** (`calibration/cli.py::_row_pairs`): define how a gold label maps to `(p_correct, correct)` pairs for your type (choice: probability of the gold option; score: proximity-weighted expectation; noul: p vs 1-p).
5. **Tests**: add cases in `tests/test_engine.py` + `tests/test_compiler.py` with the mock backend.

### Adding a sampling strategy

Strategies are selected by `profile.sampling.strategy`. Add your branch alongside `one_shot`/`logprob` in the engine's completion path, keep the one_shot fallback for servers that can't support it, and add a dedicated test module (see `tests/test_logprob_strategy.py`).

## 5. Testing

```bash
python -m pytest tests/ -q        # ~35 tests; mock backend transport — no GPU, no network
```

- `conftest.py` / `helpers.py` provide the mock OpenAI-compatible transport (scripted completions + logprobs) and profile fixtures.
- `test_compiler.py` — prompt rendering/round-trip and answer parsing edge cases.
- `test_confidence.py` — argmax / expectation / confidence math.
- `test_engine.py` — full decision pass, limits validation (422 paths), corrective retries (502 path).
- `test_logprob_strategy.py` — prefill/logprobs scoring and the one_shot fallback.

Run a single file: `python -m pytest tests/test_engine.py -v`. New behaviour gets a test before it lands; keep the suite GPU-free so CI stays cheap.

## 6. Contribution conventions

- **License**: Apache-2.0 (see `LICENSE`). Keep new files under it; model weights are licensed separately and never committed.
- **Naming discipline**: the project is spelled **JEVIATHAN** (j-e-v-i-a-t-h-a-n, with the 'a') everywhere — package dir, env vars (`JEVIATHAN_*`), aliases, docs. Terminal fonts can make it look like "jevithan"; verify names programmatically when in doubt.
- **PowerShell 5.1 gotcha**: `.ps1` files must stay **pure ASCII** (no em-dashes/unicode) — BOM-less UTF-8 is read as the system ANSI code page and corrupts parsing. `scripts/*.ps1` are enforced to be ASCII; keep it that way.
- **`manage.py` stays stdlib-only** so it runs anywhere Python does (it's the ops layer, not the app).
- **No machine-specific paths in tracked files**: local weights dirs resolve via `--model-dir` > `$JEVIATHAN_MODEL_DIR` > `.jeviathan_model_dir` (gitignored); keep docs using placeholders.
- **Commits**: imperative subject, specific file adds (never `git add -A` — the working tree carries unrelated WIP and tracked noise).

## 7. Decision feedback loop (nightly)

Capture -> review -> QLoRA -> promote, all local. The shim serves the resulting
adapter with zero extra VRAM; rollback is deleting one pointer file.

### Module map

| Piece | Role |
|---|---|
| `jeviathan/feedback/store.py` | Append-only JSONL decision log + feedback sidecar (`data/decisions.jsonl`, `data/feedback.jsonl`). Reviews attach by id; latest ts wins, file order breaks ties. Byte-stable (LF endings). |
| `jeviathan/feedback/dataset.py` | Verified records -> SFT chat pairs. Prompts are re-compiled with the same `compile_request()` as inference (no template drift); targets are the canonical JSON distribution object. Usability rule: every question must be verified or the record is excluded and counted. |
| `jeviathan/feedback/trainer.py` | QLoRA core: NF4 base (same bnb config as the shim) + LoRA r=16 on attention+MLP linears, manual loop (micro-batch 1, grad-accum N), labels=-100 over prompt tokens via chat-template prefix alignment, cosine schedule with 5% warmup, per-epoch eval, best-adapter save. Lazy-imports torch/peft/bnb so the rest of Jeviathan runs without them. |
| `scripts/feedback.py` | CLI: status/list/show/log/review/build-dataset/train/promote/export-eval. `promote` writes `.jeviathan_adapter_dir` (byte-exact; control chars rejected), which the shim resolves after `$JEVIATHAN_ADAPTER_DIR`. |
| engine hook | `systemone_engine.system_one()` appends a record when `profile.feedback.enabled`; logging failures are non-fatal by design. |

### Why adapter-first on this tier

Merging LoRA into 4-bit linears and re-saving is known to degrade quality
(huggingface/peft#2321), and a bf16 full merge needs ~17 GB RAM for an 8B model.
The shim loads the NF4 base then applies `PeftModel.from_pretrained(adapter)` —
zero extra VRAM, trivially reversible. Merged-weights export is future work for
bigger machines (and GGUF export would unlock Ollama tiers).

### Invariants to keep

- Training prompts must stay byte-identical to inference prompts (same compiler,
  same chat template). If you change `SYSTEM_PROMPT`, existing datasets are stale.
- The shim's adapter apply is non-fatal: a bad `.jeviathan_adapter_dir` serves the
  bare base and logs a warning — it must never take the model server down.
- Pointer files (`.jeviathan_model_dir`, `.jeviathan_adapter_dir`) are written as
  raw bytes with LF endings; never through shell text tools that mangle escapes.

### Tests

`tests/test_feedback.py` covers store round-trip/last-wins, usability rules,
target shape vs the contract, pointer byte-exactness, and engine-hook on/off/
non-fatal — all GPU-free. The QLoRA path is exercised manually (user guide §8).

## 8. Related projects

- **TriniGard** (closed-source) consumes Jeviathan through an adapter (`core/adapters/jeviathan.py`) behind its verification engine; the seam is the System One API, so the two evolve independently.
- Ops runbook for agents: `skills/jeviathan-ops/SKILL.md`.
