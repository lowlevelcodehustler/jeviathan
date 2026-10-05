"""Decision feedback loop — learn from Jeviathan's own decision history.

Every System One pass can be logged (profile opt-in, `feedback.enabled`), a
reviewer attaches verdicts and corrections, and the verified set is used to
QLoRA-fine-tune the local model into new weights that the shim serves after
`scripts/feedback.py promote`.

Modules:
  store.py    append-only JSONL decision log + feedback sidecar (no rewrites)
  dataset.py  corrected decisions -> SFT chat pairs with inference-identical prompts
  trainer.py  QLoRA training core (PEFT + bitsandbytes, lazy imports so the
              rest of Jeviathan runs without them)

Ollama/vLLM tiers: capture works on every backend; `promote` applies to the
shim/PyTorch tier because it is the only one that loads HF weights directly.
"""
