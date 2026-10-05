"""OpenAI-compatible shim serving a local HF model with 4-bit quantization.

Laptop tier: serves your local Llama-class weights in NF4 (~5GB VRAM for an
8B model) without any download. Point --model-dir at the folder containing
the HF weights (config.json, tokenizer files, safetensors).

The weights dir is resolved in this order:
    1. --model-dir flag
    2. $JEVIATHAN_MODEL_DIR environment variable
    3. .jeviathan_model_dir file at repo root (one line: path to the HF weights)

Optionally a LoRA adapter from the decision feedback loop is applied on top of
the base after loading (--adapter-dir > $JEVIATHAN_ADAPTER_DIR >
.jeviathan_adapter_dir at repo root). A bad adapter dir never takes the shim
down — it serves the bare base instead.

Usage:
    python scripts/transformers_server.py \
        --port 8200 --max-model-len 8192   # when $JEVIATHAN_MODEL_DIR is set
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from fastapi import FastAPI

app = FastAPI(title="Jeviathan local model shim")
STATE: dict = {}


def _load(model_dir: str, adapter_dir: str | None = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    print(f"Loading {model_dir} (cuda={use_cuda}) ...", flush=True)

    def _bnb(**extra):
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            **extra,
        )

    if use_cuda:
        # 1) Try all-on-GPU NF4 (fastest; needs ~5GB VRAM for an 8B model).
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, quantization_config=_bnb(), device_map={"": 0}
            )
            print("Loaded NF4 fully on GPU.", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # 2) NF4 with CPU offload for the modules that don't fit.
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    quantization_config=_bnb(llm_int8_enable_fp32_cpu_offload=True),
                    device_map="auto",
                )
                print(f"Loaded NF4 with CPU offload ({exc.__class__.__name__}).", flush=True)
            except Exception as exc2:
                # 3) bf16 auto-split (slow but always works).
                print(
                    f"4-bit failed ({exc2}); falling back to bf16 auto-split",
                    flush=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir, torch_dtype=torch.bfloat16, device_map="auto"
                )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)

    # Optional LoRA adapter from the decision feedback loop. Non-fatal: a bad
    # adapter dir must never take the shim down — serve the base instead.
    if adapter_dir:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
            print(f"LoRA adapter applied from {adapter_dir}.", flush=True)
        except Exception as exc:  # noqa: BLE001 - serve the base rather than crash
            print(
                f"WARNING: could not apply LoRA adapter from {adapter_dir} ({exc}); "
                "serving base model.",
                flush=True,
            )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    print("Model ready.", flush=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/v1/models")
def models() -> dict:
    name = STATE.get("served_name", "local-model")
    return {"data": [{"id": name, "object": "model", "owned_by": "jeviathan"}]}


@app.post("/v1/chat/completions")
def chat_completions(body: dict) -> dict:
    """Sync endpoint on purpose: FastAPI runs it in a worker thread so the
    event loop stays free while (slow) generation is in flight.

    v1.1 additions (logprob strategy support):
      * assistant-prefill — if the last message has role "assistant", its
        content is treated as an already-written prefix of the reply;
      * logprobs — with "logprobs": true, choices[0].logprobs.content carries
        one entry per prefill token: {token, logprob, bytes, top_logprobs}.
        The span is scored in a single forward pass over base+prefill; base
        and prefill are tokenized separately (add_special_tokens=False on the
        prefill) to avoid BPE boundary issues at the seam. Generated
        continuation tokens go into message.content as usual.
    """
    import re as _re

    import torch

    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens") or 1024)
    temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.7)
    want_logprobs = bool(body.get("logprobs"))
    top_k = max(0, int(body.get("top_logprobs") or 0))

    # --- split off an assistant-prefill (last message) ---------------------
    prefill = ""
    if messages and messages[-1].get("role") == "assistant":
        prefill = str(messages[-1].get("content") or "")
        base_messages = messages[:-1]
    else:
        base_messages = messages

    text = tokenizer.apply_chat_template(
        base_messages, tokenize=False, add_generation_prompt=True
    )
    base_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    base_len = int(base_inputs["input_ids"].shape[1])

    # --- single forward pass over base+prefill -> span logprobs ------------
    full_ids = base_inputs["input_ids"]
    span_items: list[dict] | None = None
    if prefill and want_logprobs:
        pf_ids = tokenizer(prefill, add_special_tokens=False)["input_ids"]
        pf_tensor = torch.tensor([pf_ids], device=full_ids.device)
        full_ids = torch.cat([base_inputs["input_ids"], pf_tensor], dim=1)
        with torch.inference_mode():
            out = model(input_ids=full_ids)
        # Position t predicts token t+1: span tokens occupy positions
        # base_len..L-1, predicted by logits[base_len-1 .. L-2].
        span_logits = out.logits[0, base_len - 1 : -1]  # (L_pf, V)
        span_ids = full_ids[0, base_len:]  # (L_pf,)
        lp_all = torch.log_softmax(span_logits.float(), dim=-1)
        span_lp = lp_all.gather(1, span_ids.unsqueeze(1)).squeeze(1).tolist()
        k = min(max(top_k, 1), int(lp_all.shape[-1]))
        top_vals, top_idx = torch.topk(lp_all, k=k, dim=-1)
        span_items = []
        for tok_id, lp, tv, ti in zip(
            span_ids.tolist(), span_lp, top_vals.tolist(), top_idx.tolist()
        ):
            item: dict = {
                "token": tokenizer.decode([tok_id]),
                "logprob": round(lp, 6),
                "bytes": list(tokenizer.decode([tok_id]).encode("utf-8")),
            }
            if k:
                item["top_logprobs"] = [
                    {"token": tokenizer.decode([int(t)]), "logprob": round(float(v), 6)}
                    for t, v in zip(ti, tv)
                ]
            span_items.append(item)

    # --- generate the continuation ------------------------------------------
    # Scoring calls only need a short tail; cap it so a stray large max_tokens
    # can't burn minutes of laptop GPU time on tokens nobody reads.
    gen_max = min(max_tokens, 32) if (prefill and want_logprobs) else max_tokens
    with torch.inference_mode():
        gen = model.generate(
            input_ids=full_ids,
            max_new_tokens=max(gen_max, 0),
            do_sample=temperature > 0.01,
            temperature=max(temperature, 1e-2),
            top_p=float(body.get("top_p") or 1.0),
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = gen[0][full_ids.shape[1]:]
    out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Llama-3.1 quirk: trailing <|eot_id>N</eot_id> index loops after the answer.
    out_text = _re.sub(r"<\|eot_id\|>\d*<\|eot_id\|>", "", out_text)
    out_text = _re.sub(r"</?eot_id>|<\|eot_id\|>", "", out_text)

    if new_tokens.shape[0] == 0:
        finish_reason = "stop"
    elif int(new_tokens[-1].item()) == tokenizer.eos_token_id:
        finish_reason = "stop"
    else:
        finish_reason = "length"

    choice: dict = {
        "index": 0,
        "message": {"role": "assistant", "content": out_text},
        "finish_reason": finish_reason,
    }
    if want_logprobs and span_items is not None:
        choice["logprobs"] = {"content": span_items}

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or STATE.get("served_name", "local-model"),
        "choices": [choice],
        "usage": {
            "prompt_tokens": int(full_ids.shape[1]),
            "completion_tokens": int(new_tokens.shape[0]),
            "total_tokens": int(full_ids.shape[1] + new_tokens.shape[0]),
        },
    }


def resolve_adapter_dir(cli_value: str | None = None) -> str | None:
    """LoRA adapter dir resolution order (decision feedback loop):
    --adapter-dir > $JEVIATHAN_ADAPTER_DIR > .jeviathan_adapter_dir at repo
    root (gitignored, one line). Returns None when unset — the shim then
    serves the bare base model."""
    if cli_value:
        return cli_value
    env = os.environ.get("JEVIATHAN_ADAPTER_DIR")
    if env:
        return env
    local = Path(__file__).resolve().parent.parent / ".jeviathan_adapter_dir"
    if local.is_file():
        p = local.read_text(encoding="utf-8").strip()
        if p:
            return p
    return None


def resolve_model_dir(cli_value: str | None = None) -> str | None:
    """Weights dir resolution order: --model-dir > $JEVIATHAN_MODEL_DIR >
    .jeviathan_model_dir at repo root (gitignored). Returns None if unset.

    Anchored at the script's repo root so it works from any CWD (manage.py
    keeps a parallel resolver for its own CLI; keep the two in sync).
    """
    if cli_value:
        return cli_value
    env = os.environ.get("JEVIATHAN_MODEL_DIR")
    if env:
        return env
    local = Path(__file__).resolve().parent.parent / ".jeviathan_model_dir"
    if local.is_file():
        p = local.read_text(encoding="utf-8").strip()
        if p:
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None,
                        help="HF weights dir; falls back to $JEVIATHAN_MODEL_DIR "
                             "or a .jeviathan_model_dir file at repo root")
    parser.add_argument("--adapter-dir", default=None,
                        help="LoRA adapter dir (decision feedback loop); falls "
                             "back to $JEVIATHAN_ADAPTER_DIR or the "
                             ".jeviathan_adapter_dir file at repo root")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--served-name", default="llama3.1-8b-local")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    if not model_dir:
        parser.error(
            "no weights dir given: pass --model-dir, set $JEVIATHAN_MODEL_DIR, "
            "or create a .jeviathan_model_dir file at the repo root "
            "(one line: path to HF weights)"
        )

    STATE["served_name"] = args.served_name
    _load(model_dir, resolve_adapter_dir(args.adapter_dir))

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
