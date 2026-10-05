"""QLoRA training core for the decision feedback loop.

Trains LoRA adapters on top of an NF4-quantized base (bitsandbytes) — the
same quantization the shim serves with — so an 8B model trains in ~6GB VRAM
(LoRA params are ~0.2% of the model; only they get gradients). Lazy-imports
torch/peft/bitsandbytes so the rest of Jeviathan runs without them.

Per-run outputs:
  adapter/           PEFT LoRA adapter (small; what `promote` points at)
  train_report.json  hyperparams, per-epoch losses, VRAM peak, dataset stats

Why adapters instead of merged weights on this tier: merging a LoRA into a
4-bit linear and re-saving is known to degrade quality (huggingface/peft#2321),
and a bf16 full merge needs ~17GB RAM. The adapter adds zero VRAM at serving
time (the shim applies it on top of the NF4 base) and rollback is deleting one
file. Merged-weights export is future work for machines with more headroom.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path


def _require_training_deps() -> None:
    try:
        import torch  # noqa: F401
        import peft  # noqa: F401
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "training dependencies missing — install them first:\n"
            f"  pip install -r requirements-train.txt\n({exc})"
        ) from exc


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def qlora_train(
    base_dir: str | Path,
    train_items: list[dict],
    eval_items: list[dict],
    out_dir: str | Path,
    *,
    rank: int = 16,
    alpha: int | None = None,
    lr: float = 1e-4,
    epochs: int = 3,
    max_len: int = 512,
    grad_accum: int = 4,
    seed: int = 42,
    max_steps: int = 0,
) -> dict:
    """Run QLoRA SFT on chat-template items. Returns the report dict (also
    written to out_dir/train_report.json).

    Items are {"messages": [system, user, assistant]}. Prompt tokens get
    labels=-100; only the assistant completion is supervised — exactly what
    the model generates at inference time from the same prefix. Micro-batch
    is always 1 (6GB tier); grad_accum sets the effective batch.
    """
    _require_training_deps()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not train_items:
        raise SystemExit("no training items — review decisions first (scripts/feedback.py list)")
    alpha = alpha or 2 * rank
    out_dir = Path(out_dir)
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training needs an NVIDIA GPU (CUDA).")

    t0 = time.time()
    print(f"[train] loading {base_dir} in NF4 ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, quantization_config=bnb_cfg, device_map={"": 0}
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    # torch >= 2.9 wants an explicit use_reentrant for gradient checkpointing.
    if hasattr(model, "gradient_checkpointing_kwargs"):
        model.gradient_checkpointing_kwargs = {"use_reentrant": False}

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- tokenization: inference-identical prompts -------------------------
    def encode(item: dict):
        """(input_ids, attention_mask, labels) or None when over max_len."""
        msgs = item["messages"]
        # Render with the template (inference-identical text), then tokenize
        # each side explicitly — apply_chat_template(tokenize=True) return
        # types vary across transformers versions.
        prompt_text = tokenizer.apply_chat_template(
            msgs[:2], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        # The assistant turn must extend the generation prefix; if a template
        # quirk breaks that assumption, fall back to the common prefix.
        cut = _common_prefix_len(prompt_ids, full_ids)
        if len(full_ids) <= cut or len(full_ids) > max_len:
            return None
        labels = [-100] * cut + full_ids[cut:]
        ids = torch.tensor([full_ids], dtype=torch.long)
        att = torch.ones_like(ids)
        lab = torch.tensor([labels], dtype=torch.long)
        return ids.to("cuda"), att.to("cuda"), lab.to("cuda")

    def mean_loss(items: list[dict]) -> float | None:
        total, n = 0.0, 0
        with torch.inference_mode():
            for item in items:
                enc = encode(item)
                if enc is None:
                    continue
                out = model(input_ids=enc[0], attention_mask=enc[1], labels=enc[2])
                total += float(out.loss)
                n += 1
        return total / n if n else None

    # ---- optimizer + cosine schedule with warmup ---------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(train_items) / grad_accum))
    total_steps = min(steps_per_epoch * epochs, max_steps) if max_steps else steps_per_epoch * epochs
    warmup = max(1, int(total_steps * 0.05))

    def lr_at(step: int) -> float:
        if step <= warmup:
            return lr * step / warmup
        prog = min((step - warmup) / max(1, total_steps - warmup), 1.0)
        return lr * 0.5 * (1.0 + math.cos(math.pi * prog))

    # ---- training loop ------------------------------------------------------
    random.seed(seed)
    model.train()
    history: list[dict] = []
    best = {"loss": float("inf"), "epoch": 0}
    global_step = 0

    for epoch in range(epochs):
        shuffled = train_items[:]
        random.shuffle(shuffled)
        opt.zero_grad(set_to_none=True)
        accum, epoch_loss, n_seen = 0, 0.0, 0
        for i, item in enumerate(shuffled):
            enc = encode(item)
            if enc is None:
                continue
            out = model(input_ids=enc[0], attention_mask=enc[1], labels=enc[2])
            (out.loss / grad_accum).backward()
            accum += 1
            epoch_loss += float(out.loss.detach())
            n_seen += 1
            if accum % grad_accum == 0 or i == len(shuffled) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                for group in opt.param_groups:
                    group["lr"] = lr_at(global_step)
            if max_steps and global_step >= max_steps:
                break
        entry = {"epoch": epoch + 1, "train_loss": round(epoch_loss / max(1, n_seen), 4)}
        if eval_items and not (max_steps and global_step >= max_steps):
            model.eval()
            ev = mean_loss(eval_items)
            model.train()
            if ev is not None:
                entry["eval_loss"] = round(ev, 4)
                if ev < best["loss"]:
                    best = {"loss": ev, "epoch": epoch + 1}
        history.append(entry)
        print(f"[train] epoch {epoch+1}/{epochs} {entry}", flush=True)
        if max_steps and global_step >= max_steps:
            break

    # ---- save best (or last) adapter ---------------------------------------
    model.save_pretrained(adapter_dir)
    base_cfg = Path(base_dir) / "config.json"
    base_model_id = None
    if base_cfg.is_file():
        try:
            base_model_id = json.loads(base_cfg.read_text(encoding="utf-8")).get("_name_or_path")
        except (json.JSONDecodeError, OSError):
            pass

    report = {
        "base_dir": str(Path(base_dir).resolve()),
        "base_model_id": base_model_id,
        "n_train": len(train_items),
        "n_eval": len(eval_items),
        "hyperparams": {
            "rank": rank, "alpha": alpha, "lr": lr, "epochs": epochs,
            "max_len": max_len, "grad_accum": grad_accum,
            "seed": seed, "max_steps": max_steps,
        },
        "history": history,
        "best_epoch": best["epoch"],
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "adapter_dir": str(adapter_dir.resolve()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train] done in {report['elapsed_s']}s — adapter at {adapter_dir}", flush=True)
    return report
