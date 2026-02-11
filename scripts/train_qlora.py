import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def _patch_accelerate_optimizer_for_macos():
    """Work around transformers Trainer calling optimizer.train(): PyTorch optimizers don't have .train()."""
    try:
        from accelerate.optimizer import AcceleratedOptimizer

        def _train_noop_if_missing(self):
            if hasattr(self.optimizer, "train"):
                return self.optimizer.train()
            # PyTorch AdamW etc. have no .train(); no-op.

        AcceleratedOptimizer.train = _train_noop_if_missing
    except Exception:
        pass


_patch_accelerate_optimizer_for_macos()


"""
Why this script exists (high-level):
- We want instruction-style tuning on chat data ("messages") so the model learns:
  1) Scam intent detection without revealing detection
  2) Believable Indian persona + probing questions
  3) Strict JSON-only structured extraction when requested
- QLoRA keeps training cheap: 4-bit base weights + LoRA adapters.
"""


SYSTEM_NOTE = (
    "Note: This training script expects each JSONL row to contain a `messages` array "
    "with ChatML-like roles: system/user/assistant."
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_messages_schema(rows: List[Dict[str, Any]]) -> None:
    for i, r in enumerate(rows):
        if "messages" not in r or not isinstance(r["messages"], list) or len(r["messages"]) < 3:
            raise ValueError(f"Row {i} missing valid `messages` array. {SYSTEM_NOTE}")
        for m in r["messages"]:
            if "role" not in m or "content" not in m:
                raise ValueError(f"Row {i} has malformed message: {m}. {SYSTEM_NOTE}")


def to_text(rows: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dataset:
    """
    Convert chat `messages` into a single training string using tokenizer's chat template.
    This is the most reliable way to align to LLaMA-3 / Mistral Instruct formatting.
    """

    def render(row: Dict[str, Any]) -> Dict[str, str]:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = Dataset.from_list(rows)
    ds = ds.map(render, remove_columns=ds.column_names)
    return ds


@dataclass
class Config:
    model_name_or_path: str
    train_jsonl: str
    eval_jsonl: str
    output_dir: str
    max_seq_length: int

    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int

    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str

    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    seed: int

    use_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]


def cfg_from_dict(d: Dict[str, Any]) -> Config:
    return Config(**d)


def dtype_from_name(name: str):
    name = name.lower().strip()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config (see configs/train_qlora.yaml).")
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    cfg = cfg_from_dict(raw_cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA (4-bit) requires bitsandbytes, which we skip on macOS. Use half-precision LoRA on Mac.
    use_4bit = bool(cfg.use_4bit)
    if sys.platform == "darwin":
        use_4bit = False
        print("macOS detected: using half-precision LoRA (QLoRA not available).")
    elif use_4bit:
        try:
            import bitsandbytes  # noqa: F401  # optional; not installed on macOS
        except ImportError:
            use_4bit = False
            print("bitsandbytes not installed. Using half-precision LoRA instead of QLoRA.")

    quant_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype_from_name(cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

    compute_dtype = dtype_from_name(cfg.bnb_4bit_compute_dtype)
    
    # On macOS, avoid device_map="auto" which can cause meta tensor issues
    # Use explicit device instead
    if sys.platform == "darwin":
        # Use MPS if available, otherwise CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            quantization_config=quant_config,
            torch_dtype=compute_dtype if not use_4bit else None,
        )
        model = model.to(device)
    else:
        # On Linux/GPU, use device_map="auto" for multi-GPU support
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=compute_dtype if not use_4bit else None,
        )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )
    model = get_peft_model(model, lora_config)

    train_rows = load_jsonl(cfg.train_jsonl)
    eval_rows = load_jsonl(cfg.eval_jsonl)
    ensure_messages_schema(train_rows)
    ensure_messages_schema(eval_rows)

    train_ds = to_text(train_rows, tokenizer)
    eval_ds = to_text(eval_rows, tokenizer)

    # eval_strategy (4.46+); pin_memory=False on macOS (MPS doesn't support it)
    # Determine device for training args
    if sys.platform == "darwin":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=torch.cuda.is_available(),  # Only use bf16 on CUDA
        report_to="none",
        seed=cfg.seed,
        dataloader_pin_memory=(sys.platform != "darwin"),
    )

    # SFTTrainer handles standard causal LM supervised fine-tuning.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()


