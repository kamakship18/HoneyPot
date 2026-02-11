"""
Serve the fine-tuned QLoRA model (e.g. outputs/chirag-scam-detector-v1) with an
OpenAI-compatible /v1/chat/completions endpoint so the main backend can call it
when FT_BASE_URL points here.

Usage (from repo root):
  python scripts/serve_ft_model.py --model-dir outputs/chirag-scam-detector-v1 --port 8001

Then set in backend .env:
  FT_BASE_URL=http://localhost:8001
  FT_MODEL=chirag/scam-detector-v1
"""

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Lazy imports for heavy deps
_app = None
_model = None
_tokenizer = None


def load_model_and_tokenizer(model_dir: str):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Check if directory is empty or missing required files
    files_in_dir = list(model_dir.iterdir())
    if not files_in_dir:
        raise FileNotFoundError(
            f"Model directory is empty: {model_dir}\n"
            "Please train the model first using:\n"
            "  python scripts/train_qlora.py --config configs/train_qlora_mac.yaml  # for macOS\n"
            "  python scripts/train_qlora.py --config configs/train_qlora.yaml      # for Linux/GPU"
        )

    # PEFT save contains adapter_config.json with base_model_name_or_path
    adapter_config_path = model_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        base_name = adapter_config.get("base_model_name_or_path")
        if not base_name:
            raise ValueError(f"adapter_config.json missing 'base_model_name_or_path': {adapter_config_path}")
    else:
        # If no adapter config, assume model_dir is the base model
        base_name = str(model_dir)

    # Load tokenizer from base model (not adapter dir) to avoid config conflicts
    # Also try loading from model_dir first in case tokenizer files are there
    try:
        # Check if tokenizer files exist in adapter directory
        tokenizer_files = list(model_dir.glob("tokenizer*.json")) + list(model_dir.glob("*.json"))
        has_tokenizer_in_dir = any("tokenizer" in str(f.name).lower() for f in tokenizer_files)
        
        if has_tokenizer_in_dir and (model_dir / "tokenizer_config.json").exists():
            # Load from adapter dir if tokenizer files are present
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        else:
            # Load from base model
            tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    except Exception as e:
        # Fallback to base model if adapter dir loading fails
        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter_config_path.exists():
        model = PeftModel.from_pretrained(model, str(model_dir))
    model.eval()

    _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


def run_inference(model_dir: str, messages: list, max_tokens: int = 256, temperature: float = 0.0) -> str:
    model, tokenizer = load_model_and_tokenizer(model_dir)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    reply_ids = out[0][input_len:]
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
    return reply


def create_app(model_dir: str):
    global _app
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: load model
        load_model_and_tokenizer(model_dir)
        yield
        # Shutdown: cleanup if needed
        pass
    
    app = FastAPI(title="Fine-tuned model server (OpenAI-compatible)", lifespan=lifespan)
    app.state.model_dir = model_dir

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = Field(default="chirag/scam-detector-v1")
        messages: list[ChatMessage]
        max_tokens: int = Field(default=256, ge=1, le=2048)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def _chat_completion_response(content: str):
        return {
            "id": "ft-local-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/v1/chat/completions")
    def chat_completions_v1(req: ChatCompletionRequest):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        content = run_inference(model_dir, messages, max_tokens=req.max_tokens, temperature=req.temperature)
        return _chat_completion_response(content)

    # OpenAI client calls base_url + "/chat/completions", not "/v1/chat/completions"
    @app.post("/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        content = run_inference(model_dir, messages, max_tokens=req.max_tokens, temperature=req.temperature)
        return _chat_completion_response(content)

    _app = app
    return app


def main():
    parser = argparse.ArgumentParser(description="Serve fine-tuned model with OpenAI-compatible API")
    parser.add_argument("--model-dir", default="outputs/chirag-scam-detector-v1", help="Path to saved adapter/model dir")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()

    create_app(args.model_dir)
    uvicorn.run(_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
