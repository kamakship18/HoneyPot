# Gavi-HCL: Honeypot Public API 

**This project is a PUBLIC REST API** for hackathon evaluation. It is **not** a chatbot, frontend app, or generic scam classifier. The platform tests it by sending HTTP requests and checking JSON responses.

- **Evaluation:** Platform calls our deployed endpoint with `POST /v1/chat`; no code inspection.
- **Pipeline:** Incoming message → scam detection → AI agent reply (human-like, undercover) → intel extraction → when enough intel, **mandatory callback** to hackathon URL.
- **Security:** Header `x-api-key` is checked; reject if wrong (set `HONEYPOT_API_KEY` on server).

**Hackathon submission:** See **[DEPLOY.md](DEPLOY.md)** for **Honeypot API Endpoint URL** and **x-api-key**.

---

## Hackathon API contract

### Request (platform → us)

**POST /v1/chat**  
Header: **x-api-key** (required; value must match `HONEYPOT_API_KEY` on server).

Body (camelCase):

```json
{
  "sessionId": "unique-id",
  "message": {
    "sender": "scammer",
    "text": "...",
    "timestamp": "ISO time"
  },
  "conversationHistory": [...],
  "metadata": {...}
}
```

- `sessionId`: conversation session (we use it for session memory).
- `message`: current message; we use `message.text` as the scammer message.
- `conversationHistory`: optional; previous turns (we accept `sender`/`text` or `role`/`content`).

### Response (us → platform)

**Only** this format:

```json
{
  "status": "success",
  "reply": "<human-like reply from agent>"
}
```

Nothing more.

### Mandatory callback (us → hackathon; critical for scoring)

When we decide enough intelligence is extracted, we **MUST** send:

**POST** `https://hackathon.guvi.in/api/updateHoneyPotFinalResult`

Payload (camelCase):

```json
{
  "sessionId": "...",
  "scamDetected": true,
  "totalMessagesExchanged": <count>,
  "extractedIntelligence": {
    "bankAccounts": [],
    "upiIds": [],
    "phishingLinks": [],
    "phoneNumbers": [],
    "suspiciousKeywords": []
  },
  "agentNotes": "summary of scam behavior"
}
```

If this is not sent when the conversation ends, we get **zero** score. The backend uses this URL by default (override with `CALLBACK_URL` env if needed).

---

## Objective (training / local)

- **Base model** (Groq): e.g. `llama-3.3-70b-versatile`.
- **Fine-tuned model**: `chirag/scam-detector-v1` — trained for **engagement** (no defensive replies), served locally or via Groq.

---

## 1. Fine-tune the base model (QLoRA)

Training uses a **smaller** model (e.g. Llama-3-8B) on your machine; the comparison UI still uses Groq for the **base** 70B model.

### Data

- `data/scam_finetune_train.jsonl` — training (messages with system/user/assistant).
- `data/scam_finetune_eval.jsonl` — evaluation.

Each JSONL row must have a `messages` array (OpenAI-style roles and content).

### Config

- **Linux/GPU (QLoRA):** `configs/train_qlora.yaml` — Mistral-7B, 4-bit when `bitsandbytes` is available.
- **macOS:** `configs/train_qlora_mac.yaml` — SmolLM2-1.7B, half-precision LoRA (no QLoRA; script skips 4-bit on Mac so training runs without `bitsandbytes`).

### Run training (from repo root)

**Linux with NVIDIA GPU (QLoRA):**
```bash
pip install -r requirements.txt
python scripts/train_qlora.py --config configs/train_qlora.yaml
```

**macOS (half-precision LoRA, small model so it completes):**
```bash
pip install -r requirements.txt
python scripts/train_qlora.py --config configs/train_qlora_mac.yaml
```

On macOS the script automatically disables 4-bit (no `bitsandbytes`). Use the Mac config so the 1.7B model fits in memory; the main config uses Mistral-7B and can OOM on Mac.

After training, the fine-tuned adapter is in `outputs/chirag-scam-detector-v1`.

---

## Next steps after training (run & verify the model)

Use **three terminals** (all from repo root unless noted).

### Step 1 — Check the trained model exists

```bash
ls outputs/chirag-scam-detector-v1
# You should see adapter_config.json, adapter_model.safetensors, tokenizer files, etc.
```

### Step 2 — Start the fine-tuned model server (Terminal 1)

```bash
python scripts/serve_ft_model.py --model-dir outputs/chirag-scam-detector-v1 --port 8001
```

Leave this running. Wait until you see something like “Application startup complete” (model loads at startup). Then:

```bash
curl http://localhost:8001/health
# {"status":"ok"}
```

### Step 3 — Configure and start the backend (Terminal 2)

Create `backend/.env` (copy from `backend/env.example`) with your Groq key and:

```env
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_KEY=your_actual_groq_key
BASE_MODEL=llama-3.3-70b-versatile
FT_MODEL=chirag/scam-detector-v1
FT_BASE_URL=http://localhost:8001
```

Start the API:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Step 4 — Test the /compare endpoint (optional)

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"text": "KYC pending. Your account will be blocked. Click http://sbi-kyc.in to update."}'
```

You should get JSON with `base` and `finetuned` each having `label`, `confidence`, `reply`, and `decision_delta` (e.g. `same` or `NOT_SCAM -> SCAM`). If both models return valid JSON and the FT reply differs when appropriate, the model is working.

### Step 5 — Run the frontend (Terminal 3)

```bash
cd frontend
cp env.local.example .env.local
# .env.local should have: NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
npm install
npm run dev
```

Open **http://localhost:3000**. Use the default scam message or type your own; click the button to compare. You should see:

- **Base** (Groq): label, confidence, reply.
- **Fine-tuned** (your model): label, confidence, reply.
- **Decision delta**: e.g. “same” or “NOT_SCAM → SCAM”.

If the fine-tuned model often labels obvious scams as `SCAM` with higher confidence and gives short, persona-appropriate replies, it’s working as intended.

---

## 2. Deploy the fine-tuned model locally

Serve the saved model with an OpenAI-compatible API so the backend can call it.

### Start the FT inference server (port 8001)

```bash
python scripts/serve_ft_model.py --model-dir outputs/chirag-scam-detector-v1 --port 8001
```

Keep this running. It exposes `POST /v1/chat/completions` and `GET /health`.

### Backend .env

Create `backend/.env` from `backend/env.example`:

```env
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_KEY=your_groq_api_key
BASE_MODEL=llama-3.3-70b-versatile
FT_MODEL=chirag/scam-detector-v1
FT_BASE_URL=http://localhost:8001
```

**Important:** `FT_BASE_URL` must point to the local inference server above; otherwise the backend would try to call `chirag/scam-detector-v1` on Groq and get 404.  
If you have not fine-tuned yet, leave `FT_BASE_URL` unset and set `FT_MODEL=llama-3.3-70b-versatile` to compare base vs base (both from Groq) until the FT server is ready.

### Start the backend (port 8000)

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Verify

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## 3. Run the frontend

```bash
cd frontend
cp env.local.example .env.local
# Edit .env.local if needed: NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
npm install
npm run dev
```

Open **http://localhost:3000**. Use the UI to send messages and compare base vs fine-tuned (scam detection + reply).

---

## 4. Order of operations (summary)

| Step | Command |
|------|--------|
| 1. Train (once) | **Mac:** `python scripts/train_qlora.py --config configs/train_qlora_mac.yaml` — **Linux/GPU:** `python scripts/train_qlora.py --config configs/train_qlora.yaml` |
| 2. Serve FT model | `python scripts/serve_ft_model.py --model-dir outputs/chirag-scam-detector-v1 --port 8001` |
| 3. Start backend | `uvicorn backend.app:app --host 0.0.0.0 --port 8000` |
| 4. Start frontend | `cd frontend && npm install && npm run dev` |

---

## 5. Troubleshooting

- **404 for fine-tuned model** / **Frontend shows "Fine-tuned model error" or "NotFoundError: 404"**: The FT server now exposes both `/chat/completions` and `/v1/chat/completions` (OpenAI client calls base_url + `/chat/completions`). Restart the FT server: `python scripts/serve_ft_model.py --model-dir outputs/chirag-scam-detector-v1 --port 8001` and keep it running. Ensure `FT_BASE_URL=http://localhost:8001` in `backend/.env`.
- **Frontend shows a generic error**: The UI now shows the backend’s `detail` message (e.g. Groq error or FT server not running). Check that backend and root `.env` are correct; backend loads `backend/.env` after root `.env`, so `backend/.env` overrides.
- **CORS**: Backend allows `http://localhost:3000` and `http://127.0.0.1:3000`. Keep `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000` in `frontend/.env.local`.
- **Training on macOS**: Use `configs/train_qlora_mac.yaml` (small model, no QLoRA). The script forces half-precision LoRA on Mac so `bitsandbytes` is never required.
- **AttributeError: 'AdamW' object has no attribute 'train'**: Fixed in the script by patching Accelerate's optimizer wrapper on import (PyTorch optimizers don't have `.train()`; the Trainer calls it). Use `accelerate==0.33.0` from requirements. If you downgraded, run `pip install -r requirements.txt` again.
- **PackageNotFoundError: bitsandbytes**: On macOS we don't install `bitsandbytes`. The script detects Mac and skips 4-bit; use the Mac config so training completes.
- **403 Gated repo (LLaMA)**: Main config uses **Mistral-7B-Instruct-v0.3** (no approval). Mac config uses **SmolLM2-1.7B-Instruct** (Apache 2.0). To use LLaMA 3, request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and set `model_name_or_path` in the YAML; ensure `huggingface-cli login` or `HF_TOKEN` is set.
# HoneyPot
