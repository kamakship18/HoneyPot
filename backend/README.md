## Backend (FastAPI) — Agentic Honeypot

### Env vars
- `GROQ_BASE_URL`, `GROQ_API_KEY`, `BASE_MODEL`: Groq API.
- `FT_MODEL`, `FT_BASE_URL`, `FT_API_KEY`: Fine-tuned model (local or Groq).
- **`CALLBACK_URL`**: Mandatory for hackathon; POST payload when `conversation_ended=true`.

### Local .env
Create `backend/.env` from `backend/env.example`. Set `FT_BASE_URL`, `CALLBACK_URL`, and run the FT inference server for local FT model.

### Run

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Endpoints

**Public API (evaluation): POST /v1/chat**

Request: `{ "session_id": "…", "message": "…", "conversation_history": [{"role":"scammer|agent","content":"…"}] }`  
Response: `{ "reply", "scam_detected", "confidence", "intelligence", "conversation_ended", "callback_sent" }`.  
When `conversation_ended=true`, backend POSTs to `CALLBACK_URL`.

**Dev / health**
- `GET /health`
- `POST /compare` — base vs fine-tuned reply (dev only)


