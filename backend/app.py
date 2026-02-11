import json
import logging
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.requests import Request

# MongoDB imports
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning("MongoDB libraries not installed. Install pymongo and motor for database storage.")

try:
    # Loads .env from repo root (parent of backend/) for local dev.
    from dotenv import load_dotenv  # type: ignore

    # Try repo root .env first, then backend/.env
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv(repo_root / "backend" / ".env")
except Exception:
    # Optional dependency; backend can still run with exported env vars.
    pass


"""
Hackathon: Public REST API for Honeypot evaluation.

- Platform sends POST to our endpoint with sessionId, message (sender, text, timestamp), conversationHistory, metadata.
- We return ONLY: {"status": "success", "reply": "<agent reply>"}.
- When we decide enough intel is extracted, we MUST POST to hackathon callback URL with the required payload.

Env: GROQ_*, BASE_MODEL, FT_*, CALLBACK_URL (defaults to hackathon URL), HONEYPOT_API_KEY (x-api-key check).
MongoDB: MONGODB_URI (optional, defaults to mongodb://localhost:27017), MONGODB_DB_NAME (defaults to honeypot_db).
"""

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ========== MongoDB Connection ==========
_mongo_client: Optional[Any] = None
_mongo_db: Optional[Any] = None
_mongodb_available: bool = False


async def _init_mongodb_async():
    """Initialize MongoDB at app startup (uses uvicorn's event loop)."""
    global _mongo_client, _mongo_db, _mongodb_available

    if not MONGO_AVAILABLE:
        logger.warning("MongoDB libraries not installed - using in-memory storage only")
        return

    mongo_uri = os.environ.get("MONGODB_URI", "").strip()
    db_name = (os.environ.get("MONGODB_DB_NAME", "honeypot_db") or "").strip() or "honeypot_db"

    if not mongo_uri:
        logger.warning("MONGODB_URI not set - MongoDB storage disabled. Set MONGODB_URI in .env or environment.")
        return

    timeout_ms = 15000 if "mongodb+srv://" in mongo_uri else 5000
    try:
        _mongo_client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)
        _mongo_db = _mongo_client[db_name]
        await _mongo_client.admin.command("ping")
        _mongodb_available = True
        logger.info(f"MongoDB connected successfully: database={db_name}")
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
        logger.warning(f"MongoDB connection failed: {e}. Using in-memory storage only.")
        _mongo_client = None
        _mongo_db = None
        _mongodb_available = False


def _load_prompt(name: str) -> str:
    path = _REPO_ROOT / "prompts" / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


# Honeypot: engagement persona only; NO defensive/safety replies.
# Enhanced prompt that works like fine-tuned model
HONEYPOT_SYSTEM_PROMPT = _load_prompt("enhanced_honeypot_prompt.txt") or _load_prompt("honeypot_system_prompt.txt") or (
    "You are Aman, a real Indian user. Output ONLY JSON: "
    '{"label":"SCAM|NOT_SCAM|UNCERTAIN","confidence":0-1,"reply":"<your message>"}. '
    "Reply must be confused, curious, cooperative; never say contact bank or refuse. Extract details by asking (UPI again, link, bank). "
    "Use full conversation history to maintain continuity. Reference previous messages naturally."
)
INTEL_EXTRACTION_PROMPT = _load_prompt("intel_extraction_prompt.txt") or (
    "Extract from conversation. Output ONLY JSON: "
    '{"upi_ids":[],"links":[],"bank_accounts":[],"phone_numbers":[],"tactics":[]}.'
)

CLASSIFIER_SYSTEM_PROMPT = HONEYPOT_SYSTEM_PROMPT


class ModelOutput(BaseModel):
    label: str
    confidence: float
    reply: str
    raw: str


# ---------- Hackathon contract: request/response ----------
# Hackathon docs: timestamp is "Epoch time format in ms" (number or string); we accept both.
def _coerce_timestamp_to_str(v: Any) -> Optional[str]:
    """Accept epoch ms as number or string; normalize to str for schema compatibility."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return str(int(v))
    if isinstance(v, str):
        return v
    return None


class IncomingMessage(BaseModel):
    sender: str = Field(..., description="e.g. scammer")
    text: str = Field(..., min_length=1, max_length=8000)
    timestamp: Optional[str] = Field(None, description="Epoch time in ms (number or string)")

    @field_validator("timestamp", mode="before")
    @classmethod
    def timestamp_accept_number_or_string(cls, v: Any) -> Optional[str]:
        return _coerce_timestamp_to_str(v)


class ConversationHistoryEntry(BaseModel):
    """One turn in conversationHistory; platform may send sender/text or role/content."""
    sender: Optional[str] = None
    text: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    timestamp: Optional[str] = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def timestamp_accept_number_or_string(cls, v: Any) -> Optional[str]:
        return _coerce_timestamp_to_str(v)


class HackathonRequest(BaseModel):
    """Exact contract: camelCase from platform."""
    sessionId: str = Field(..., min_length=1, alias="sessionId")
    message: IncomingMessage = Field(..., alias="message")
    conversationHistory: Optional[List[ConversationHistoryEntry]] = Field(None, alias="conversationHistory")
    metadata: Optional[Dict[str, Any]] = Field(None, alias="metadata")

    class Config:
        populate_by_name = True


class HackathonResponse(BaseModel):
    """Exact contract: status/reply, plus scamDetected for scam gating."""
    status: str = Field(..., description="Must be 'success'")
    reply: str = Field(..., description="Human-like reply from agent (empty if NOT_SCAM)")
    scamDetected: bool = Field(..., description="True when scam intent confirmed")


def _client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency `openai`. Install requirements.txt.") from e

    base_url = os.environ.get("GROQ_BASE_URL", "").strip()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not base_url or not api_key:
        raise RuntimeError("Set GROQ_BASE_URL and GROQ_API_KEY env vars.")
    return OpenAI(base_url=base_url, api_key=api_key)


# FINE-TUNED MODEL CODE COMMENTED OUT - Using base model with enhanced prompt instead
# def _ft_client():
#     """Client for fine-tuned model: uses FT_BASE_URL if set, else same as Groq (base)."""
#     try:
#         from openai import OpenAI  # type: ignore
#     except Exception as e:
#         raise RuntimeError("Missing dependency `openai`. Install requirements.txt.") from e
# 
#     ft_base = os.environ.get("FT_BASE_URL", "").strip()
#     if ft_base:
#         api_key = os.environ.get("FT_API_KEY", "").strip() or "local"
#         return OpenAI(base_url=ft_base.rstrip("/"), api_key=api_key)
#     return _client()

# Always use base client (Groq API)
def _ft_client():
    """Always returns base client - fine-tuned model disabled."""
    return _client()


def _build_conversation_messages(
    conversation_history: List[Dict[str, str]], 
    current_message: str,
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Build proper chat history with system/user/assistant roles for multi-turn conversations.
    This ensures the model sees full conversation context and can continue naturally.
    """
    prompt = system_prompt or CLASSIFIER_SYSTEM_PROMPT
    messages = [{"role": "system", "content": prompt}]
    
    # Add full conversation history with proper roles
    for msg in conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        
        # Map internal roles to OpenAI chat format
        if role == "scammer":
            messages.append({"role": "user", "content": content})
        elif role == "agent":
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": role, "content": content})
    
    # Add current message
    messages.append({"role": "user", "content": current_message})
    
    return messages


def _parse_model_output(raw: str, strict: bool = True) -> ModelOutput:
    """Parse raw model text into ModelOutput. If strict, raise 502 on parse/label errors."""
    try:
        parsed: Any = json.loads(raw)
    except Exception:
        if strict:
            raise HTTPException(status_code=502, detail=f"Model did not return valid JSON. raw={raw[:200]}")
        return ModelOutput(
            label="UNCERTAIN",
            confidence=0.5,
            reply=raw[:500] if raw else "(no output)",
            raw=raw,
        )

    # FT/model may return a JSON string or non-dict (e.g. "hello") -> treat as raw text
    if not isinstance(parsed, dict):
        if strict:
            raise HTTPException(status_code=502, detail=f"Model did not return a JSON object. raw={raw[:200]}")
        return ModelOutput(
            label="UNCERTAIN",
            confidence=0.5,
            reply=(raw[:500] if raw else "(no output)"),
            raw=raw,
        )

    label = str(parsed.get("label", "")).strip()
    if not label and parsed.get("is_scam") is not None:
        label = "SCAM" if parsed.get("is_scam") else "NOT_SCAM"
    confidence = float(parsed.get("confidence", 0.0))
    reply = str(parsed.get("reply", "")).strip() or (raw[:300] if raw else "")
    if label not in ("SCAM", "NOT_SCAM", "UNCERTAIN"):
        if strict:
            raise HTTPException(status_code=502, detail=f"Bad label from model: {label}")
        label = "UNCERTAIN"
    return ModelOutput(label=label, confidence=confidence, reply=reply, raw=raw)


def _call_model(
    model: str, 
    conversation_history: List[Dict[str, str]],
    current_message: str,
    use_ft_client: bool = False,  # Ignored now - always uses base model
    max_tokens: int = 512,
    system_prompt: Optional[str] = None,
    use_rag: bool = True,  # Enable RAG by default
    extracted_intel: Optional[Dict[str, List[str]]] = None,
) -> ModelOutput:
    """
    Call base model with full conversation history, enhanced prompt, and RAG.
    Fine-tuned model disabled - using enhanced prompt + RAG instead.
    """
    # Always use base client (fine-tuned model disabled)
    client = _client()
    
    # Use RAG-enhanced messages if enabled
    if use_rag:
        messages = _build_rag_enhanced_messages(
            conversation_history, current_message, system_prompt, extracted_intel
        )
    else:
        messages = _build_conversation_messages(conversation_history, current_message, system_prompt)
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,  # Higher temperature for varied, natural questions and human-like replies
        max_tokens=max_tokens,
        stop=None,  # Let model decide when to stop naturally
    )
    raw = (resp.choices[0].message.content or "").strip()
    
    # Log for debugging
    logger.debug(f"Model response (first 200 chars): {raw[:200]}")
    
    # Use strict=False for base model to handle parsing errors gracefully
    return _parse_model_output(raw, strict=False)


def _call_llm(system_prompt: str, user_content: str, use_ft_client: bool = True, max_tokens: int = 512) -> str:
    """Single LLM call with given system and user content. Returns raw content."""
    # Always use base model - fine-tuned model disabled
    model = os.environ.get("BASE_MODEL", "").strip()
    if not model:
        raise RuntimeError("Set BASE_MODEL env var.")
    # Always use base client
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------- Session store (in-memory fallback; keyed by session_id) ----------
_session_store: Dict[str, Dict[str, Any]] = {}


async def _save_to_mongodb(collection: str, document: Dict[str, Any]) -> bool:
    """Save document to MongoDB. Returns True if successful."""
    if (not _mongodb_available) or (_mongo_db is None):
        logger.debug(f"MongoDB not available, skipping save to {collection}")
        return False
    
    try:
        result = await _mongo_db[collection].insert_one(document)
        if result.inserted_id:
            logger.info(f"✅ MongoDB: Inserted into {collection}, _id: {result.inserted_id}")
            return True
        else:
            logger.warning(f"⚠️ MongoDB: Insert returned no _id for {collection}")
            return False
    except Exception as e:
        logger.error(f"❌ MongoDB save error to {collection}: {e}", exc_info=True)
        return False


async def _get_from_mongodb(collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get document from MongoDB. Returns None if not found or error."""
    if (not _mongodb_available) or (_mongo_db is None):
        return None
    
    try:
        result = await _mongo_db[collection].find_one(query)
        if result and "_id" in result:
            result["_id"] = str(result["_id"])  # Convert ObjectId to string
        return result
    except Exception as e:
        logger.error(f"MongoDB get error: {e}")
        return None


async def _update_mongodb(collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
    """Update document in MongoDB. Returns True if successful."""
    if not _mongodb_available or _mongo_db is None:
        return False
    
    try:
        result = await _mongo_db[collection].update_one(query, {"$set": update})
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"MongoDB update error: {e}")
        return False


def _get_session(session_id: str) -> Dict[str, Any]:
    """Get session from memory (MongoDB is async, so we use sync in-memory for now)."""
    if session_id not in _session_store:
        _session_store[session_id] = {
            "messages": [],
            "turn_count": 0,
            "scam_detected": False,
            "callback_sent": False,
            "extracted_intel": {
                "upi_ids": [],
                "links": [],
                "bank_accounts": [],
                "ifsc_codes": [],
                "bank_names": [],
                "branch_names": [],
                "phone_numbers": [],
                "emails": [],
                "suspicious_keywords": [],
                "tactics": [],
            },
        }
    return _session_store[session_id]


def _conversation_context(messages: List[Dict[str, str]]) -> str:
    """
    Convert messages to readable text format for logging/debugging.
    Note: This is NOT used for model input - we use _build_conversation_messages instead.
    """
    if not messages:
        return ""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if role == "scammer":
            lines.append(f"Scammer: {content}")
        elif role == "agent":
            lines.append(f"Aman: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


BANK_NAME_KEYWORDS = [
    "sbi",
    "state bank",
    "hdfc",
    "icici",
    "axis",
    "kotak",
    "pnb",
    "punjab national",
    "bank of baroda",
    "canara",
    "union bank",
    "idfc",
    "yes bank",
    "indusind",
    "bank of india",
    "central bank",
    "uco",
    "federal bank",
]


def _regex_extract_intel(text: str) -> Dict[str, List[str]]:
    import re

    lowered = (text or "").lower()
    upi_ids = re.findall(r"\b[\w.-]{2,}@[\w.-]{2,}\b", text)
    links = re.findall(r"(https?://\S+|www\.\S+)", text)
    phones = re.findall(r"\b(?:\+?\d{1,3}[-\s]?)?\d{10}\b", text)
    accounts = re.findall(r"\b\d{9,18}\b", text)
    emails = re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE)
    ifsc = re.findall(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", text)
    branch = re.findall(r"\bbranch[:\-]?\s*([A-Za-z ]{2,30})\b", text, flags=re.IGNORECASE)

    bank_names = []
    for name in BANK_NAME_KEYWORDS:
        if name in lowered:
            bank_names.append(name)

    return {
        "upi_ids": list(set(upi_ids)),
        "links": list(set([l[0] if isinstance(l, tuple) else l for l in links])),
        "bank_accounts": list(set(accounts)),
        "ifsc_codes": list(set([c.upper() for c in ifsc])),
        "bank_names": list(set(bank_names)),
        "branch_names": list(set([b.strip() for b in branch if b.strip()])),
        "phone_numbers": list(set(phones)),
        "emails": list(set(emails)),
        "suspicious_keywords": list(set(_keyword_hits(lowered))),
        "tactics": [],
    }


def _merge_intel(base: Optional[Dict[str, List[str]]], extra: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    if base:
        for key, values in base.items():
            merged[key] = list(values or [])
    for key, values in extra.items():
        merged[key] = list(set(merged.get(key, [])) | set(values or []))
    return merged


def _stop_condition_met(intel: Dict[str, List[str]]) -> bool:
    """Stop if ANY of the primary extraction targets have been captured."""
    if intel.get("upi_ids"):
        return True
    if intel.get("bank_accounts") and intel.get("ifsc_codes"):
        return True
    if intel.get("phone_numbers"):
        return True
    if intel.get("emails"):
        return True
    # Name — check both bank_names and any explicit names extracted
    if intel.get("bank_names"):
        return True
    return False


def _agent_turns(messages: List[Dict[str, str]]) -> int:
    return len([m for m in messages if m.get("role") == "agent"])



# ========== RAG (Retrieval Augmented Generation) Implementation ==========
# Strategy per scam type with question guidance
_RAG_KNOWLEDGE_BASE = [
    {
        "pattern": "KYC pending account freeze",
        "keywords": ["kyc", "pending", "freeze", "block", "account", "compromised", "locked", "otp"],
        "intel_targets": ["bank_accounts", "ifsc_codes", "names", "upi_ids", "emails", "phone_numbers"],
        "question_types": [
            "Ask which bank/company this is from",
            "Ask why account is blocked and how to fix it",
            "Mention UPI not working, ask for bank account and IFSC",
            "Ask if there's another way to verify",
            "Ask for their contact number to discuss",
        ],
    },
    {
        "pattern": "lottery prize processing fee",
        "keywords": ["lottery", "prize", "processing fee", "lakh", "crore", "won"],
        "intel_targets": ["upi_ids", "phone_numbers", "bank_accounts", "names", "emails"],
        "question_types": [
            "Ask which company/lottery this is",
            "Ask how you won and when",
            "Mention UPI not working, ask for alternative payment method",
            "Ask which account should receive the money",
            "Ask for their UPI ID or bank details",
        ],
    },
    {
        "pattern": "parcel customs duty",
        "keywords": ["parcel", "customs", "duty", "courier", "stuck"],
        "intel_targets": ["upi_ids", "bank_accounts", "phone_numbers", "emails"],
        "question_types": [
            "Ask which courier company",
            "Ask for tracking number",
            "Mention UPI not working, ask for bank transfer details",
            "Ask which account to pay to",
            "Ask for their contact number",
        ],
    },
    {
        "pattern": "refund payment request",
        "keywords": ["refund", "payment", "upi", "transfer", "money"],
        "intel_targets": ["upi_ids", "bank_accounts", "ifsc_codes", "phone_numbers", "emails"],
        "question_types": [
            "Ask which company/order this refund is for",
            "Mention UPI not working, ask for bank account and IFSC",
            "Ask which account should receive the refund",
            "Ask for their UPI ID or bank details",
            "Ask if there's another payment method",
        ],
    },
    {
        "pattern": "electricity bill overdue",
        "keywords": ["bill", "overdue", "electricity", "power", "last date"],
        "intel_targets": ["links", "upi_ids", "bank_accounts", "phone_numbers"],
        "question_types": [
            "Ask which electricity company",
            "Mention UPI not working, ask for bank account details",
            "Ask which account number to pay to",
            "Ask for their contact number",
            "Ask if there's another way to pay",
        ],
    },
    {
        "pattern": "IT support remote access",
        "keywords": ["IT", "support", "license", "expired", "anydesk", "teamviewer"],
        "intel_targets": ["phone_numbers", "emails", "bank_accounts", "names"],
        "question_types": [
            "Ask which company they're from",
            "Ask what exactly happened to the license",
            "Ask for their phone number to discuss",
            "Ask for their email to send details",
            "Ask if there's another way to fix this",
        ],
    },
    {
        "pattern": "police cyber cell penalty",
        "keywords": ["police", "cyber", "penalty", "case", "legal"],
        "intel_targets": ["bank_accounts", "ifsc_codes", "phone_numbers", "names"],
        "question_types": [
            "Ask which police station",
            "Ask what the case is about",
            "Mention UPI not working, ask for bank account and IFSC",
            "Ask which account to pay to",
            "Ask for their contact number",
        ],
    },
    {
        "pattern": "crypto investment",
        "keywords": ["crypto", "investment", "guaranteed", "monthly", "deposit"],
        "intel_targets": ["links", "upi_ids", "bank_accounts", "phone_numbers"],
        "question_types": [
            "Ask which company/platform",
            "Ask how this investment works",
            "Mention UPI not working, ask for bank account details",
            "Ask which account to transfer money to",
            "Ask for their UPI ID or contact",
        ],
    },
]


def _rag_retrieve_context(
    message_text: str, 
    conversation_history: List[Dict[str, str]],
    extracted_intel: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    RAG: Build situational guidance — what intel is missing, what questions to ask, what to avoid repeating.
    """
    message_lower = message_text.lower()
    context_parts = []
    
    # 1. Identify scam pattern and get question guidance
    matched_pattern = None
    for kb_entry in _RAG_KNOWLEDGE_BASE:
        keyword_matches = sum(1 for kw in kb_entry["keywords"] if kw in message_lower)
        if keyword_matches >= 2:
            matched_pattern = kb_entry
            context_parts.append(f"Likely scam type: {kb_entry['pattern']}")
            # Add question guidance
            if "question_types" in kb_entry:
                context_parts.append("Suggested question approaches (use ONE, reword naturally):")
                for q_type in kb_entry["question_types"][:3]:  # Show top 3
                    context_parts.append(f"  - {q_type}")
            break  # Only match one pattern
    
    # 2. Tell the LLM what's already extracted vs still missing
    if extracted_intel:
        got = []
        missing = []
        missing_details = []
        for key, label in [
            ("upi_ids", "UPI ID"), ("bank_accounts", "Bank account"), 
            ("ifsc_codes", "IFSC"), ("phone_numbers", "Phone number"),
            ("emails", "Email"), ("bank_names", "Name"),
        ]:
            if extracted_intel.get(key) and len(extracted_intel.get(key, [])) > 0:
                got.append(label)
            else:
                missing.append(label)
                missing_details.append(key)
        if got:
            context_parts.append(f"Already obtained: {', '.join(got)}")
        if missing:
            context_parts.append(f"Still needed: {', '.join(missing)}")
            # Prioritize what to ask for based on pattern
            if matched_pattern and "intel_targets" in matched_pattern:
                priority = [d for d in missing_details if d in matched_pattern["intel_targets"]]
                if priority:
                    priority_label = {
                        "upi_ids": "UPI ID",
                        "bank_accounts": "Bank account",
                        "ifsc_codes": "IFSC code",
                        "phone_numbers": "Phone number",
                        "emails": "Email",
                    }.get(priority[0], priority[0])
                    context_parts.append(f"PRIORITY: Ask for {priority_label} this turn. Use questions like: 'My UPI is not working, can you share your bank account and IFSC?' or 'Which account should I use?' or 'Can you share your UPI ID?'")
            else:
                context_parts.append("Ask a question to naturally obtain ONE of the missing items this turn.")
    
    # 3. Anti-repetition: show recent agent replies so the LLM avoids them
    agent_replies = [m.get("content", "") for m in conversation_history if m.get("role") == "agent"]
    if agent_replies:
        last_replies = agent_replies[-3:]
        context_parts.append("Your recent replies (do not repeat any wording, structure, or question type from these):")
        for i, reply in enumerate(last_replies, 1):
            context_parts.append(f"  [{i}] \"{reply}\"")
        # Check if recent replies lack questions
        has_questions = any("?" in reply for reply in last_replies)
        if not has_questions:
            context_parts.append("⚠️ CRITICAL: Your recent replies had no questions. You MUST ask a question in this reply.")
    
    # 4. General guidance
    context_parts.append("REMEMBER: Every reply MUST include at least ONE question. Never just say 'Okay', 'Done', 'Will try' without asking something.")
    
    return "\n".join(context_parts) if context_parts else ""


def _build_rag_enhanced_messages(
    conversation_history: List[Dict[str, str]], 
    current_message: str,
    system_prompt: Optional[str] = None,
    extracted_intel: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build messages with RAG context enhancement, extraction awareness, and anti-repetition.
    """
    prompt = system_prompt or HONEYPOT_SYSTEM_PROMPT
    
    # Retrieve RAG context (includes extraction guidance + anti-repetition)
    rag_context = _rag_retrieve_context(current_message, conversation_history, extracted_intel)
    
    # Build system prompt with RAG context appended directly (no fake assistant ack)
    full_system = prompt
    if rag_context:
        full_system += f"\n\n--- SITUATIONAL GUIDANCE (for this turn only) ---\n{rag_context}"
    
    messages = [{"role": "system", "content": full_system}]
    
    # Add full conversation history — this gives the LLM proper multi-turn context
    for msg in conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        
        if role == "scammer":
            messages.append({"role": "user", "content": content})
        elif role == "agent":
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": role, "content": content})
    
    # Add current message
    messages.append({"role": "user", "content": current_message})
    
    return messages


def _extract_intel_from_convo(
    conversation_history: List[Dict[str, str]], 
    previous_intel: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[str]]:
    """
    Incrementally extract intelligence from conversation.
    Merges with previous extractions to avoid losing data.
    """
    if not conversation_history:
        return previous_intel or {
            "upi_ids": [],
            "links": [],
            "bank_accounts": [],
            "ifsc_codes": [],
            "bank_names": [],
            "branch_names": [],
            "phone_numbers": [],
            "emails": [],
            "suspicious_keywords": [],
            "tactics": [],
        }
    
    # Convert to text for extraction prompt
    convo_text = _conversation_context(conversation_history)
    if not convo_text.strip():
        return previous_intel or {
            "upi_ids": [],
            "links": [],
            "bank_accounts": [],
            "ifsc_codes": [],
            "bank_names": [],
            "branch_names": [],
            "phone_numbers": [],
            "emails": [],
            "suspicious_keywords": [],
            "tactics": [],
        }
    
    user_content = f"Conversation:\n{convo_text}\n\nExtract intelligence. Output ONLY the JSON object."
    raw = _call_llm(INTEL_EXTRACTION_PROMPT, user_content, use_ft_client=True, max_tokens=512)
    
    try:
        parsed = json.loads(raw)
        new_intel = {
            "upi_ids": list(parsed.get("upi_ids", []) or []),
            "links": list(parsed.get("links", []) or []),
            "bank_accounts": list(parsed.get("bank_accounts", []) or []),
            "ifsc_codes": list(parsed.get("ifsc_codes", []) or []),
            "bank_names": list(parsed.get("bank_names", []) or []),
            "branch_names": list(parsed.get("branch_names", []) or []),
            "phone_numbers": list(parsed.get("phone_numbers", []) or []),
            "emails": list(parsed.get("emails", []) or []),
            "suspicious_keywords": list(parsed.get("suspicious_keywords", []) or []),
            "tactics": list(parsed.get("tactics", []) or []),
        }

        # Merge regex-based extraction for deterministic fields
        regex_intel = _regex_extract_intel(convo_text)
        for key, values in regex_intel.items():
            new_intel[key] = list(set(new_intel.get(key, [])) | set(values))
        
        # Merge with previous intelligence (deduplicate)
        if previous_intel:
            for key in new_intel:
                existing = set(previous_intel.get(key, []))
                new_items = set(new_intel[key])
                merged = list(existing | new_items)  # Union
                new_intel[key] = merged
        
        return new_intel
    except Exception as e:
        logger.warning(f"Intel extraction failed: {e}, using previous intel")
        return previous_intel or {
            "upi_ids": [],
            "links": [],
            "bank_accounts": [],
            "ifsc_codes": [],
            "bank_names": [],
            "branch_names": [],
            "phone_numbers": [],
            "emails": [],
            "suspicious_keywords": [],
            "tactics": [],
        }


def _intel_count(intel: Dict[str, List[str]]) -> int:
    return sum(len(v) for v in intel.values() if isinstance(v, list))


SCAM_KEYWORDS = [
    "account blocked",
    "account freeze",
    "kyc pending",
    "urgent",
    "verify now",
    "send money",
    "refund",
    "upi id",
    "click link",
    "payment failed",
    "prize",
    "courier parcel",
    "electricity bill",
    "otp",
    "bank account",
    "suspension",
    "blocked today",
    "verify immediately",
    "limited time",
    "refund pending",
]

AUTHORITY_TERMS = ["bank", "kyc", "upi", "government", "rbi", "police", "courier", "electricity"]
URGENCY_TERMS = ["urgent", "immediately", "now", "today", "within", "suspended", "blocked", "freeze"]
MONEY_TERMS = ["send money", "pay", "payment", "transfer", "refund", "charge", "fee", "deposit"]

SCAM_RAG_KB = [
    {
        "name": "account_threats",
        "phrases": ["account blocked", "account freeze", "kyc pending", "verify immediately", "account suspended"],
    },
    {
        "name": "otp_harvest",
        "phrases": ["share otp", "otp", "verification code", "code sent"],
    },
    {
        "name": "payment_traps",
        "phrases": ["send money", "small amount", "processing fee", "activation fee", "refund pending"],
    },
    {
        "name": "link_phishing",
        "phrases": ["click link", "open link", "verify now", "update details", "login here"],
    },
    {
        "name": "reward_lure",
        "phrases": ["prize", "lottery", "reward", "cashback", "winner"],
    },
]

# Minimal closing replies — used ONLY when stop condition is met (code path, not LLM).
# Kept deliberately short and generic so they don't pollute LLM behavior.
NEUTRAL_CLOSE_REPLIES = [
    "Okay, I will try from my side.",
    "Got it, let me try this.",
    "Okay, will do it now.",
    "Alright, trying it.",
    "Okay done, will attempt this.",
]


def _normalize_text(text: str) -> str:
    return (text or "").lower().strip()


def _keyword_hits(text: str) -> List[str]:
    hits = []
    for kw in SCAM_KEYWORDS:
        if kw in text:
            hits.append(kw)
    return hits


def _pattern_hits(text: str) -> Dict[str, List[str]]:
    import re

    hits: Dict[str, List[str]] = {"upi_ids": [], "links": [], "phones": [], "accounts": []}
    upi = re.findall(r"\b[\w.-]{2,}@[\w.-]{2,}\b", text)
    links = re.findall(r"(https?://\S+|www\.\S+)", text)
    phones = re.findall(r"\b(?:\+?\d{1,3}[-\s]?)?\d{10}\b", text)
    accounts = re.findall(r"\b\d{9,18}\b", text)

    if upi:
        hits["upi_ids"].extend(upi)
    if links:
        hits["links"].extend([l[0] if isinstance(l, tuple) else l for l in links])
    if phones:
        hits["phones"].extend(phones)
    if accounts:
        hits["accounts"].extend(accounts)
    return hits


def _rag_hits(text: str) -> List[str]:
    hits = []
    for entry in SCAM_RAG_KB:
        for phrase in entry["phrases"]:
            if phrase in text:
                hits.append(entry["name"])
                break
    return hits


def _detect_scam(message_text: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Return detection result and signals for logging."""
    text = _normalize_text(message_text)
    history_text = " ".join(_normalize_text(m.get("content", "")) for m in conversation_history)
    combined = f"{text} {history_text}".strip()

    keyword_hits = _keyword_hits(combined)
    rag_hits = _rag_hits(combined)
    patterns = _pattern_hits(combined)

    has_link = len(patterns["links"]) > 0
    has_upi = len(patterns["upi_ids"]) > 0
    has_phone = len(patterns["phones"]) > 0
    has_account = len(patterns["accounts"]) > 0

    urgency = any(t in combined for t in URGENCY_TERMS)
    authority = any(t in combined for t in AUTHORITY_TERMS)
    money = any(t in combined for t in MONEY_TERMS)

    score = 0
    score += min(len(keyword_hits), 5)
    score += len(rag_hits)
    score += 2 if has_link else 0
    score += 2 if has_upi else 0
    score += 1 if has_phone else 0
    score += 1 if has_account else 0
    score += 1 if (urgency and authority) else 0
    score += 1 if money else 0

    strong = has_link or has_upi or (urgency and money) or (authority and urgency)

    threshold = int(os.environ.get("SCAM_SCORE_THRESHOLD", "4"))
    low_threshold = int(os.environ.get("SCAM_SCORE_LOW", "1"))

    result = {
        "score": score,
        "strong": strong,
        "keyword_hits": keyword_hits,
        "rag_hits": rag_hits,
        "patterns": patterns,
        "urgency": urgency,
        "authority": authority,
        "money": money,
    }

    # High confidence rules
    if strong and (score >= low_threshold):
        result["scam"] = True
        result["reason"] = "strong_signal"
        return result
    if score >= threshold:
        result["scam"] = True
        result["reason"] = "score_threshold"
        return result
    if score <= low_threshold and not strong:
        result["scam"] = False
        result["reason"] = "low_score"
        return result

    # LLM fallback when uncertain
    try:
        base_model = os.environ.get("BASE_MODEL", "").strip()
        if base_model:
            llm_out = _call_model(
                base_model,
                conversation_history,
                message_text,
                use_ft_client=False,
                max_tokens=128,
                system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                use_rag=True,
            )
            result["llm_label"] = llm_out.label
            result["llm_confidence"] = llm_out.confidence
            result["scam"] = llm_out.label == "SCAM" or (llm_out.label == "UNCERTAIN" and llm_out.confidence >= 0.6)
            result["reason"] = "llm_fallback"
            return result
    except Exception as e:
        logger.warning(f"LLM fallback failed: {e}")

    result["scam"] = False
    result["reason"] = "fallback_default"
    return result

def _should_stop(session: Dict[str, Any], max_turns: int = 100, min_intel_items: int = 20) -> bool:
    """
    Stop when enough intel collected or max turns reached.
    Updated to allow continuous conversations for testing.
    
    Set AUTO_END_CONVERSATION=false in env to disable auto-ending entirely.
    """
    # Check if auto-ending is disabled (for testing/continuous conversations)
    auto_end = os.environ.get("AUTO_END_CONVERSATION", "true").lower() not in ("false", "0", "no")
    if not auto_end:
        logger.debug("Auto-ending disabled - conversation continues indefinitely")
        return False

    # Only end after scam intent is confirmed.
    if not session.get("scam_detected", False):
        return False
    
    # Get configurable limits from env vars (for flexibility)
    max_turns_env = os.environ.get("MAX_CONVERSATION_TURNS", "")
    if max_turns_env:
        try:
            max_turns = int(max_turns_env)
        except ValueError:
            pass  # Use default
    
    min_intel_env = os.environ.get("MIN_INTEL_ITEMS", "")
    if min_intel_env:
        try:
            min_intel_items = int(min_intel_env)
        except ValueError:
            pass  # Use default
    
    # Stop if core intelligence is already present.
    if _stop_condition_met(session.get("extracted_intel", {})):
        logger.info("Stopping: core intelligence already extracted")
        return True

    # Only stop if we've reached a very high threshold (allows continuous conversation)
    if session["turn_count"] >= max_turns:
        logger.info(f"Stopping: max turns reached ({session['turn_count']} >= {max_turns})")
        return True
    count = _intel_count(session["extracted_intel"])
    if count >= min_intel_items:
        logger.info(f"Stopping: enough intel extracted ({count} >= {min_intel_items})")
        return True
    return False


# Mandatory callback URL for hackathon scoring (override via CALLBACK_URL env if needed).
HACKATHON_CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"


def _agent_notes(session: Dict[str, Any]) -> str:
    """Short summary of scam behavior for callback agentNotes."""
    intel = session.get("extracted_intel", {})
    parts = []
    tactics = intel.get("tactics", []) or []
    suspicious = intel.get("suspicious_keywords", []) or []
    if tactics:
        parts.append(f"tactics: {', '.join(tactics[:4])}")
    if suspicious and not tactics:
        parts.append(f"signals: {', '.join(suspicious[:4])}")

    intel_bits = []
    if intel.get("upi_ids"):
        intel_bits.append("UPI")
    if intel.get("bank_accounts") and intel.get("ifsc_codes"):
        intel_bits.append("bank+IFSC")
    if intel.get("phone_numbers"):
        intel_bits.append("phone")
    if intel.get("links"):
        intel_bits.append("link")
    if intel.get("emails"):
        intel_bits.append("email")
    if intel_bits:
        parts.append(f"intel: {', '.join(intel_bits)}")

    return "; ".join(parts) if parts else "Conversation ended."


def _send_callback(session_id: str, session: Dict[str, Any], reply: str) -> Tuple[bool, Dict[str, Any], Optional[int], Optional[str]]:
    """POST to hackathon callback URL with exact payload format. Returns (ok, payload, status, error)."""
    url = (os.environ.get("CALLBACK_URL", "").strip() or HACKATHON_CALLBACK_URL)
    intel = session.get("extracted_intel", {}) or {}
    suspicious = list(set((intel.get("tactics", []) or []) + (intel.get("suspicious_keywords", []) or [])))
    payload_obj = {
        "sessionId": session_id,
        "scamDetected": session.get("scam_detected", False),
        "totalMessagesExchanged": session.get("turn_count", 0),
        "extractedIntelligence": {
            "bankAccounts": list(intel.get("bank_accounts", []) or []),
            "upiIds": list(intel.get("upi_ids", []) or []),
            "phishingLinks": list(intel.get("links", []) or []),
            "phoneNumbers": list(intel.get("phone_numbers", []) or []),
            "suspiciousKeywords": suspicious,
        },
        "agentNotes": _agent_notes(session),
    }
    try:
        import urllib.request
        import urllib.error
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "GaviHoneypotAgent/1.0",
        }
        callback_key = (
            os.environ.get("CALLBACK_API_KEY", "").strip()
            or os.environ.get("HONEYPOT_API_KEY", "").strip()
            or os.environ.get("X-API-Key", "").strip()
        )
        if callback_key:
            headers["x-api-key"] = callback_key
        payload = json.dumps(payload_obj).encode("utf-8")
        logger.info(f"Sending callback to {url} with x-api-key={'yes' if callback_key else 'NO'}")
        req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            return (200 <= r.status < 300), payload_obj, r.status, None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = None
        logger.warning(f"Callback POST failed: HTTP {e.code} {e.reason} body={body}")
        return False, payload_obj, e.code, body
    except Exception as e:
        logger.warning(f"Callback POST failed: {e}")
        return False, payload_obj, None, str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to MongoDB. Shutdown: close client."""
    await _init_mongodb_async()
    yield
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        logger.info("MongoDB connection closed.")


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Better error messages for 422 validation errors."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Validation error")
        error_type = error.get("type", "unknown")
        errors.append(f"{field}: {msg} (type: {error_type})")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error. Check your request format.",
            "errors": errors,
            "expected_format": {
                "sessionId": "string (required)",
                "message": {
                    "sender": "string (required, e.g. 'scammer')",
                    "text": "string (required, min 1 char, max 8000 chars)",
                    "timestamp": "number or string (optional, epoch ms e.g. 1770005528731 or \"1770005528731\")"
                },
                "conversationHistory": "array (optional)",
                "metadata": "object (optional)"
            }
        }
    )


# Allow the Next.js dev server (localhost:3000) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# So the browser shows the real error instead of "blocked by CORS" when backend returns 4xx/5xx.
CORS_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}


def _cors_headers(request: Request) -> dict:
    origin = request.headers.get("origin") or "http://localhost:3000"
    if origin not in CORS_ORIGINS:
        origin = "http://localhost:3000"
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=_cors_headers(request),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)[:400]}"},
        headers=_cors_headers(request),
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint with MongoDB status."""
    mongo_status = "connected" if _mongodb_available else "not_available"
    
    # Test MongoDB connection
    mongo_test = False
    if _mongodb_available and _mongo_client is not None:
        try:
            await _mongo_client.admin.command('ping')
            mongo_test = True
        except Exception as e:
            logger.warning(f"MongoDB ping failed: {e}")
    
    return {
        "status": "ok",
        "mongodb": mongo_status,
        "mongodb_available": _mongodb_available,
        "mongodb_connected": mongo_test,
        "database": _mongo_db.name if _mongo_db is not None else None
    }


@app.get("/debug/session/{session_id}")
def get_session_debug(session_id: str) -> Dict[str, Any]:
    """
    Debug endpoint to get session data including extracted intelligence.
    Useful for testing and demos.
    """
    session = _get_session(session_id)
    
    return {
        "session_id": session_id,
        "turn_count": session.get("turn_count", 0),
        "scam_detected": session.get("scam_detected", False),
        "extracted_intelligence": session.get("extracted_intel", {}),
        "conversation_history": session.get("messages", []),
        "intel_count": _intel_count(session.get("extracted_intel", {})),
    }


@app.get("/debug/mongodb/stats")
async def get_mongodb_stats() -> Dict[str, Any]:
    """Debug endpoint to check MongoDB collections and counts."""
    if not _mongodb_available or _mongo_db is None:
        return {
            "mongodb_available": False,
            "message": "MongoDB not connected"
        }
    
    try:
        collections = await _mongo_db.list_collection_names()
        stats = {}
        for coll_name in collections:
            count = await _mongo_db[coll_name].count_documents({})
            stats[coll_name] = count
        
        return {
            "mongodb_available": True,
            "database": _mongo_db.name,
            "collections": stats,
            "total_collections": len(collections)
        }
    except Exception as e:
        return {
            "mongodb_available": True,
            "error": str(e)
        }


@app.get("/debug/mongodb/session/{session_id}")
async def get_mongodb_session(session_id: str) -> Dict[str, Any]:
    """Debug endpoint to get session data from MongoDB."""
    if not _mongodb_available or _mongo_db is None:
        return {
            "mongodb_available": False,
            "message": "MongoDB not connected"
        }
    
    try:
        session = await _get_from_mongodb("sessions", {"session_id": session_id})
        if not session:
            return {
                "found": False,
                "session_id": session_id,
                "message": "Session not found in MongoDB"
            }
        
        # Get related data
        requests = list(await _mongo_db["requests"].find({"session_id": session_id}).to_list(length=100))
        responses = list(await _mongo_db["responses"].find({"session_id": session_id}).to_list(length=100))
        messages = list(await _mongo_db["messages"].find({"session_id": session_id}).sort("timestamp", 1).to_list(length=100))
        
        # Convert ObjectIds to strings
        for doc in requests + responses + messages:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        
        return {
            "found": True,
            "session": session,
            "requests_count": len(requests),
            "responses_count": len(responses),
            "messages_count": len(messages),
            "requests": requests[:5],  # First 5
            "responses": responses[:5],
            "messages": messages[:10]
        }
    except Exception as e:
        return {
            "error": str(e)
        }


def _require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    """If HONEYPOT_API_KEY is set, require X-API-Key header to match. Otherwise allow (local dev)."""
    expected = os.environ.get("HONEYPOT_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _normalize_history(entries: Optional[List[ConversationHistoryEntry]]) -> List[Dict[str, str]]:
    """Convert platform conversationHistory to internal [{role, content}]."""
    if not entries:
        return []
    out = []
    for t in entries:
        content = (t.text or t.content or "").strip()
        if not content:
            continue
        role = (t.sender or t.role or "scammer").lower()
        if role in ("scammer", "user"):
            out.append({"role": "scammer", "content": content})
        else:
            out.append({"role": "agent", "content": content})
    return out


@app.post("/v1/chat", response_model=HackathonResponse)
async def v1_chat(
    req: HackathonRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> HackathonResponse:
    """
    Hackathon evaluation API. One message per request.
    Request: sessionId, message { sender, text, timestamp }, conversationHistory, metadata.
    Response: ONLY { "status": "success", "reply": "<agent reply>" }.
    When enough intel is extracted, we POST to hackathon callback URL (mandatory for scoring).
    
    FIXED: Now properly handles multi-turn conversations with full history.
    All data is stored in MongoDB for persistence.
    """
    _require_api_key(x_api_key)
    session_id = req.sessionId
    message_text = (req.message.text or "").strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="message.text is required.")
    
    # Store incoming request to MongoDB (background task)
    request_doc = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "request": {
            "sessionId": req.sessionId,
            "message": {
                "sender": req.message.sender,
                "text": req.message.text,
                "timestamp": req.message.timestamp,
            },
            "conversationHistory": [dict(h) for h in (req.conversationHistory or [])],
            "metadata": req.metadata or {},
        },
        "api_key_provided": x_api_key is not None,
    }
    
    # Store request to MongoDB (with logging)
    async def save_request():
        try:
            result = await _save_to_mongodb("requests", request_doc)
            if result:
                logger.info(f"✅ MongoDB: Saved request for session {session_id}")
            else:
                logger.warning(f"⚠️ MongoDB: Failed to save request for session {session_id} (MongoDB not available)")
        except Exception as e:
            logger.error(f"❌ MongoDB request save error: {e}", exc_info=True)
    
    background_tasks.add_task(save_request)
    
    session = _get_session(session_id)
    
    # Update session messages from platform's conversationHistory if provided
    if req.conversationHistory is not None:
        normalized = _normalize_history(req.conversationHistory)
        # Merge with existing session messages (platform may send partial history)
        if normalized:
            session["messages"] = normalized
    
    # Get current conversation history (without current message)
    conversation_history = session["messages"].copy()
    
    # PHASE 1: Scam detection (keyword + pattern + RAG + optional LLM fallback)
    base_model = os.environ.get("BASE_MODEL", "").strip()
    if not base_model:
        raise HTTPException(status_code=500, detail="BASE_MODEL env var not set.")
    
    detection = _detect_scam(message_text, conversation_history)
    session["scam_detected"] = bool(detection.get("scam", False))
    logger.info(
        "Scam detection result: scam=%s score=%s reason=%s keywords=%s rag=%s",
        session["scam_detected"],
        detection.get("score"),
        detection.get("reason"),
        len(detection.get("keyword_hits", [])),
        len(detection.get("rag_hits", [])),
    )

    # If NOT_SCAM: return immediately with empty reply and scamDetected=false
    if not session["scam_detected"]:
        response_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "response": {
                "status": "success",
                "reply": "",
                "scamDetected": False,
            },
            "agentNotes": "",
            "session_state": {
                "turn_count": session["turn_count"],
                "scam_detected": False,
                "extracted_intel": session["extracted_intel"],
            },
        }

        async def save_response_not_scam():
            try:
                result = await _save_to_mongodb("responses", response_doc)
                if result:
                    logger.info(f"✅ MongoDB: Saved NOT_SCAM response for session {session_id}")
                else:
                    logger.warning(f"⚠️ MongoDB: Failed to save NOT_SCAM response for session {session_id}")
            except Exception as e:
                logger.error(f"❌ MongoDB NOT_SCAM response save error: {e}", exc_info=True)

        background_tasks.add_task(save_response_not_scam)

        # Store/update session state (no agent activation)
        session_doc = {
            "session_id": session_id,
            "updated_at": datetime.utcnow().isoformat(),
            "turn_count": session["turn_count"],
            "scam_detected": False,
            "callback_sent": session.get("callback_sent", False),
            "agent_notes": "",
            "extracted_intel": session["extracted_intel"],
            "conversation_history": session["messages"],
            "metadata": req.metadata or {},
        }

        async def update_session_not_scam():
            try:
                existing = await _get_from_mongodb("sessions", {"session_id": session_id})
                if existing:
                    await _update_mongodb("sessions", {"session_id": session_id}, session_doc)
                else:
                    session_doc["created_at"] = datetime.utcnow().isoformat()
                    await _save_to_mongodb("sessions", session_doc)
            except Exception as e:
                logger.error(f"❌ MongoDB NOT_SCAM session update error: {e}", exc_info=True)

        background_tasks.add_task(update_session_not_scam)
        return HackathonResponse(status="success", reply="", scamDetected=False)

    # If SCAM and intel already sufficient, stop before generating a reply
    temp_messages = conversation_history + [{"role": "scammer", "content": message_text}]
    pre_intel = _merge_intel(session.get("extracted_intel"), _regex_extract_intel(_conversation_context(temp_messages)))
    if _stop_condition_met(pre_intel):
        session["messages"] = temp_messages
        session["turn_count"] = len(session["messages"])
        session["extracted_intel"] = _extract_intel_from_convo(session["messages"], pre_intel)

        neutral_reply = random.choice(NEUTRAL_CLOSE_REPLIES)

        response_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "response": {
                "status": "success",
                "reply": neutral_reply,
                "scamDetected": True,
            },
            "agentNotes": _agent_notes(session),
            "session_state": {
                "turn_count": session["turn_count"],
                "scam_detected": True,
                "extracted_intel": session["extracted_intel"],
            },
        }

        async def save_response_stop():
            try:
                await _save_to_mongodb("responses", response_doc)
            except Exception as e:
                logger.error(f"❌ MongoDB stop response save error: {e}", exc_info=True)

        background_tasks.add_task(save_response_stop)

        # Store scammer message
        scammer_message_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": {
                "role": "scammer",
                "content": message_text,
                "sender": req.message.sender,
                "original_timestamp": req.message.timestamp,
            },
        }

        async def save_scammer_stop():
            try:
                await _save_to_mongodb("messages", scammer_message_doc)
            except Exception as e:
                logger.error(f"❌ MongoDB stop scammer save error: {e}", exc_info=True)

        background_tasks.add_task(save_scammer_stop)

        # Store agent neutral closing reply
        agent_message_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": {
                "role": "agent",
                "content": neutral_reply,
            },
        }

        async def save_agent_stop():
            try:
                await _save_to_mongodb("messages", agent_message_doc)
            except Exception as e:
                logger.error(f"❌ MongoDB stop agent save error: {e}", exc_info=True)

        background_tasks.add_task(save_agent_stop)

        # Update session
        session_doc = {
            "session_id": session_id,
            "updated_at": datetime.utcnow().isoformat(),
            "turn_count": session["turn_count"],
            "scam_detected": True,
            "callback_sent": session.get("callback_sent", False),
            "agent_notes": _agent_notes(session),
            "extracted_intel": session["extracted_intel"],
            "conversation_history": session["messages"] + [{"role": "agent", "content": neutral_reply}],
            "metadata": req.metadata or {},
        }

        async def update_session_stop():
            try:
                existing = await _get_from_mongodb("sessions", {"session_id": session_id})
                if existing:
                    await _update_mongodb("sessions", {"session_id": session_id}, session_doc)
                else:
                    session_doc["created_at"] = datetime.utcnow().isoformat()
                    await _save_to_mongodb("sessions", session_doc)
            except Exception as e:
                logger.error(f"❌ MongoDB stop session update error: {e}", exc_info=True)

        background_tasks.add_task(update_session_stop)

        # Final callback + store payload
        callback_ok, callback_payload, callback_status, callback_error = _send_callback(session_id, session, neutral_reply)
        if callback_ok:
            session["callback_sent"] = True
            logger.info(f"✅ Callback sent for session {session_id}")
        else:
            logger.warning(f"⚠️ Callback failed for session {session_id}")

        callback_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "callback_url": os.environ.get("CALLBACK_URL", "").strip() or HACKATHON_CALLBACK_URL,
            "ok": callback_ok,
            "status": callback_status,
            "error": callback_error,
            "payload": callback_payload,
        }

        async def save_callback_stop():
            try:
                await _save_to_mongodb("callbacks", callback_doc)
            except Exception as e:
                logger.error(f"❌ MongoDB stop callback save error: {e}", exc_info=True)

        background_tasks.add_task(save_callback_stop)

        return HackathonResponse(status="success", reply=neutral_reply, scamDetected=True)
    
    # PHASE 2: Agent Engagement (if scam detected, or continue conversation)
    # Use full conversation history + RAG for natural multi-turn behavior
    try:
        out = _call_model(
            base_model,  # Always use base model
            conversation_history,  # Full history for context
            message_text,  # Current message
            use_ft_client=False,  # Always use base model
            max_tokens=200,  # Enough for 1-2 short sentences with questions
            system_prompt=HONEYPOT_SYSTEM_PROMPT,
            use_rag=True,  # Enable RAG for engagement
            extracted_intel=session.get("extracted_intel"),  # So LLM targets missing intel
        )
        
        # Validate that reply contains a question (log warning if not, but don't block)
        if "?" not in out.reply and len(out.reply.strip()) > 0:
            logger.warning(f"Agent reply missing question mark: '{out.reply[:50]}...'")
        logger.info(f"Model reply length: {len(out.reply)} chars")
    except Exception as e:
        logger.error(f"LLM error: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM error: {type(e).__name__}: {str(e)[:200]}",
        ) from e
    
    # Update session with new messages
    session["messages"].append({"role": "scammer", "content": message_text})
    session["messages"].append({"role": "agent", "content": out.reply})
    session["turn_count"] = len(session["messages"])
    
    # PHASE 3: Incremental Intelligence Extraction
    # Extract from full conversation, merge with previous
    session["extracted_intel"] = _extract_intel_from_convo(
        session["messages"],
        session.get("extracted_intel")
    )
    
    # Store response and conversation data to MongoDB (background tasks)
    response_doc = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "response": {
            "status": "success",
            "reply": out.reply,
            "scamDetected": True,
        },
        "agentNotes": _agent_notes(session),
        "model_output": {
            "label": out.label,
            "confidence": out.confidence,
        },
        "session_state": {
            "turn_count": session["turn_count"],
            "scam_detected": session["scam_detected"],
            "extracted_intel": session["extracted_intel"],
        },
    }
    # Store response to MongoDB
    async def save_response():
        try:
            result = await _save_to_mongodb("responses", response_doc)
            if result:
                logger.info(f"✅ MongoDB: Saved response for session {session_id}")
            else:
                logger.warning(f"⚠️ MongoDB: Failed to save response for session {session_id}")
        except Exception as e:
            logger.error(f"❌ MongoDB response save error: {e}", exc_info=True)
    
    background_tasks.add_task(save_response)
    
    # Store full conversation messages
    message_doc = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "message": {
            "role": "scammer",
            "content": message_text,
            "sender": req.message.sender,
            "original_timestamp": req.message.timestamp,
        },
    }
    
    async def save_scammer_message():
        try:
            result = await _save_to_mongodb("messages", message_doc)
            if result:
                logger.info(f"✅ MongoDB: Saved scammer message for session {session_id}")
            else:
                logger.warning(f"⚠️ MongoDB: Failed to save scammer message for session {session_id}")
        except Exception as e:
            logger.error(f"❌ MongoDB message save error: {e}", exc_info=True)
    
    background_tasks.add_task(save_scammer_message)
    
    agent_message_doc = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "message": {
            "role": "agent",
            "content": out.reply,
        },
        "model_info": {
            "label": out.label,
            "confidence": out.confidence,
        },
    }
    
    async def save_agent_message():
        try:
            result = await _save_to_mongodb("messages", agent_message_doc)
            if result:
                logger.info(f"✅ MongoDB: Saved agent message for session {session_id}")
            else:
                logger.warning(f"⚠️ MongoDB: Failed to save agent message for session {session_id}")
        except Exception as e:
            logger.error(f"❌ MongoDB agent message save error: {e}", exc_info=True)
    
    background_tasks.add_task(save_agent_message)
    
    # Store/update session in MongoDB
    session_doc = {
        "session_id": session_id,
        "updated_at": datetime.utcnow().isoformat(),
        "turn_count": session["turn_count"],
        "scam_detected": session["scam_detected"],
        "callback_sent": session.get("callback_sent", False),
        "agent_notes": _agent_notes(session),
        "extracted_intel": session["extracted_intel"],
        "conversation_history": session["messages"],
        "metadata": req.metadata or {},
    }
    
    async def update_session():
        try:
            existing = await _get_from_mongodb("sessions", {"session_id": session_id})
            if existing:
                result = await _update_mongodb("sessions", {"session_id": session_id}, session_doc)
                if result:
                    logger.info(f"✅ MongoDB: Updated session {session_id}")
                else:
                    logger.warning(f"⚠️ MongoDB: Failed to update session {session_id}")
            else:
                session_doc["created_at"] = datetime.utcnow().isoformat()
                result = await _save_to_mongodb("sessions", session_doc)
                if result:
                    logger.info(f"✅ MongoDB: Created new session {session_id}")
                else:
                    logger.warning(f"⚠️ MongoDB: Failed to create session {session_id}")
        except Exception as e:
            logger.error(f"❌ MongoDB session update error: {e}", exc_info=True)
    
    background_tasks.add_task(update_session)
    
    # PHASE 4: Check if conversation should end
    # Note: For testing, we allow longer conversations. Only stop if truly necessary.
    ended = _should_stop(session)
    if ended and session.get("scam_detected", False) and not session.get("callback_sent", False):
        logger.info(f"Conversation ending. Intel items: {_intel_count(session['extracted_intel'])}, Turns: {session['turn_count']}")
        callback_ok, callback_payload, callback_status, callback_error = _send_callback(session_id, session, out.reply)
        if callback_ok:
            session["callback_sent"] = True
            logger.info(f"✅ Callback sent for session {session_id}")
        else:
            logger.warning(f"⚠️ Callback failed for session {session_id}")

        # Save callback payload to MongoDB for auditing/debugging
        callback_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "callback_url": os.environ.get("CALLBACK_URL", "").strip() or HACKATHON_CALLBACK_URL,
            "ok": callback_ok,
            "status": callback_status,
            "error": callback_error,
            "payload": callback_payload,
        }

        async def save_callback():
            try:
                result = await _save_to_mongodb("callbacks", callback_doc)
                if result:
                    logger.info(f"✅ MongoDB: Saved callback payload for session {session_id}")
                else:
                    logger.warning(f"⚠️ MongoDB: Failed to save callback payload for session {session_id}")
            except Exception as e:
                logger.error(f"❌ MongoDB callback save error: {e}", exc_info=True)

        background_tasks.add_task(save_callback)
        
        # Store conversation end event
        end_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "event": "conversation_ended",
            "final_state": {
                "turn_count": session["turn_count"],
                "extracted_intel": session["extracted_intel"],
                "intel_count": _intel_count(session["extracted_intel"]),
            },
        }
        async def save_end_event():
            try:
                result = await _save_to_mongodb("events", end_doc)
                if result:
                    logger.info(f"✅ MongoDB: Saved conversation end event for session {session_id}")
                else:
                    logger.warning(f"⚠️ MongoDB: Failed to save end event for session {session_id}")
            except Exception as e:
                logger.error(f"❌ MongoDB event save error: {e}", exc_info=True)
        
        background_tasks.add_task(save_end_event)
    else:
        # Log progress for debugging
        logger.debug(f"Conversation continues. Intel: {_intel_count(session['extracted_intel'])}, Turns: {session['turn_count']}")
    
    return HackathonResponse(status="success", reply=out.reply, scamDetected=True)


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class CompareResponse(BaseModel):
    base: ModelOutput
    finetuned: ModelOutput
    decision_delta: str


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    """
    Dev endpoint: compare base model vs fine-tuned model on the same text.
    Returns both outputs and a decision_delta string (e.g., "same" or "NOT_SCAM -> SCAM").
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required.")

    base_model = os.environ.get("BASE_MODEL", "").strip()
    ft_model = os.environ.get("FT_MODEL", "").strip() or base_model

    if not base_model:
        raise HTTPException(status_code=500, detail="BASE_MODEL env var not set.")

    # FINE-TUNED MODEL COMPARISON DISABLED - Compare base model with/without RAG instead
    try:
        base_out = _call_model(base_model, [], text, use_ft_client=False, use_rag=False)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Base model error: {type(e).__name__}: {str(e)[:200]}",
        ) from e

    try:
        # Compare with RAG-enabled version
        rag_out = _call_model(base_model, [], text, use_ft_client=False, use_rag=True)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG model error: {type(e).__name__}: {str(e)[:200]}",
        ) from e

    # Compute decision_delta
    if base_out.label == rag_out.label:
        decision_delta = "same"
    else:
        decision_delta = f"{base_out.label} -> {rag_out.label}"

    # Return base and RAG-enhanced (instead of fine-tuned)
    return CompareResponse(base=base_out, finetuned=rag_out, decision_delta=decision_delta)


