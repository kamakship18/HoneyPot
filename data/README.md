# Training data for honeypot agent

## Reply pattern (required for engagement examples)

Every **assistant** reply in conversational scam engagement must follow this order:

1. **Question the situation** — Why? Which bank/app/company? From whom? (e.g. "Refund? I didn't request anything." "Blocked? I used my account yesterday.")
2. **Ask for clarification** — Official SMS? Receipt? Ticket number? Branch name? (e.g. "Order number bhej do." "Kaunsi department se ho aap?")
3. **Show confusion or hesitation** — Only then mention UPI/link/bank in a confused way (e.g. "I'm not sure where to find UPI ID." "UPI ID phir se bhej do, mujhe clear nahi dikha.")

**Do NOT:**
- Lead with compliance (e.g. "How do I find my UPI ID?" by itself).
- Jump straight to "UPI ID bhej do" or "Account number bhej do" without first questioning.
- Sound eager to comply before the scammer has explained more.

**Good example:**  
"Refund? I didn't request anything. Which bank is this from? I'm not sure where to find UPI ID."

**Bad example:**  
"How do I find my UPI ID?" (no questioning first)

## Files

- `scam_finetune_train.jsonl` — Training (classification + engagement + intel extraction + edge cases). Engagement rows use the 3-step pattern above.
- `scam_finetune_eval.jsonl` — Evaluation.

Each JSONL row: `{"messages": [{"role":"system"|"user"|"assistant", "content":"..."}, ...]}`. Engagement rows should have system message that states: "Reply order: (1) question, (2) clarify, (3) confusion. Do NOT lead with compliance."
